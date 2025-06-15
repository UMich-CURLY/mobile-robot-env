import copy
import time
import os

import numpy as np
import torch

from go1_gym_deploy.utils.logger import MultiLogger


class DeploymentRunner:
    def __init__(self, experiment_name="unnamed", se=None, log_root="."):
        self.agents = {}
        self.policy = None
        # 修改
        self.policy_extra = []
        self.command_profile = None
        self.logger = MultiLogger()
        self.se = se
        self.vision_server = None

        self.log_root = log_root
        self.init_log_filename()
        self.control_agent_name = None
        self.command_agent_name = None

        self.triggered_commands = {i: None for i in range(4)} # command profiles for each action button on the controller
        self.button_states = np.zeros(4)

        self.is_currently_probing = False
        self.is_currently_logging = [False, False, False, False]

    def init_log_filename(self):
        datetime = time.strftime("%Y/%m_%d/%H_%M_%S")

        for i in range(100):
            try:
                os.makedirs(f"{self.log_root}/{datetime}_{i}")
                self.log_filename = f"{self.log_root}/{datetime}_{i}/log.pkl"
                return
            except FileExistsError:
                continue


    def add_open_loop_agent(self, agent, name):
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_control_agent(self, agent, name):
        self.control_agent_name = name
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_vision_server(self, vision_server):
        self.vision_server = vision_server

    def set_command_agents(self, name):
        self.command_agent = name

    def add_policy(self, policy):
        # 修改
        if (self.policy == None):
            self.policy = policy
            self.policy_extra.append(policy)
        elif(self.policy != None):
            self.policy_extra.append(policy)


    def add_command_profile(self, command_profile):
        self.command_profile = command_profile


    def calibrate(self, wait=True):
        # first, if the robot is not in nominal pose, move slowly to the nominal pose
        for agent_name in self.agents.keys():
            if hasattr(self.agents[agent_name], "get_obs"):
                agent = self.agents[agent_name]
                agent.get_obs()
                joint_pos = agent.dof_pos
                final_goal = np.zeros(12)
                nominal_joint_pos = agent.default_dof_pos

                print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
                while wait:
                    self.button_states = self.command_profile.get_buttons()
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

                cal_action = np.zeros((agent.num_envs, agent.num_actions))
                target_sequence = []
                target = nominal_joint_pos - joint_pos
                while np.max(np.abs(target)) > 0.01:
                    target -= np.clip((target), -0.02, 0.02)
                    target_sequence += [copy.deepcopy(target)]
                for target in target_sequence:
                    action_scale=0.25
                    print("tar_pos:", -target + nominal_joint_pos)
                    cal_action[:, 0:12] = -target/action_scale
                    agent.step(torch.from_numpy(cal_action))
                    agent.get_obs()
                    print("dof_pos:", agent.dof_pos, "\n")
                    time.sleep(0.05)

                print("Starting pose calibrated [Press R2 to start controller]")
                while True:
                    self.button_states = self.command_profile.get_buttons()
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

                for agent_name in self.agents.keys():
                    obs = self.agents[agent_name].reset()
                    if agent_name == self.control_agent_name:
                        control_obs = obs

        return control_obs


    def run(self, num_log_steps=1000000000, max_steps=100000000, logging=True):
        assert self.control_agent_name is not None, "cannot deploy, runner has no control agent!"
        assert self.policy is not None, "cannot deploy, runner has no policy!"
        assert self.command_profile is not None, "cannot deploy, runner has no command profile!"

        # TODO: add basic test for comms

        for agent_name in self.agents.keys():
            obs = self.agents[agent_name].reset()
            if agent_name == self.control_agent_name:
                control_obs = obs

        control_obs = self.calibrate(wait=True)

        plot_action = []
        plot_steps = []

        # 修改
        policy_now = 0
        policy_use = self.policy_extra[policy_now]
        

        # now, run control loop
        try:
            for i in range(max_steps):

                # 修改
                # action = self.policy(control_obs)
                action = policy_use(control_obs)


                for agent_name in self.agents.keys():
                    obs, ret, done, info = self.agents[agent_name].step(action)

                    # info.update(policy_info)
                    # info.update({"observation": obs, "reward": ret, "done": done, "timestep": i,
                    #              "time": i * self.agents[self.control_agent_name].dt, "action": action, "rpy": self.agents[self.control_agent_name].se.get_rpy(), "torques": self.agents[self.control_agent_name].torques})
                    # info.update({"vel_x": vel[0, 0], "yaw_z" : vel[0, 1]})

                    # if logging: self.logger.log(agent_name, info)

                    if agent_name == self.control_agent_name:
                        control_obs, control_ret, control_done, control_info = obs, ret, done, info

                # bad orientation emergency stop  todo
                rpy = self.agents[self.control_agent_name].se.get_rpy()
                # if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                #     self.calibrate(wait=False, low=True)

                # check for logging command
                prev_button_states = self.button_states[:]
                self.button_states = self.command_profile.get_buttons()

                # stop running
                if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                    control_obs = self.calibrate(wait=False)
                    time.sleep(1)
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                    # self.button_states = self.command_profile.get_buttons()
                    while not self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        time.sleep(0.01)
                        # self.button_states = self.command_profile.get_buttons()
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False

                # 修改
                # print(vars(self.command_profile.state_estimator))

                if self.command_profile.state_estimator.right_upper_switch:
                    print("switch model")
                    time.sleep(0.4)
                    policy_num = len(self.policy_extra)
                    policy_now += 1
                    if policy_now == policy_num:
                        policy_now  = 0
                    print("policy_mode_now is:",policy_now)
                    policy_use = self.policy_extra[policy_now]

            # finally, return to the nominal pose
            control_obs = self.calibrate(wait=False)
            # self.logger.save(self.log_filename)

        except KeyboardInterrupt:
            no=1
        #     self.logger.save(self.log_filename)
