import glob
import pickle as pkl
import lcm
import sys

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from go1_gym_deploy.envs.actor_critic import *
from go1_gym_deploy.envs.estimator import *
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import *
from go1_gym_deploy.envs.depth_backbone import *
from copy import copy, deepcopy


import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # load agent
    dirs = glob.glob(f"../../runs/{label}/*")
    logdir = sorted(dirs)[0]


    cfg = None

    se = StateEstimator(lc)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=max_vel, yaw_scale=max_yaw_vel)

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)

    policy = load_policy(logdir, py_name = '/model_stable_walking.pt')

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    # 修改
    deployment_runner.add_policy(policy)
    # deployment_runner.add_policy(policy_recover)
    # deployment_runner.add_policy(policy2)

    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

# 修改
def load_policy(logdir, py_name = '/checkpoint/model_stair.pt'):
    print("load pt: ", py_name)


    num_actions = 12
    num_prop= 2 + 3 +3 +  3 + num_actions*3
    num_scan= 17*11
    num_priv_latent = 4 + 1 + num_actions * 2
    num_priv_explicit = 3 +3 +3
    num_hist = 10
    num_critic_obs = num_prop + num_scan + num_priv_latent + num_priv_explicit + num_prop * num_hist

    ac =  ActorCriticRMA(num_prop=num_prop,
                        num_scan=num_scan,
                        num_critic_obs=num_critic_obs,
                        num_priv_latent = num_priv_latent,
                        num_priv_explicit = num_priv_explicit,
                        num_hist = num_hist,
                        num_actions = num_actions,
                        scan_encoder_dims=[128, 64, 32],
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        priv_encoder_dims = [256, 128],
                        )
    estimator = Estimator(input_dim=47, output_dim=9, hidden_dims=[128, 64])

    loaded_dict = torch.load(logdir + py_name)

    estimator.load_state_dict(loaded_dict['estimator_state_dict'])
    ac.load_state_dict(loaded_dict['model_state_dict'])

    estimator.eval() 
    ac.eval()
    import os

    def policy(obs):
        obs_priv = estimator(obs[:,:num_prop].to('cpu'))
        obs[:,num_prop:num_prop+num_priv_explicit] = obs_priv

        
        actions = ac.act_inference(obs.to('cpu'), hist_encoding=True)

        return actions

    return policy



if __name__ == '__main__':
    label = "Lau"

    experiment_name = "go1_experiment"

    load_and_run_policy(label, experiment_name=experiment_name, max_vel=1, max_yaw_vel=1)
