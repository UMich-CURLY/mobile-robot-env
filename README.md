# mobile-robot-env
## Policy:
for nuc, ssh curlynuc@35.3.201.75
on nuc run:
sudo ip route add 224.0.0.0/4 dev eno1 metric 80 


then login to nx via: ssh unitree@192.168.123.15 (assuming you are connected to nuc) password 123
on nx run

sudo ip route add 224.0.0.0/4 dev eth0

then run on nx (you may need tmux)
```
cd ~/unitree_go1_deploy/unitree_legged_sdk/build && ./lcm_position
```
WARNING：the above script "hijacks" the low level control of the dog and exposes an lcm endpoint. 
it has to fight the built in unitree script to do so, so you may need to restart this script multiple times.

(for a principled way, log into the raspberry pi of the dog and shut down the unitree sport program. this is usually not worth your time.)

furthermore, the dog MUST be prone or else there may be danger.

if at any point you kill this script, the dog will instantly fall down due to the motors going slack.
```
cd $HOME/mobile-robot-server/unitree_go1_deploy/go1_deploy/go1_gym_deploy/scripts && /usr/bin/python3 deploy.py
```
this script enables the custom RL policy. it talks to the earlier script, and accepts movement commands from lcm. when it starts up, you will be prompted to press the "R2" button on the joystick controller to make the dog calibrate and stand up. (the dog should be prone and aligned at this point, if not you should restart everything)



## Robot Server
make sure you are in the ~/mobile-robot-env directory on the dog.

Simply run ~/mobile-robot-env/run_nuc_orb.sh to start everything on the dog.

<!-- ## Isaac Entry Point:
python3 isaac_provider.py --task=go2_matterport_vision --history_length=9 --load_run=2024-09-25_23-22-02 --episode_index 15 --enable_camera

## Starting the d435 node:
ros2 launch realsense2_camera rs_launch.py depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30 align_depth.enable:=true serial_no:="'827312072741'" json_file_path:="/home/curlynuc/ros2_ws/src/realsense-ros/realsense2_camera/launch/HighAccuracyPreset.json"

## Starting the ORB SLAM 3 node
note that the 0 means 
cd ~/ORB-SLAM3-STEREO-FIXED && ros2 run orbslam3 realsense_direct Vocabulary/ORBvoc.txt Examples/Stereo-Inertial/RealSense_D435i.yaml 050422070068 0  -->


