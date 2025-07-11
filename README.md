# mobile-robot-server

## Isaac Entry Point:
python3 isaac_provider.py --task=go2_matterport_vision --history_length=9 --load_run=2024-09-25_23-22-02 --episode_index 15 --enable_camera

## Starting the d435 node:
ros2 launch realsense2_camera rs_launch.py depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30 align_depth.enable:=true serial_no:="'827312072741'" json_file_path:="/home/curlynuc/ros2_ws/src/realsense-ros/realsense2_camera/launch/HighAccuracyPreset.json"

## Starting the ORB SLAM 3 node
note that the 0 means 
cd ~/ORB-SLAM3-STEREO-FIXED && ros2 run orbslam3 realsense_direct Vocabulary/ORBvoc.txt Examples/Stereo-Inertial/RealSense_D435i.yaml 050422070068 0 

## Policy:
on nuc run:
sudo ip route add 224.0.0.0/4 dev eno1 metric 80 