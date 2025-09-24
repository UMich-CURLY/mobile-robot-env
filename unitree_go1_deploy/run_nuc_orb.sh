#!/usr/bin/env bash
#
# create_tmux_sessions.sh
#
# This script creates (or restarts) five detached tmux sessions,
# each running one of the specified commands. Adjust session names if you like.

sudo ip route add 224.0.0.0/4 dev eno1 metric 80

# realsense (High‐accuracy preset)
tmux has-session -t realsense_high 2>/dev/null
if [ $? -ne 0 ]; then
  echo "Starting realsense (High‐accuracy preset)"
  tmux new-session -d -s realsense_high \
    ros2 launch realsense2_camera rs_launch.py \
      depth_module.profile:=640x480x30 \
      rgb_camera.profile:=640x480x30 \
      align_depth.enable:=true \
      serial_no:="'827312072741'" \
      json_file_path:="/home/curlynuc/ros2_ws/src/realsense-ros/realsense2_camera/launch/HighAccuracyPreset.json"
fi

# orb slam
tmux has-session -t orb_slam 2>/dev/null
if [ $? -ne 0 ]; then
  echo "Starting orb slam 3"
  tmux new-session -d -s orb_slam ros2 run orbslam3 realsense_direct ~/ORB-SLAM3-STEREO-FIXED/Vocabulary/ORBvoc.txt ~/ORB-SLAM3-STEREO-FIXED/Examples/Stereo-Inertial/RealSense_D435i.yaml 050422070068 0 

fi

# websocket server
tmux has-session -t ws_server 2>/dev/null
if [ $? -ne 0 ]; then
  echo "Starting WebSocket server"
  tmux new-session -d -s ws_server \
    python3 $HOME/mobile-robot-server/go1_server.py
fi

echo "Tmux sessions are now running (or have been restarted)."
tmux ls
echo "Use 'tmux ls' to list them and 'tmux attach -t <session_name>' to attach. Use 'tmux kill-server' to kill all sessions."