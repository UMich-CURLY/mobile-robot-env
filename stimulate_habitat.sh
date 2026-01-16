#!/bin/bash
# Publish sim sensor settings
ros2 topic pub /sim_settings std_msgs/msg/String "data: 'sensors: head_rgb_left head_rgb_right head_stereo_left head_stereo_right rear_rgb rear_depth'" --once
# sleep 0.3
# # Publish model selection
# ros2 topic pub /sim_settings std_msgs/msg/String "data: 'model: hab_spot'" --once
# sleep 0.3
# # Publish policy activation
# ros2 topic pub /sim_settings std_msgs/msg/String "data: 'policy: true'" --once
# sleep 0.3
# # Confirm settings
# ros2 topic pub /sim_settings std_msgs/msg/String "data: 'confirm settings'" --once

# original version
ros2 topic pub /sim_settings std_msgs/msg/String "data: 'name: spot, model: hab_spot, policy: false, confirm settings'" \
  --once --qos-durability transient_local --qos-reliability reliable