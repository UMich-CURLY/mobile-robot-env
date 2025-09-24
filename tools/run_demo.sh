python isaac_lab_server_spot_json.py \
--enable_cameras \
--scene_path "/home/junzhewu/data/isaac_scenes_v1/nvidia_edit/park/park_morning_edit.usd" \
--plan_json /home/junzhewu/pohsun/SG-VLN/robot_env/test_jsons/nvidia.json --use_plan \
--waypoint_stride 1 --arrive_thresh 0.15 --max_v 0.5 --max_yaw_rate 1.2 --k_p_ang 2.0 --base_height 0.6

python /home/junzhewu/pohsun/SG-VLN/robot_env/fix_material.py --usd_path ~/data/isaac_scenes_v1/nvidia_edit/park/park_morning.usd

python isaac_lab_server_spot_json.py \
--enable_cameras \
--scene_path "/home/junzhewu/data/isaac_scenes_v1/DETROIT_sky/DETROIT_sky.usd" \
--plan_json /home/junzhewu/pohsun/SG-VLN/robot_env/test_jsons/detroit.json --use_plan \
--waypoint_stride 1 --arrive_thresh 0.15 --max_v 0.5 --max_yaw_rate 1.2 --k_p_ang 2.0 --base_height 43


python isaac_lab_server_spot_json.py \
--enable_cameras \
--scene_path "/home/junzhewu/data/isaac_scenes_v1/nature/mountain/0/fine_omniverse/export_scene.blend/export_scene.usdc" \
--plan_json /home/junzhewu/pohsun/SG-VLN/robot_env/test_jsons/nature.json --use_plan \
--waypoint_stride 1 --arrive_thresh 0.15 --max_v 0.5 --max_yaw_rate 1.2 --k_p_ang 2.0 --base_height 15.75


python isaac_lab_server_spot_json.py \
--enable_cameras \
--scene_path "/home/junzhewu/data/isaac_scenes_v1/umich/umich.usdc" \
--plan_json /home/junzhewu/pohsun/SG-VLN/robot_env/test_jsons/umich.json --use_plan \
--waypoint_stride 1 --arrive_thresh 0.15 --max_v 0.5 --max_yaw_rate 1.2 --k_p_ang 2.0 --base_height 236.7