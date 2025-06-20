import os
import json
import zipfile
import shutil

def load_all_info(log_dir):
    per_episode_error = []
    if os.path.exists(log_dir + "all_info.log"):
        #print(f"Loading {log_dir}all_info.log")
        with open(log_dir + "all_info.log", 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.strip()
                if line != "":
                    per_episode_error.append(json.loads(line))
    return per_episode_error

def save_all_info(log_dir, per_episode_error):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + "all_info.log", 'w') as fp:
        for item in per_episode_error:
            json.dump(item, fp)
            fp.write("\n")
    return

# sample episode labels
def load_env(task_config="objectnav_hm3d.yaml", split="val"):
    from habitat.config.default import get_config
    from habitat import Env, make_dataset

    config_env = get_config(config_paths=["configs/" + task_config])
    config_env.defrost()
    config_env.DATASET.SPLIT = split
    config_env.freeze()
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    env = Env(config_env, dataset)
    return env

def get_scene_id(scene_path):
    import re
    # Get the scene_id from the path
    scene_id = scene_path.split("/")[-1]
    scene_id = re.sub(r'\.basis\.glb$', '', scene_id)
    return scene_id

def get_episode_labels(env):
    episode_labels = []
    for i in range(len(env.episodes)):
        scene_id = get_scene_id(env.episodes[i].scene_id)
        episode_id = env.episodes[i].episode_id
        episode_label = f'{scene_id}_{episode_id}'
        episode_labels.append(episode_label)
    return episode_labels

def pack_videos(output_path, output_name, videos_path1, videos_path2, episode_labels, 
               suffix1="_old", suffix2="_new", to_zip=True, to_folder=False):
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Track successful and failed video additions
    added_videos1 = []
    added_videos2 = []
    missing_videos1 = []
    missing_videos2 = []
    # Paths to return
    result_paths = {}
    # Create folder if requested
    folder_path = None
    if to_folder:
        folder_path = os.path.join(output_path, output_name)
        if os.path.exists(folder_path):
            # Clean the existing folder
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        result_paths['folder'] = folder_path
    # Create the zip file if requested
    zip_file_path = None
    zip_obj = None
    if to_zip:
        zip_file_path = os.path.join(output_path, f"{output_name}.zip")
        zip_obj = zipfile.ZipFile(zip_file_path, 'w')
        result_paths['zip'] = zip_file_path
    # Process each episode
    for episode_label in episode_labels:
        # Construct video file names
        video_name = f"eps_{episode_label}_vis.mp4"
        video_path1 = os.path.join(videos_path1, video_name)
        video_path2 = os.path.join(videos_path2, video_name)
        # Get file extension
        _, ext = os.path.splitext(video_name)
        # Process video from first path
        if os.path.exists(video_path1):
            # Create new name with suffix
            new_name1 = f"eps_{episode_label}{suffix1}{ext}"
            # Add to zip if needed
            if to_zip:
                zip_obj.write(video_path1, arcname=new_name1)
            # Copy to folder if needed
            if to_folder:
                shutil.copy2(video_path1, os.path.join(folder_path, new_name1))
            added_videos1.append(episode_label)
        else:
            missing_videos1.append(episode_label)
        # Process video from second path
        if os.path.exists(video_path2):
            # Create new name with suffix
            new_name2 = f"eps_{episode_label}{suffix2}{ext}"
            # Add to zip if needed
            if to_zip:
                zip_obj.write(video_path2, arcname=new_name2)
            # Copy to folder if needed
            if to_folder:
                shutil.copy2(video_path2, os.path.join(folder_path, new_name2))
            added_videos2.append(episode_label)
        else:
            missing_videos2.append(episode_label)
    # Close the zip file if we created one
    if to_zip:
        zip_obj.close()
    # Print results
    print(f"Pack videos completed:")
    if to_zip:
        print(f"- Created zip file: {zip_file_path}")
    if to_folder:
        print(f"- Created folder: {folder_path}")
    print(f"Added {len(added_videos1)} videos from path1 and {len(added_videos2)} videos from path2.")
    if missing_videos1:
        print(f"Warning: Could not find {len(missing_videos1)} videos in path1:")
        print(missing_videos1)
    if missing_videos2:
        print(f"Warning: Could not find {len(missing_videos2)} videos in path2:")
        print(missing_videos2)
    # Add statistics to result
    result_paths['stats'] = {
        'added_path1': len(added_videos1),
        'added_path2': len(added_videos2),
        'missing_path1': len(missing_videos1),
        'missing_path2': len(missing_videos2)
    }
    return result_paths

# assume results2 have equal or less entries than results1
# log_path1 = './dump/fmm/logs/objectnav-dino/'
# log_path2 = './dump/new_baseline_apr2/objectnav-dino/logs/'

# log_path1 = '/home/junzhewu/scratch/shared_data/dump_ext/new_baseline_apr2/objectnav-dino/logs/'
# log_path2 = './ext_dump/apr9_object_exploitation_observed_update/objectnav-dino/logs/'
# vid_path1 = '/home/junzhewu/scratch/shared_data/dump_ext/new_baseline_apr2/objectnav-dino/episodes_video'
# vid_path2 = './ext_dump/apr9_object_exploitation_observed_update/objectnav-dino/episodes_video/'

# log_path1 = '/home/junzhewu/scratch/shared_data/dump_ext/new_baseline_apr2/objectnav-dino/logs/'
# log_path2 = './ext_dump/apr10_baseline_multifloor/objectnav-dino/logs/'
# vid_path1 = '/home/junzhewu/scratch/shared_data/dump_ext/new_baseline_apr2/objectnav-dino/episodes_video'
# vid_path2 = './ext_dump/apr10_baseline_multifloor/objectnav-dino/episodes_video/'


# log_path1 = './ext_dump/apr9_object_exploitation_observed/objectnav-dino/logs/'
# log_path2 = './ext_dump/apr9_object_exploitation_observed_update/objectnav-dino/logs/'
# vid_path1 = './ext_dump/apr9_object_exploitation_observed/objectnav-dino/episodes_video'
# vid_path2 = './ext_dump/apr9_object_exploitation_observed_update/objectnav-dino/episodes_video/'

