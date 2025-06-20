#!/bin/bash
# example usage: bash run_vis_multi.sh 2>&1 | tee offline_analysis.log
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet


# Set experiment name and configuration variables
expname="offline_analysis_sample100_apr26"
exp_config="Utility_ObjectRegion_ExploitationSum_ExplorationRaycast.yaml"
dump_dir="dump"
log_file="offline_analysis.log"

# Process and GPU configuration variables
n_process_per_gpu=6
num_gpu=2
# Set GPU IDs to use (comma-separated list, e.g. "0,1,2,3")
gpu_ids="0,1"
export CUDA_VISIBLE_DEVICES=${gpu_ids}

echo "Using GPUs: ${gpu_ids}"
echo "Number of processes per GPU: ${n_process_per_gpu}"
echo "Number of GPUs: ${num_gpu}"
echo "Experiment config: ${exp_config}"

# remove old log link if exists
mkdir -p ${dump_dir}/${expname}/objectnav-dino
log_target=${dump_dir}/${expname}/objectnav-dino/log.txt
if [ -f "${log_target}" ]; then
    rm -f ${log_target}
fi
ln $(pwd)/${log_file} ${log_target}

# Set up base command
cmd="python -u main_vis_multi.py \
    -n ${n_process_per_gpu} \
    --num_gpu ${num_gpu} \
    --fmm_planner \
    --imagine_nav_planner \
    --no_llm \
    --gt_perception \
    --gt_scenegraph \
    --save_step_data \
    --exp_config ${exp_config} \
    --group_caption_vlm llama3.2-vision \
    --scene_graph_prediction_llm llama3.2-vision \
    --episode_labels sample100 \
    --ollama_port_start 12000 \
    --dump_location ${dump_dir}/${expname}"

    # --gt_scenegraph \
    # --group_caption_vlm gpt-4o-mini \
    # --scene_graph_prediction_llm gpt-4o-mini \
    # --save_step_data \
    # --episode_labels sample100 \
    # --imagine_nav_planner \
    # --no_llm \


# Conda environment name
# conda_env_name="vln"

# Remove log file if it exists
# if [ -f "${log_file}" ]; then
#     rm -f ${log_file}
# fi

# Maximum number of restart attempts
MAX_RESTARTS=100
restart_count=0

# Function to clean up processes
cleanup() {
    echo "Cleaning up processes..."
    pkill -f "python.*main_vis_multi.py"
    pkill ollama
    sleep 2
}

# Register cleanup function for script exit
trap cleanup EXIT

# Main execution loop
while [ $restart_count -lt $MAX_RESTARTS ]; do
    echo "Starting execution (attempt $(($restart_count + 1)) of $MAX_RESTARTS)"

    # Run the command
    echo "Executing command:\n${cmd}"
    eval ${cmd}

    # Check exit status
    exit_code=$?

    # If exit was clean (0), break the loop
    if [ $exit_code -eq 0 ]; then
        echo "Script completed successfully."
        break
    fi

    # Handle segmentation fault or other errors
    if [ $exit_code -eq 139 ] || [ $exit_code -eq 134 ] || [ $exit_code -ne 0 ]; then
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        echo "[$timestamp] ERROR: Script crashed with exit code $exit_code"
        
        # Clean up before restart
        cleanup
        
        # Wait a moment before restarting
        sleep 5
    fi

    restart_count=$((restart_count + 1))

    if [ $restart_count -ge $MAX_RESTARTS ]; then
        echo "Maximum restart attempts ($MAX_RESTARTS) reached. Giving up."
    else
        echo "Restarting script (attempt $(($restart_count + 1)) of $MAX_RESTARTS)..."
    fi
done

echo "Script execution finished."
