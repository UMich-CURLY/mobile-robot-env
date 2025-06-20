#!/bin/bash
# example usage: bash experiments/run_exp.sh experiment_name config_file "5,4,0,1,2,3" sample100 true
# Parameters:
#   $1 - expname: experiment name
#   $2 - exp_config: experiment configuration file
#   $3 - n_process_per_gpu
#   $4 - num_gpu
#   $5 - gpu_ids
#   $6 - episode_labels: episode labels to use
#   $7 - save_step_data: (optional) if "true", add the --save_step_data flag
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# Parse command line arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <expname> <exp_config> <n_process_per_gpu,num_gpu,gpu_ids> <episode_labels> [save_step_data] [additional_args...]"
    echo "Example: $0 experiment_name config_file \"5,4,0,1,2,3\" sample100 true"
    exit 1
fi

# Set experiment name and configuration variables
expname=$1
exp_config=$2
dump_dir="ext_dump"
log_file="log.txt"

# Parse processing configuration
n_process_per_gpu=$3
num_gpu=$4
gpu_ids=$5
export CUDA_VISIBLE_DEVICES=${gpu_ids}

# Episode labels
episode_labels=$6

echo "Using GPUs: ${gpu_ids}"
echo "Number of processes per GPU: ${n_process_per_gpu}"
echo "Number of GPUs: ${num_gpu}"
echo "Experiment config: ${exp_config}"

# Create directory for logs
mkdir -p ${dump_dir}/${expname}/objectnav-dino/
log_file=${dump_dir}/${expname}/objectnav-dino/${log_file}

# Check if save_step_data flag should be added
save_step_data=""
if [ "$7" == "true" ]; then
    echo "Adding --save_step_data flag"
    save_step_data="--save_step_data"
fi

# Collect additional arguments (arguments after $7)
additional_args=""
if [ $# -gt 7 ]; then
    for i in $(seq 8 $#); do
        additional_args="${additional_args} ${!i}"
    done
    echo "Adding additional arguments: ${additional_args}"
fi

# Set up base command
cmd="EXP_NAME=${expname} /workspace/conda/envs/vln/bin/python -u main_vis_multi.py \
    -n ${n_process_per_gpu} \
    --num_gpu ${num_gpu} \
    --fmm_planner \
    --edge_goal \
    --temporal_collision \
    --imagine_nav_planner \
    --gt_scenegraph \
    --exp_config ${exp_config} \
    --group_caption_vlm gpt-4o-mini \
    --scene_graph_prediction_llm gpt-4o-mini \
    --episode_labels ${episode_labels} \
    --ollama_port_start 14200 \
    --dump_location ${dump_dir}/${expname} \
    ${save_step_data} \
    ${additional_args}"

    # --gt_perception \
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
if [ -f "${log_file}" ]; then
    rm -f ${log_file}
    rm -f ${dump_dir}/${expname}/objectnav-dino/${expname}.log
fi

# Set up log redirection for the entire script
# Redirect all output (stdout and stderr) to both the console and the log file
exec > >(tee -a "${log_file}") 2>&1
ln -s ${log_file} ${dump_dir}/${expname}/objectnav-dino/${expname}.log

# Maximum number of restart attempts
MAX_RESTARTS=100
restart_count=0

# Function to clean up processes
cleanup() {
    echo "Cleaning up processes..."
    pkill -f "EXP_NAME=${expname}.*python.*main_vis_multi.py"
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
        sleep 3
    fi

    restart_count=$((restart_count + 1))

    if [ $restart_count -ge $MAX_RESTARTS ]; then
        echo "Maximum restart attempts ($MAX_RESTARTS) reached. Giving up."
    else
        echo "Restarting script (attempt $(($restart_count + 1)) of $MAX_RESTARTS)..."
    fi
done

echo "Script execution finished."
