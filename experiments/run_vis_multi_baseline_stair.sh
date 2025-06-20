#!/bin/bash
# example usage: bash run_vis_multi.sh 2>&1 | tee greatlake.log
export MAGNUM_LOG=quiet GLOG_minloglevel=2

# Set experiment name and configuration variables
# expname="test_scenegraph_raycast_19"
expname="apr21_hm3d_baseline_stair"
exp_config="Utility_ObjectRegion_ExploitationOnly_Prediction_Global.yaml"
# exp_config="Utility_ObjectRegion_ExploitationOnly_Prediction_Global.yaml"
# exp_config="Utility_ObjectRegion_ExploitationOnly_ExplorationRaycast_19.yaml"
dump_dir="dump"
log_file="batch.log"
episode_labels="sample400"

# Process and GPU configuration variables
n_process_per_gpu=3
num_gpu=4
# Set GPU IDs to use (comma-separated list, e.g. "0,1,2,3")
gpu_ids="0,1,2,3"
export CUDA_VISIBLE_DEVICES=${gpu_ids}

# Feature flags
use_imagine_nav_planner=False  # Set to true to enable imagine_nav_planner

# Conda environment name
conda_env_name="vln"

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
    # echo "Starting execution (attempt $(($restart_count + 1)) of $MAX_RESTARTS)"
    echo "Using GPUs: ${gpu_ids}"
    echo "Number of processes per GPU: ${n_process_per_gpu}"
    echo "Number of GPUs: ${num_gpu}"
    echo "Experiment config: ${exp_config}"

    # Set up base command
    cmd="python -u main_vis_multi.py \
        -n ${n_process_per_gpu} \
        --num_gpu ${num_gpu} \
        --fmm_planner \
        --exp_config ${exp_config} \
        --dump_location ${dump_dir}/${expname} \
        --episode_labels ${episode_labels}"

    # Add imagine_nav_planner flag if enabled
    if [ "$use_imagine_nav_planner" = true ]; then
        cmd="${cmd} --imagine_nav_planner"
        echo "Using imagine_nav_planner"
    fi

    # Remove log file if it exists
    if [ -f "${log_file}" ]; then
        rm -f ${log_file}
    fi

    # Run the command
    echo "Executing command:\n${cmd}"
    # screen -L -Logfile ${log_file} conda run -n ${conda_env_name} --no-capture-output --live-stream ${cmd}
    eval ${cmd}
    # 2>&1 | tee -a ${log_file}

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
