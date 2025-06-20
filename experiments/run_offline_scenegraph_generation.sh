#!/bin/bash

# Config
n_process_per_gpu=4
total_range=96
gpu_ids="2,3"
IFS=',' read -r -a gpu_array <<< "$gpu_ids"
num_gpu=${#gpu_array[@]}
total_processes=$((num_gpu * n_process_per_gpu))
range_per_proc=$((total_range / total_processes))
remainder=$((total_range % total_processes))

echo "GPUs: ${gpu_array[@]}"
echo "Total processes: $total_processes"
echo "Range per process: $range_per_proc (+ remainder: $remainder)"

# Run processes
proc_idx=0
split_l=0
for gpu in "${gpu_array[@]}"; do
    for ((i = 0; i < n_process_per_gpu; i++)); do
        this_range=$range_per_proc

        # Spread the remainder across the first N processes
        if [ $proc_idx -lt $remainder ]; then
            this_range=$((this_range + 1))
        fi

        split_r=$((split_l + this_range))

        echo "Launching on GPU $gpu with split_l=$split_l, split_r=$split_r"

        CUDA_VISIBLE_DEVICES=$gpu python -u test_scenegraph_offline.py \
            --split_l $split_l \
            --split_r $split_r &

        split_l=$split_r
        proc_idx=$((proc_idx + 1))
    done
done

# Wait for all background jobs to finish
wait
echo "All processes completed."
