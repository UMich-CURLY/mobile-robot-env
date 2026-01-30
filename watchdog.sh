#!/bin/bash

# Define the command you want to run here
COMMAND="/path/to/your/application --argument"

echo "---------------------------------"
echo "Starting wrapper for: $COMMAND"
echo "Use CTRL+C to stop the loop."
echo "---------------------------------"

while true; do
    # 1. Run the command
    python run_task_generator.py --num_workers 1 --tg_config_path episodes/task_config.yaml &> task_generation.log
    
    # 2. Capture the exit code (optional, but helpful for debugging)
    EXIT_CODE=$?
    echo "Process exited with code $EXIT_CODE."
    
    # 3. Wait before restarting to avoid thrashing the CPU
    echo "Restarting in 2 seconds..."
    sleep 2
done