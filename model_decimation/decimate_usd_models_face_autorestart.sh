#!/bin/bash
## Script Description: This script runs a pipeline to process USD models
##                      to decimate the number of faces.
## Author: Po-Hsun Chang
## Contact: pohsun@umich.edu

# Check if input folder argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_folder>"
    echo "Example: $0 /home/junzhewu/pohsun/data_decimated/grscenes_commercial/models"
    exit 1
fi

INPUT_FOLDER="$1"

# Validate input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

echo "Processing USD models in folder: $INPUT_FOLDER"

# Step 1: Run renaming script using IsaacLab
PRE_SCRIPT="/home/junzhewu/pohsun/SG-VLN/robot_env/model_decimation/make_usd_prim_unique.py"
echo "=== Running IsaacLab preprocessing at $(date) ==="
python "$PRE_SCRIPT" --input_folder "$INPUT_FOLDER"
PRE_STATUS=$?

if [ $PRE_STATUS -ne 0 ]; then
    echo ">>> IsaacLab preprocessing failed (exit $PRE_STATUS). Aborting."
    exit 1
fi
echo "=== IsaacLab preprocessing finished successfully ==="

# Step 2: Run face decimation in Blender with auto-restart on crash or hang
SCRIPT="/home/junzhewu/pohsun/SG-VLN/robot_env/model_decimation/decimate_usd_models_face_blender.py"
LOG="blender_run.log"
MAX_IDLE=5   # seconds of no output before killing
MEM_LIMIT="7G"  # memory limit for Blender process
RATIO=0.1  # default decimation ratio

while true; do
    echo "=== Starting Blender job at $(date) ==="
    : > "$LOG"   # truncate old log
    systemd-run --user --scope -p MemoryMax=$MEM_LIMIT blender --background --python "$SCRIPT" -- --input_folder "$INPUT_FOLDER" --ratio $RATIO >"$LOG" 2>&1 &
    PID=$!

    while kill -0 $PID 2>/dev/null; do
        # last modification time of log (seconds since epoch)
        LAST_UPDATE=$(stat -c %Y "$LOG")
        NOW=$(date +%s)
        AGE=$((NOW - LAST_UPDATE))

        if (( AGE > MAX_IDLE )); then
            echo ">>> No output for $MAX_IDLE seconds, killing Blender (PID $PID)"
            kill -9 $PID 2>/dev/null
            break
        fi
        sleep 2
    done

    wait $PID 2>/dev/null
    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "=== Blender finished successfully at $(date) ==="
        break
    else
        echo ">>> Blender crashed or was killed (exit $STATUS). Restarting..."
    fi
done

# Step 3: Run replacing decimated mesh to renamed usd in IsaacLab
POST_SCRIPT="/home/junzhewu/pohsun/SG-VLN/robot_env/model_decimation/replace_usd_models_isaac.py"
echo "=== Running post-processing at $(date) ==="
python "$POST_SCRIPT" --input_folder "$INPUT_FOLDER"
POST_STATUS=$?

if [ $POST_STATUS -ne 0 ]; then
    echo ">>> Post-processing script failed (exit $POST_STATUS)."
    exit 1
fi

echo "=== Pipeline completed successfully at $(date) ==="