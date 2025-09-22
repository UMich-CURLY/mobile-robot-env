#!/bin/bash

SCRIPT="/home/junzhewu/pohsun/SG-VLN/robot_env/decimate_usd_models_face_blender.py"
LOG="blender_run.log"
MAX_IDLE=15   # seconds of no output before killing

while true; do
    echo "=== Starting Blender job at $(date) ==="
    : > "$LOG"   # truncate old log
    blender --background --python "$SCRIPT" >"$LOG" 2>&1 &
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
