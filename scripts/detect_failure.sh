#!/bin/bash

SCRIPT_NAME="fault_tolerant_ttc.py"
FULL_PATH=$( dirname -- "$0"; )
echo "$FULL_PATH/$SCRIPT_NAME"
LOG_FILE="monitor.log"

# checks every 5s
CHECK_INTERVAL=5

while true; do
    if ! pgrep -f "[^/]*$SCRIPT_NAME" > /dev/null; then
        echo "$(date) script down. restarting" >> "$LOG_FILE"
        python "$FULL_PATH/$SCRIPT_NAME" >> "$LOG_FILE" &

        sleep "$CHECK_INTERVAL" # after restarting wait a bit
    fi
    
    # Wait before next check
    sleep "$CHECK_INTERVAL"
done
