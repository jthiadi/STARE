#!/bin/bash
# Start the on-demand job service (port 8020, localhost only) for the full system: their TCN + V-JEPA 2 actions, their
# action-end labels, frozen v14 PSR, hands, engine.
#   bash system/assistant/live/start_jobs.sh        (after `source env.sh` from the STARE folder)
cd "$(dirname "$0")/.."
export PSR_AR_DIR=$PWD/output/assist HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p ../logs
if curl -s -o /dev/null http://127.0.0.1:8020/api/status/13_main_2_1; then
  echo "job service already running on port 8020"
else
  nohup python -m live.api --port 8020 > ../logs/api.log 2>&1 &
  echo "job service starting on port 8020 (log: system/logs/api.log)"
fi
