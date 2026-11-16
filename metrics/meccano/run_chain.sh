#!/bin/bash
# MECCANO STARE-lite, after features.py and run_vision.sh: causal TCN, verifier readings, then one scoring run.
S=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); L=$S/metrics/meccano/logs; mkdir -p $L
source $S/env.sh >/dev/null 2>&1
export HOME=$S/.tmp
cd $S
until [ "$(grep -c "FEATURES DONE" $L/features_0.log $L/features_1.log 2>/dev/null | awk -F: '{s+=$2} END {print s}')" = 2 ]; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 python -u metrics/meccano/stare_lite.py tcn > $L/tcn.log 2>&1 || { echo "TCN FAILED" >> $L/chain.log; exit 1; }
echo "TCN DONE" >> $L/chain.log
until grep -q "VISION DONE" $L/verifier.log 2>/dev/null; do
  grep -q "Traceback" $L/verifier.log $L/crops.log 2>/dev/null && { echo "VISION FAILED" >> $L/chain.log; exit 1; }
  sleep 60
done
CUDA_VISIBLE_DEVICES=0 python -u metrics/meccano/stare_lite.py vprobs 0/2 > $L/vprobs_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u metrics/meccano/stare_lite.py vprobs 1/2 > $L/vprobs_1.log 2>&1 &
wait
echo "VPROBS DONE" >> $L/chain.log
python -u metrics/meccano/stare_lite.py run > $L/run.log 2>&1 && echo "RUN DONE" >> $L/chain.log || echo "RUN FAILED" >> $L/chain.log
