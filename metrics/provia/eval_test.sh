#!/bin/bash
# GPU: PROVIA's own test evaluation of the 3 retrained seeds (after run_provia_train.sh). Usage: bash metrics/provia/eval_test.sh
S=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); P=$S/third_party/PROVIA; L=$S/metrics/provia/logs; mkdir -p $L
source $S/env.sh >/dev/null 2>&1
export HOME=$S/.tmp PROVIA_DEVICE=cuda PROVIA_GPUS=1 PYTHONPATH=$P:$PYTHONPATH
cd $P
CUDA_VISIBLE_DEVICES=0 bash scripts/single_actor/evaluate_test.sh industreal > $L/evaluate_test.log 2>&1 && echo PROVIA TRAINED AND EVALUATED >> $L/run_provia_train.log || echo "TEST EVAL FAILED" >> $L/run_provia_train.log
