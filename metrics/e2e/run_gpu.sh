#!/bin/bash
# GPU jobs for the end-to-end comparison (PREREGISTRATION.md): measured compute (latency.py) and the validation
# ensemble (val_ensemble.py), on two GPUs. Usage: bash metrics/e2e/run_gpu.sh
S=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source $S/env.sh >/dev/null 2>&1
export KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=$S/.tmp
L=$S/metrics/e2e/logs; mkdir -p $TMPDIR $L
cd $S
(CUDA_VISIBLE_DEVICES=1 python metrics/e2e/val_ensemble.py 0/2 > $L/val_ens_0.log 2>&1) &
(CUDA_VISIBLE_DEVICES=0 python metrics/e2e/latency.py > $L/latency.log 2>&1; \
 CUDA_VISIBLE_DEVICES=0 python metrics/e2e/val_ensemble.py 1/2 > $L/val_ens_1.log 2>&1) &
wait
echo ALL GPU DONE > $L/gpu_done
