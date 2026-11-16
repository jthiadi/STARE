#!/bin/bash
# Controlled PREGO baseline from the PROVIA repository (authors' implementation, unchanged), seeds 0-2, on the inputs built by
# run_provia.sh. Writes third_party/PROVIA/runs/industreal/baselines/prego_seed*/{validation,test}_candidates.npz.
S=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); P=$S/third_party/PROVIA; L=$S/metrics/provia/logs
source $S/env.sh >/dev/null 2>&1
export HOME=$S/.tmp PROVIA_DEVICE=cuda PROVIA_GPUS=1 PYTHONPATH=$P:$PYTHONPATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # required by PyTorch for the deterministic mode PROVIA's code enables
cd $P
(CUDA_VISIBLE_DEVICES=0 bash scripts/baselines/single_actor.sh industreal prego 0 > $L/prego_seed0.log 2>&1 && \
 CUDA_VISIBLE_DEVICES=0 bash scripts/baselines/single_actor.sh industreal prego 2 > $L/prego_seed2.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 bash scripts/baselines/single_actor.sh industreal prego 1 > $L/prego_seed1.log 2>&1) &
wait
for s in 0 1 2; do [ -f runs/industreal/baselines/prego_seed$s/test_candidates.npz ] || { echo "PREGO SEED $s FAILED"; exit 1; }; done
echo PREGO DONE
