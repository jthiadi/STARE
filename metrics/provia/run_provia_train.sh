#!/bin/bash
# PROVIA retrained with the authors' unchanged scripts and settings (PREREGISTRATION_untyped.md, amendment):
# inputs rebuilt from our data with the released step segmenter (run_provia.sh did that), seeds 0-2, then their test
# evaluation (validation thresholds, test once). Writes third_party/PROVIA/runs/industreal/model_seed*/.
S=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); P=$S/third_party/PROVIA; L=$S/metrics/provia/logs
source $S/env.sh >/dev/null 2>&1
export HOME=$S/.tmp PROVIA_DEVICE=cuda PROVIA_GPUS=1 PYTHONPATH=$P:$PYTHONPATH
cd $P
[ -d runs/industreal/model_seed0 ] && [ ! -f runs/industreal/model_seed0/latest.pt ] && mv runs/industreal/model_seed0 runs/industreal/released_seed0_unused
(CUDA_VISIBLE_DEVICES=0 bash scripts/single_actor/train_provia.sh industreal 0 > $L/train_seed0.log 2>&1 && \
 CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29502 bash scripts/single_actor/train_provia.sh industreal 2 > $L/train_seed2.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29501 bash scripts/single_actor/train_provia.sh industreal 1 > $L/train_seed1.log 2>&1) &
wait
for s in 0 1 2; do [ -f runs/industreal/model_seed$s/validation.json ] || { echo "SEED $s FAILED"; exit 1; }; done
CUDA_VISIBLE_DEVICES=0 bash scripts/single_actor/evaluate_test.sh industreal > $L/evaluate_test.log 2>&1 || { echo "TEST EVAL FAILED"; exit 1; }
echo PROVIA TRAINED AND EVALUATED
