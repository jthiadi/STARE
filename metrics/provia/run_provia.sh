#!/bin/bash
# PROVIA on IndustReal with the authors' released models, their code unchanged (third_party/PROVIA).
#   1. their IndustReal manifest (make_manifest.sh, already in metrics/provia/data)
#   2. frozen DINOv2 ViT-S/14 features at 8 Hz (their extract_stream_features, DINOv2 hub code + weights in STARE)
#   3. their scripts/evaluate_checkpoints.sh: rebuilds the inputs with the released observer, fixes the operating
#      thresholds on validation, scores test once per seed; writes runs/industreal/model_seed*/test_candidates.npz
S=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); P=$S/third_party/PROVIA; H=$S/env/cache/torch/hub
source $S/env.sh >/dev/null 2>&1
export HOME=$S/.tmp PROVIA_DEVICE=cuda PROVIA_GPUS=1 PYTHONPATH=$P:$PYTHONPATH
cd $P
mkdir -p data && cp $S/metrics/provia/data/industreal.jsonl data/industreal.jsonl
for part in train_val test; do
  [ -f cache/industreal/$part/manifest.json ] || [ -n "$(ls cache/industreal/$part 2>/dev/null)" ] && [ -n "$SKIP_FEATURES" ] && continue
  args="--partition train --partition validation"; [ $part = test ] && args="--partition test"
  python -m provia.training.extract_stream_features --manifest data/industreal.jsonl --output-root cache/industreal/$part \
    $args --sample-rate-hz 8 --batch-size 256 --device cuda \
    --repo $H/facebookresearch_dinov2_main --checkpoint $H/checkpoints/dinov2_vits14_pretrain.pth || exit 1
done
bash scripts/evaluate_checkpoints.sh industreal $S/third_party/PROVIA_checkpoints/industreal || exit 1
echo PROVIA DONE
