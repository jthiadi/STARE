#!/bin/bash
# PROVIA's own IndustReal conversion (third_party/PROVIA, unmodified), run on the dataset inside STARE.
# Writes metrics/provia/data/industreal.jsonl (the completion manifest PROVIA's protocol scores against).
S=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); P=$S/metrics/provia; D=$P/data
source $S/env.sh
rm -rf $D && mkdir -p $D/recordings
for pair in train:train/train_annotations val:validation/val_annotations test:test/test_annotations; do
  sp=${pair%%:*}; src=$S/system/dataset/${pair#*:}
  for r in $src/*/; do r=${r%/}; mkdir -p $D/recordings/$sp/$(basename $r)
    cp $r/*.csv $D/recordings/$sp/$(basename $r)/; done
done
cat > $D/industreal.local.json <<JSON
{"dataset": "industreal", "dataset_root": "$D", "recordings_root": "$D/recordings",
 "video_root": "$S/system/assistant/data/generated/industreal/videos", "video_patterns": ["{video_id}.mp4"],
 "procedure_info": "$S/system/psr_v14/third_party/industreal_psr/procedure_info.json", "fps": 10.0,
 "splits": ["train", "val", "test"]}
JSON
cd $S/third_party/PROVIA
python -m provia.data.cli convert --dataset industreal --config $D/industreal.local.json --out $D/industreal.all.jsonl
python -m provia.data.project_single_actor --input $D/industreal.all.jsonl --sample-rate-hz 8 --action-marks --output $D/industreal.jsonl
