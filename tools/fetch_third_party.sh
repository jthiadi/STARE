#!/bin/bash
# Clone the third-party code STARE uses, at the exact commits we ran, into third_party/.
#   bash tools/fetch_third_party.sh
# PROVIA (MIT) is the mistake-detection baseline of Table 8; STORM-PSR (Apache-2.0) provides the MECCANO step-recognition
# evaluator of Table 5; PSR-annotations holds the MECCANO step labels (from the STORM-PSR authors, under their terms).
# PROVIA's released IndustReal step segmenter is not cloned: download it from the PROVIA authors (link in their README)
# into third_party/PROVIA_checkpoints/industreal/ if you want to retrain PROVIA (metrics/provia/run_provia.sh).
set -e
cd "$(dirname "$0")/.."
mkdir -p third_party
fetch() {  # <folder> <url> <commit>
  if [ -d "third_party/$1/.git" ] || [ -f "third_party/$1/SOURCE.txt" ]; then echo "third_party/$1 exists, skipped"; return; fi
  git clone "$2" "third_party/$1"
  git -C "third_party/$1" checkout --quiet "$3"
  echo "third_party/$1 at $3"
}
fetch PROVIA https://github.com/Kratos-Wen/PROVIA.git 178e78ffca804410c45ced29df959aa9f68a4ccd
fetch STORM-PSR https://github.com/shaohsuanhung/STORM-PSR.git 08e97aac1f63e3db9bfdd9e7badeb8d55ccff0f0
fetch PSR-annotations https://github.com/TimSchoonbeek/PSR-annotations.git b75a230f61f7cd2575fcf6f171bbb320b73c41be
