#!/bin/bash
# GPU: localizer prototypes, training crops, verifier training (MECCANO STARE-lite).
S=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); L=$S/metrics/meccano/logs; mkdir -p $L
source $S/env.sh >/dev/null 2>&1
export HOME=$S/.tmp MECCANO_VIDEOS=${MECCANO_VIDEOS:-$S/data/MECCANO/MECCANO_RGB_Videos} MECCANO_ROOT=${MECCANO_ROOT:-$S/data/MECCANO}
cd $S
[ -f $S/metrics/meccano/work/vision/localizer_protos.npz ] || python -u metrics/meccano/stare_lite.py locproto > $L/locproto.log 2>&1 || exit 1
python -u metrics/meccano/stare_lite.py crops > $L/crops.log 2>&1 || exit 1
python -u metrics/meccano/stare_lite.py verifier > $L/verifier.log 2>&1 || exit 1
echo VISION DONE >> $L/verifier.log
