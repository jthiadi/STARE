# STARE environment: run "source env.sh" from anywhere, then use python. Everything is inside this folder:
# portable Python 3.12 (env/python312), packages (env/venv, env/pkgs), model cache (env/cache/hf, offline), temp files.
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export STARE=$S
export PATH=$S/env/bin:$S/env/python312/python/bin:$PATH
export PYTHONPATH=$S/env/pkgs:$S/env/venv/lib/python3.12/site-packages:.
export TMPDIR=$S/.tmp TMP=$S/.tmp TEMP=$S/.tmp
export HF_HOME=$S/env/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=$S/env/cache TORCH_HOME=$S/env/cache/torch MPLCONFIGDIR=$S/env/cache/mpl
export YOLO_CONFIG_DIR=$S/env/cache/yolo PIP_CACHE_DIR=$S/env/cache/pip
export KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p $S/.tmp $S/env/cache/torch $S/env/cache/mpl $S/env/cache/yolo
