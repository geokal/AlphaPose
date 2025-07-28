#!/bin/bash
# Device handling is automatically managed by the Python script:
# 1. Prioritizes CUDA if available (for multi-GPU systems)
# 2. Falls back to Apple Silicon MPS (Metal Performance Shaders) if CUDA unavailable
# 3. Uses CPU as final fallback if neither CUDA nor MPS is available
# No need to specify device parameters - the script will use optimal available hardware

set -x

CONFIG=$1
CKPT=$2
VIDEO=$3
OUTDIR=${4:-"./examples/res"}

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --video ${VIDEO} \
    --outdir ${OUTDIR} \
    --detector yolo  --save_img --save_video
