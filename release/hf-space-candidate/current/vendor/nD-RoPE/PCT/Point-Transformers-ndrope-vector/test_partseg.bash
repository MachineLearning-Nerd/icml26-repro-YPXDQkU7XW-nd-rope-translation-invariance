#!/bin/bash

# ===============================
# Point cloud extrapolation test
# ===============================

GPU_ID=0
BATCH_SIZE=16

TRAIN_NPOINTS=2048

NPOINTS_LIST=(256 768 1024 1536 2048 3072 4096)

echo "=============================================="
echo " Point Transformer PartSeg Extrapolation Test "
echo " Train npoints = ${TRAIN_NPOINTS}"
echo " GPU = ${GPU_ID}"
echo "=============================================="

for NPOINTS in "${NPOINTS_LIST[@]}"
do
    echo ""
    echo "----------------------------------------------"
    echo " Testing with npoints = ${NPOINTS}"
    echo "----------------------------------------------"

    python test_partseg.py \
        gpu=${GPU_ID} \
        batch_size=${BATCH_SIZE} \
        num_point=${NPOINTS}

done

echo ""
echo "=============================================="
echo " All tests finished."
echo "=============================================="