#!/bin/bash
# You need to change the name of ndrope to yarn
# ----------------------------------------------------------
# Config files to test
# ----------------------------------------------------------
CONFIG_FILES=(
  "configs/Kinetics/TimeSformer_divST_8x32_224_TEST-ndrope.yaml"
)

# ----------------------------------------------------------
# Input resolutions to test
# ----------------------------------------------------------
SIZES=(256 320 384 448 512 576 640 704 768 832 896 960 1024)

# ----------------------------------------------------------
# Base output directory
# ----------------------------------------------------------
BASE_OUTPUT_DIR="test_results_yarn_best"
mkdir -p "$BASE_OUTPUT_DIR"

# ----------------------------------------------------------
# YaRN scaling anchors (EXPLICIT OPTIMAL DESIGN)
# ----------------------------------------------------------
TRAIN_CROP_SIZE=224
TARGET_SIZE=1024
FMAX=8.0      

echo "Starting batch tests with optimal YaRN scaling..."

# ----------------------------------------------------------
# Loop over configs
# ----------------------------------------------------------
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do

    CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
    CONFIG_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONFIG_NAME}"
    mkdir -p "$CONFIG_OUTPUT_DIR"

    echo
    echo "======================================================="
    echo "=== Testing config: $CONFIG_NAME"
    echo "======================================================="

    # ------------------------------------------------------
    # Loop over input sizes
    # ------------------------------------------------------
    for sz in "${SIZES[@]}"; do

        # --------------------------------------------------
        # Compute optimal YaRN factor (power-law scaling)
        # --------------------------------------------------
        FACTOR=$(python - <<EOF
import math
sz = $sz
train = $TRAIN_CROP_SIZE
target = $TARGET_SIZE
Fmax = $FMAX

r = sz / train
r_target = target / train
alpha = math.log(Fmax) / math.log(r_target)

factor = Fmax * (r / r_target) ** alpha
print(f"{factor:.4f}")
EOF
)

        echo "-------------------------------------"
        echo "--- Size: $sz | YaRN Factor: $FACTOR ---"
        echo "-------------------------------------"

        RUN_OUTPUT_DIR="${CONFIG_OUTPUT_DIR}/test_size_${sz}"
        mkdir -p "$RUN_OUTPUT_DIR"
        LOG_FILE="${RUN_OUTPUT_DIR}/run.log"

        # --------------------------------------------------
        # Dynamic batch size (safe & conservative)
        # --------------------------------------------------
        if [ "$sz" -le 512 ]; then
            BS=128
        elif [ "$sz" -le 640 ]; then
            BS=64
        elif [ "$sz" -le 704 ]; then
            BS=32
        elif [ "$sz" -le 768 ]; then
            BS=16
        elif [ "$sz" -le 832 ]; then
            BS=12
        elif [ "$sz" -le 896 ]; then
            BS=10
        else
            BS=8
        fi

        echo "Using TEST.BATCH_SIZE = $BS"

        # --------------------------------------------------
        # Run evaluation 
        # --------------------------------------------------
        python tools/run_net.py \
            --cfg "$CONFIG_FILE" \
            DATA.TEST_CROP_SIZE "$sz" \
            TEST.BATCH_SIZE "$BS" \
            OUTPUT_DIR "$RUN_OUTPUT_DIR" \
            MODEL.YARN_FACTOR "$FACTOR" \
            MODEL.YARN_CUTOFF 0.30 \
            MODEL.YARN_SHARPNESS 8.0 \
            MODEL.YARN_POWER 0.75 \
            > "$LOG_FILE" 2>&1

        echo "✔ Done: $CONFIG_NAME @ size $sz → factor=$FACTOR, BS=$BS"

    done
done

echo
echo "======================================================="
echo "✔ All tests finished. Results in: $BASE_OUTPUT_DIR"
echo "======================================================="