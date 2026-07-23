#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

MODEL="ndrope_deit_small_patch16_LS"
NPROC=1
data_path="/media/xxx/storage/dataset/imagenet"
save_path="/home/xxx/testing/rope-vit-main-yarn/eval_results"

BASE=224

PRE_DIR="$(ls -1dt ${save_path}/pretrain_224_* 2>/dev/null | head -n 1)"
CKPT=""
for f in best_checkpoint.pth checkpoint-best.pth best.pth model_best.pth.tar last.pth; do
  if [ -f "${PRE_DIR}/${f}" ]; then
    CKPT="${PRE_DIR}/${f}"
    break
  fi
done
if [ -z "$CKPT" ]; then
  CKPT="$(ls -1t ${PRE_DIR}/checkpoint*.pth 2>/dev/null | head -n 1 || true)"
fi
if [ -z "$CKPT" ]; then
  echo "no checkpoint"; exit 1
fi

OUTBASE="${save_path}/eval_224_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTBASE"

SZS=(144 160 192 224 256 320 384 448 512 576 640 704 768 832 896 960 1024)
BASE=224
MAX_BS=256
MIN_BS=4

for sz in "${SZS[@]}"; do
  OUT="${OUTBASE}/sz${sz}"
  mkdir -p "$OUT"

  FACTOR="$(python -c "sz=${sz}; base=${BASE}; print((sz/base)**2)")"

  BATCH="$(python - "${sz}" "${BASE}" "${MAX_BS}" "${MIN_BS}" <<'PY'
import sys, math
sz=float(sys.argv[1]); base=float(sys.argv[2])
max_bs=int(sys.argv[3]); min_bs=int(sys.argv[4])
factor=(sz/base)**2
bs=max(min_bs, min(max_bs, int(max_bs//factor)))
for cand in [256,192,128,96,64,48,40,36,32,28,24,20,16,14,12,10,8,6,4]:
    if cand <= bs:
        bs = cand
        break
print(bs)
PY
)"

  echo "[Eval] sz=${sz}  yarn_factor=${FACTOR}  auto_batch=${BATCH}  -> ${OUT}"

  torchrun --standalone --nproc_per_node=${NPROC} main.py \
    --model ${MODEL} --data-path "${data_path}" --output_dir "${OUT}" \
    --resume "${CKPT}" --batch-size ${BATCH} --input-size ${sz} \
    --eval --eval-crop-ratio 0.95 --dist-eval \
    --yarn-factor ${FACTOR} \
    --yarn-cutoff 0.6 \
    --yarn-sharpness 8.0 \
    --yarn-power 1.0 \
    2>&1 | tee "${OUT}/eval.log"
done