export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
MODEL="ndrope_deit_small_patch16_LS"; NPROC=1
data_path="/media/xxx/storage/dataset/imagenet"
save_path="/home/xxx/testing/rope-vit-main/eval_results"

PRE_DIR="$(ls -1dt ${save_path}/pretrain_224_* 2>/dev/null | head -n 1)"
CKPT=""
for f in best_checkpoint.pth checkpoint-best.pth best.pth model_best.pth.tar last.pth; do
  [ -f "${PRE_DIR}/${f}" ] && CKPT="${PRE_DIR}/${f}" && break
done
[ -z "$CKPT" ] && CKPT="$(ls -1t ${PRE_DIR}/checkpoint*.pth 2>/dev/null | head -n 1)"
[ -z "$CKPT" ] && { echo "no checkpoint"; exit 1; }

OUTBASE="${save_path}/eval_224_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUTBASE"
for sz in 144 160 192 224 256 320 384 448 512 576 640 704 768 832 896 960 1024; do
  OUT="${OUTBASE}/sz${sz}"; mkdir -p "$OUT"
  echo "[Eval] ${sz} -> ${OUT}"
  torchrun --standalone --nproc_per_node=${NPROC} main.py \
    --model ${MODEL} --data-path "${data_path}" --output_dir "${OUT}" \
    --resume "${CKPT}" --batch-size 256 --input-size ${sz} \
    --eval --eval-crop-ratio 0.95 --dist-eval 2>&1 | tee "${OUT}/eval.log"
done