cd /home/hpcserver/testing/rope-vit-main/deit

data_path="xxx/dataset/imagenet"
save_path="/home/hpcserver/testing/rope-vit-main/eval_results"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=1 main.py \
  --model ndrope_deit_small_patch16_LS \
  --data-path ${data_path} \
  --output_dir ${save_path} \
  --batch-size 256 \
  --epochs 400 \
  --smoothing 0.0 --reprob 0.0 \
  --opt fusedlamb --color-jitter 0.3 --lr 4e-3 --weight-decay 0.03 \
  --input-size 224 --drop 0.0 --drop-path 0.0 \
  --unscale-lr --repeated-aug --bce-loss --ThreeAugment \
  --eval-crop-ratio 1.0 --dist-eval


########## paths ##########
data_path="/media/hpcserver/storage/dataset/imagenet"
save_path="/home/hpcserver/testing/rope-vit-main/eval_results"

########## training setup ##########
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

MODEL="ndrope_deit_small_patch16_LS"
NPROC=1

timestamp() { date +"%Y%m%d_%H%M%S"; }

############ Stage 1: 224 pretrain ############
OUT1="${save_path}/pretrain_224_$(timestamp)"
mkdir -p "${OUT1}"
echo "[Stage 1] Output dir: ${OUT1}"

torchrun --standalone --nproc_per_node=${NPROC} main.py \
  --model ${MODEL} \
  --data-path "${data_path}" \
  --output_dir "${OUT1}" \
  --batch-size 512 \
  --epochs 400 \
  --opt adamw --lr 5e-4 --weight-decay 0.05 \
  --sched cosine --warmup-epochs 5 --warmup-lr 1e-6 --min-lr 1e-6 \
  --input-size 224 \
  --smoothing 0.1 --reprob 0.25 --drop-path 0.1 --color-jitter 0.4 \
  --repeated-aug \
  --model-ema --model-ema-decay 0.9999 \
  --eval-crop-ratio 0.95 \
  --clip-grad 1.0 \
  --num_workers 8 --pin-mem \
  --mixup 0.8 --cutmix 1.0 --aa rand-m9-mstd0.5-inc1