#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --job-name=efficiency_SAD
#SBATCH --output=efficiency_SAD_%j.out
set -euo pipefail

# Edit this block if the server paths or target runs differ.
PROJECT=/data02/home/scv7eyk/cross-model/VM-DMD
PYTHON=/data02/home/scv7eyk/.conda/envs/Femas/bin/python
DATASET="${SAB_DATASET:-isic18}"
SEED="${SAB_SEED:-42}"
TRAIN_GPUS="${EFF_GPUS:-4}"
GLOBAL_BATCH="${EFF_BATCH_SIZE:-0}"  # 0 = saved config (normally 16)
WARMUP="${EFF_WARMUP:-50}"
STEPS="${EFF_STEPS:-200}"
REPEATS="${EFF_REPEATS:-3}"
CACHED_BATCHES=8  # CPU only; same augmented train batches for all three methods
# Empty = resolve one unique known run. Paper D = old train_B, NOT endpoint train_D.
RUN_S="${EFF_RUN_S:-}"
RUN_A="${EFF_RUN_A:-}"
RUN_D="${EFF_RUN_D:-}"
case "$DATASET" in
  isic17) DATA_ROOT=/data02/home/scv7eyk/ISIC2017 ;;
  isic18) DATA_ROOT=/data02/home/scv7eyk/ISIC2018 ;;
  bus) DATA_ROOT=/data02/home/scv7eyk/Breast_BUS ;;
  busi) DATA_ROOT=/data02/home/scv7eyk/Breast_ultrasound ;;
  *) echo 'SAB_DATASET must be isic17, isic18, bus, or busi' >&2; exit 2 ;;
esac
OUTPUT="$PROJECT/reports/efficiency/$DATASET/seed${SEED}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
export PATH="$(dirname "$PYTHON"):$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "$PROJECT"
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 2; }
cmd=("$PYTHON" scripts/benchmark_efficiency.py
  --project-root "$PROJECT" --dataset "$DATASET" --seed "$SEED" --data-root "$DATA_ROOT"
  --run-s "$RUN_S" --run-a "$RUN_A" --run-d "$RUN_D"
  --gpus "$TRAIN_GPUS" --batch-size "$GLOBAL_BATCH"
  --warmup "$WARMUP" --steps "$STEPS" --repeats "$REPEATS" --cached-batches "$CACHED_BATCHES"
  --output "$OUTPUT")
if [[ "${EFF_DRY_RUN:-0}" == 1 ]]; then cmd+=(--dry-run); fi
printf 'Efficiency benchmark: dataset=%s training_GPUs=%s output=%s\n' "$DATASET" "$TRAIN_GPUS" "$OUTPUT"
exec "${cmd[@]}"
