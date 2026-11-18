#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --job-name=figure4_segmentation
#SBATCH --output=figure4_segmentation_%j.out
set -euo pipefail
PROJECT=/data02/home/scv7eyk/cross-model/VM-DMD
PYTHON=/data02/home/scv7eyk/.conda/envs/Femas/bin/python
DATASET="${SAB_DATASET:-isic18}"
SEED="${SAB_SEED:-42}"
SPLIT="${FIG4_SPLIT:-test}"
# Paper D = full scan-transfer model (old train_B), NOT train_D_endpoint.
# Blank: resolve the unique known run folder, as in Figure 3.
RUN_A="${FIG4_RUN_A:-${FIG3_RUN_A:-}}"
RUN_D="${FIG4_RUN_D:-${FIG3_RUN_D:-}}"
# all: PNG for every image; selected: representative images only.
# Both modes always evaluate ALL images and save both models' binary masks.
RENDER="${FIG4_RENDER:-all}"
NEAR_ZERO_PP=0.1
case "$DATASET" in
 isic17) DATA_ROOT=/data02/home/scv7eyk/ISIC2017 ;;
 isic18) DATA_ROOT=/data02/home/scv7eyk/ISIC2018 ;;
 bus) DATA_ROOT=/data02/home/scv7eyk/Breast_BUS ;;
 busi) DATA_ROOT=/data02/home/scv7eyk/Breast_ultrasound ;;
 *) echo 'Invalid SAB_DATASET' >&2; exit 2 ;;
esac
OUTPUT="$PROJECT/reports/figure4/$DATASET/${SPLIT}_seed${SEED}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$PROJECT"
printf 'Figure 4: dataset=%s split=%s render=%s output=%s\n' "$DATASET" "$SPLIT" "$RENDER" "$OUTPUT"
exec "$PYTHON" scripts/visualize_segmentation.py --project-root "$PROJECT" \
 --dataset "$DATASET" --seed "$SEED" --data-root "$DATA_ROOT" \
 --run-a "$RUN_A" --run-d "$RUN_D" --split "$SPLIT" --output "$OUTPUT" \
 --render "$RENDER" --near-zero-pp "$NEAR_ZERO_PP" --device cuda
