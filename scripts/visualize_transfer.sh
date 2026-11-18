#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --job-name=figure3_transfer
#SBATCH --output=figure3_transfer_%j.out
set -euo pipefail

# Central settings: paper A = relation baseline; paper D = old train_B full transfer.
PROJECT=/data02/home/scv7eyk/cross-model/VM-DMD
PYTHON=/data02/home/scv7eyk/.conda/envs/Femas/bin/python
DATASET="${SAB_DATASET:-isic18}"
SEED="${SAB_SEED:-42}"
SPLIT="${FIG3_SPLIT:-test}"
DIRECTION="${FIG3_DIRECTION:-row_forward}"
# 0 means every eligible image in the split; positive values limit the count.
NUM_IMAGES="${FIG3_LIMIT:-0}"
# Blank: find the single existing known result directory automatically.
# If more than one is found, enter the exact folder name here or set FIG3_RUN_A/D.
RUN_A="${FIG3_RUN_A:-}"
RUN_D="${FIG3_RUN_D:-}"
# Optional comma-separated sample IDs, WITHOUT filename extension.
SAMPLE_IDS="${FIG3_SAMPLE_IDS:-}"
case "$DATASET" in
 isic17) DATA_ROOT=/data02/home/scv7eyk/ISIC2017 ;;
 isic18) DATA_ROOT=/data02/home/scv7eyk/ISIC2018 ;;
 bus) DATA_ROOT=/data02/home/scv7eyk/Breast_BUS ;;
 busi) DATA_ROOT=/data02/home/scv7eyk/Breast_ultrasound ;;
 *) echo 'SAB_DATASET must be isic17, isic18, bus or busi' >&2; exit 2 ;;
esac
TEACHER="${SAB_TEACHER:-/data02/home/scv7eyk/cross-model/swin/work_dirs/teachers/$DATASET/exp_v1_seed42/best_teacher.pth}"
OUTPUT="$PROJECT/reports/figure3/$DATASET/${SPLIT}_${DIRECTION}_seed${SEED}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$PROJECT"
args=(--project-root "$PROJECT" --dataset "$DATASET" --seed "$SEED"
 --run-a "$RUN_A" --run-d "$RUN_D" --teacher "$TEACHER" --data-root "$DATA_ROOT"
 --split "$SPLIT" --direction "$DIRECTION" --limit "$NUM_IMAGES" --output "$OUTPUT" --device cuda)
if [[ -n "$SAMPLE_IDS" ]]; then
 IFS=',' read -r -a selected_ids <<< "$SAMPLE_IDS"
 args+=(--sample-id "${selected_ids[@]}")
fi
printf 'Figure 3: dataset=%s split=%s direction=%s output=%s\n' "$DATASET" "$SPLIT" "$DIRECTION" "$OUTPUT"
exec "$PYTHON" scripts/visualize_transfer.py "${args[@]}"
