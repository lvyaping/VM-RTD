#!/usr/bin/env bash
# Fixed shallow protocol; SAB_DATASET=isic17 or isic18 (default). Submit: sbatch --gpus=4 scripts/train_transfer_only.sh
set -euo pipefail

# Explicit protocol: legacy training environment variables are not consumed.
project=/data02/home/scv7eyk/cross-model/VM-DMD
python_bin=/data02/home/scv7eyk/.conda/envs/Femas/bin/python
dataset="${SAB_DATASET:-isic18}"
case "$dataset" in
  isic17) data_root=/data02/home/scv7eyk/ISIC2017 ;;
  isic18) data_root=/data02/home/scv7eyk/ISIC2018 ;;
  *) echo 'SAB_DATASET must be isic17 or isic18' >&2; exit 2 ;;
esac
teacher="${SAB_TEACHER:-/data02/home/scv7eyk/cross-model/swin/work_dirs/teachers/$dataset/exp_v1_seed42/best_teacher.pth}"
seed="${SAB_SEED:-42}"
action="${SAB_ACTION:-train}"
run="${SAB_RUN:-transfer_only_stage3_300_v1}"
[[ "$seed" =~ ^[0-9]+$ ]] || { echo 'SAB_SEED must be a nonnegative integer' >&2; exit 2; }
case "$action" in train|preflight|test) ;; *) echo 'SAB_ACTION must be train, preflight, or test' >&2; exit 2 ;; esac
[[ "$run" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid SAB_RUN' >&2; exit 2; }
export PATH="/data02/home/scv7eyk/.conda/envs/Femas/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cmd=("$python_bin" scripts/experiment.py "$action"
  --dataset "$dataset" --experiment branch_relation --student-arch shallow
  --data-root "$data_root"
  --teacher "$teacher"
  --student-pretrained "$project/weights/vmamba_tiny_e292.pth"
  --run "$run" --seed "$seed" --epochs 300 --gpus 4
  --batch-size 16 --workers 4 --lr 0.0001 --beta 0.1
  --stage 3 --relation-radius 1 --relation-purity 0.9
  --kd-stages 2 3 --kd-stage-weights 0.5 0.5 --scan-transfer-weight 0.015216837498380707 --scan-transfer-only)
printf 'Dataset=%s data=%s teacher=%s\n' "$dataset" "$data_root" "$teacher"
printf 'Transfer only: action=%s seed=%s run=%s\n' "$action" "$seed" "$run"
printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
# Inspect the resolved command without loading models or creating run files.
if [[ "${SAB_DRY_RUN:-0}" == 1 ]]; then exit 0; fi
cd "$project"
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 2; }
# Never resume or overwrite an existing training run implicitly.
output="$project/runs/$dataset/${run}_branch_relation_seed${seed}"
if [[ "$action" == train && -d "$output" ]]; then
  echo "Run already exists: $output. Use a new SAB_RUN for an independent run." >&2
  exit 2
fi
exec "${cmd[@]}"
