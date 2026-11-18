#!/usr/bin/env bash
# Fixed ISIC18 protocol. Submit: sbatch --gpus=4 scripts/train_D_endpoint.sh
set -euo pipefail

# Explicit protocol: legacy training environment variables are not consumed.
project=/data02/home/scv7eyk/cross-model/VM-DMD
python_bin=/data02/home/scv7eyk/.conda/envs/Femas/bin/python
seed="${SAB_SEED:-42}"
action="${SAB_ACTION:-train}"
run="${SAB_RUN:-sab_D_endpoint_300_v1}"
[[ "$seed" =~ ^[0-9]+$ ]] || { echo 'SAB_SEED must be a nonnegative integer' >&2; exit 2; }
case "$action" in train|preflight|test) ;; *) echo 'SAB_ACTION must be train, preflight, or test' >&2; exit 2 ;; esac
[[ "$run" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid SAB_RUN' >&2; exit 2; }
export PATH="/data02/home/scv7eyk/.conda/envs/Femas/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cmd=("$python_bin" scripts/experiment.py "$action"
  --dataset isic18 --experiment branch_relation --student-arch shallow
  --data-root /data02/home/scv7eyk/ISIC2018
  --teacher /data02/home/scv7eyk/cross-model/swin/work_dirs/teachers/isic18/exp_v1_seed42/best_teacher.pth
  --student-pretrained "$project/weights/vmamba_tiny_e292.pth"
  --run "$run" --seed "$seed" --epochs 300 --gpus 4
  --batch-size 16 --workers 4 --lr 0.0001 --beta 0.1
  --stage 3 --relation-radius 1 --relation-purity 0.9
  --kd-stages 2 3 --kd-stage-weights 0.5 0.5 --scan-transfer-variant endpoint --scan-transfer-weight 0.01)
printf 'Experiment D: action=%s seed=%s run=%s\n' "$action" "$seed" "$run"
printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
# Inspect the resolved command without loading models or creating run files.
if [[ "${SAB_DRY_RUN:-0}" == 1 ]]; then exit 0; fi
cd "$project"
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 2; }
# Never resume or overwrite an existing training run implicitly.
output="$project/runs/isic18/${run}_branch_relation_seed${seed}"
if [[ "$action" == train && -d "$output" ]]; then
  echo "Run already exists: $output. Use a new SAB_RUN for an independent run." >&2
  exit 2
fi
# Calibrate on seed 42 training batches, never on validation data.
# This subprocess exits before the distributed training process is launched.
if [[ "$action" == train ]]; then
  calibration_run="${run}_calibration_s${seed}"
  calibration_cmd=("${cmd[@]}")
  calibration_cmd[2]=preflight
  "${calibration_cmd[@]}" --seed 42 --gpus 1 --run "$calibration_run"
  report="$project/reports/${calibration_run}_branch_relation_seed42_preflight.json"
  weight=$("$python_bin" - "$report" endpoint <<'PY'
import json,math,sys
with open(sys.argv[1]) as f: report=json.load(f)
c=report['scan_transfer_calibration']
w=c['recommended_weight']
if c.get('variant')!=sys.argv[2] or c.get('calibration_seed')!=42:
    raise SystemExit('Calibration variant/seed mismatch')
if not math.isfinite(w) or w<=0:raise SystemExit('Invalid calibrated weight')
print(w)
PY
  )
  cmd+=(--scan-transfer-weight "$weight")
  echo "Calibrated variant=endpoint weight=$weight report=$report"
fi
printf 'Final command:'; printf ' %q' "${cmd[@]}"; printf '\n'
exec "${cmd[@]}"
