#!/usr/bin/env bash
# Run from VM-DMD: sbatch --gpus=4 scripts/train_TransMamba_adapted.sh
set -euo pipefail
root=/data02/home/scv7eyk/cross-model/VM-DMD
comparison=/data02/home/scv7eyk/cross-model/TransMamba
export PATH="/data02/home/scv7eyk/.conda/envs/Femas/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
reference="${TM_REFERENCE:-$root/runs/isic18/stage34_relation_300_control_v1_branch_relation_seed42/config.json}"
cmd=(/data02/home/scv7eyk/.conda/envs/Femas/bin/python
  "$comparison/medical_seg/launch.py" --vmdmd-root "$root" --reference "$reference"
  --seed "${TM_SEED:-42}" --run "${TM_RUN:-transmamba_amd_adapted_300_v1}"
  --action "${TM_ACTION:-train}" --beta "${TM_BETA:-0.1}")
if [[ "${TM_DRY_RUN:-0}" == 1 ]]; then cmd+=(--dry-run); fi
exec "${cmd[@]}"
