#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --job-name=busi_teacher_test
#SBATCH --output=busi_teacher_test_%j.out
set -euo pipefail

project=/data02/home/scv7eyk/cross-model/swin
python_bin=/data02/home/scv7eyk/.conda/envs/open-mmlab/bin/python
data_root=/data02/home/scv7eyk/Breast_ultrasound
checkpoint="$project/work_dirs/teachers/busi/exp_v1_seed42/best_teacher.pth"
export PATH="/data02/home/scv7eyk/.conda/envs/open-mmlab/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
[[ -d "$data_root/test/image" && -d "$data_root/test/mask" ]] || { echo "Missing test/image or test/mask under $data_root" >&2; exit 1; }
cd "$project"
printf 'BUSI teacher test\nCheckpoint: %s\nData: %s\n' "$checkpoint" "$data_root"
exec "$python_bin" scripts/teacher.py test busi \
  --data-root "$data_root" --gpus 1 --run exp_v1 --seed 42 \
  --checkpoint "$checkpoint"
