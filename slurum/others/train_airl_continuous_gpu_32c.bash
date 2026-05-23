#!/bin/bash
#SBATCH --job-name=ps_airl_stage1_50veh
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_airl_stage1_50veh_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_airl_stage1_50veh_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/script_pretrain/train_airl_continuous_gpu_32c_stage1_50veh.bash"
