#!/bin/bash
#SBATCH --job-name=ps_unified_expert_train
#SBATCH --account=bt60
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_unified_collect_train_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_unified_collect_train_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/script_data_collection/collect_ps_unified_expert_continuous_train.bash"
