#!/usr/bin/env bash
# Evaluate a checkpoint with the shared depth/breadth evaluation config.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/eval.sh <checkpoint-path> [evaluate.py overrides...]

Examples:
  scripts/eval.sh downloaded_checkpoints/sudoku-extreme/eqr.pth
  scripts/eval.sh /path/to/checkpoint.pth halt_max_steps=64
  scripts/eval.sh /path/to/checkpoint.pth halt_max_steps=64 different_init=128 convergence_top_k=1 global_batch_size=16
EOF
}

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

checkpoint_path="$1"
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python evaluate.py eval_yaml=config/eval/depth_breadth.yaml checkpoint="${checkpoint_path}" "$@"
