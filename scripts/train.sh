#!/usr/bin/env bash
# Launch training from a Hydra train config.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train.sh <train-config> [pretrain.py overrides...]

Examples:
  scripts/train.sh eqr_sudoku
  scripts/train.sh train/eqr_maze_unique
  NPROC_PER_NODE=2 scripts/train.sh config/train/eqr_maze_unique.yaml
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

config_name="$1"
shift

case "${config_name}" in
  config/train/*.yaml)
    config_name="${config_name#config/}"
    config_name="${config_name%.yaml}"
    ;;
  train/*.yaml)
    config_name="${config_name%.yaml}"
    ;;
  train/*)
    ;;
  *.yaml)
    config_name="train/${config_name%.yaml}"
    ;;
  *)
    config_name="train/${config_name}"
    ;;
esac

default_nproc=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

torchrun --standalone --nproc-per-node "${NPROC_PER_NODE:-${default_nproc}}" \
  pretrain.py --config-name "${config_name}" "$@"
