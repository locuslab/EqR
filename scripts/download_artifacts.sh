#!/usr/bin/env bash
# Download EqR datasets and checkpoints from Hugging Face.

set -euo pipefail

DATA_REPO="${DATA_REPO:-locuslab/EqR-data}"
CKPT_REPO="${CKPT_REPO:-locuslab/EqR-model}"
CACHE_DIR="${CACHE_DIR:-downloads/eqr-artifacts}"
DATA_DIR="${DATA_DIR:-data}"
CKPT_DIR="${CKPT_DIR:-downloaded_checkpoints}"

download_data=1
download_ckpts=0
ckpt_patterns=()

usage() {
  cat <<'EOF'
Usage: scripts/download_artifacts.sh [options]

Downloads EqR datasets from Hugging Face into data/ by default. Checkpoints are
optional because they are larger and are copied to downloaded_checkpoints/.

Options:
  --with-ckpts              Download checkpoints in addition to data.
  --ckpts-only              Download only checkpoints.
  --skip-data               Do not download data.
  --data-repo REPO_ID       Hugging Face dataset repo id. Default: locuslab/EqR-data.
  --ckpt-repo REPO_ID       Hugging Face model repo id. Default: locuslab/EqR-model.
  --ckpt-pattern GLOB       Checkpoint allow-pattern. May be repeated.
                            Defaults to Sudoku and Maze checkpoints.
  --cache-dir DIR           Snapshot Hugging Face repos under DIR.
  --data-dir DIR            Copy datasets into DIR.
  --ckpt-dir DIR            Copy checkpoints into DIR.
  -h, --help                Show this help.

Environment overrides:
  DATA_REPO, CKPT_REPO, CACHE_DIR, DATA_DIR, CKPT_DIR, HF_TOKEN
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-ckpts)
      download_ckpts=1
      shift
      ;;
    --ckpts-only)
      download_data=0
      download_ckpts=1
      shift
      ;;
    --skip-data)
      download_data=0
      shift
      ;;
    --data-repo)
      DATA_REPO="$2"
      shift 2
      ;;
    --ckpt-repo)
      CKPT_REPO="$2"
      shift 2
      ;;
    --ckpt-pattern)
      ckpt_patterns+=("$2")
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --ckpt-dir)
      CKPT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${#ckpt_patterns[@]}" -eq 0 ]]; then
  ckpt_patterns=(
    "sudoku-extreme/*.pth"
    "maze-unique/*.pth"
  )
fi

copy_tree() {
  local src="$1"
  local dest="$2"

  mkdir -p "${dest}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude='.cache/' \
      --exclude='.git/' \
      --exclude='.gitattributes' \
      --exclude='README*' \
      --exclude='LICENSE*' \
      "${src}/" "${dest}/"
    return
  fi

  shopt -s dotglob nullglob
  for entry in "${src}"/*; do
    case "$(basename -- "${entry}")" in
      .cache|.git|.gitattributes|README*|LICENSE*) continue ;;
    esac
    cp -a "${entry}" "${dest}/"
  done
  shopt -u dotglob nullglob
}

download_snapshot() {
  local repo_id="$1"
  local repo_type="$2"
  local local_dir="$3"
  shift 3

  mkdir -p "${local_dir}"
  HF_REPO_ID="${repo_id}" HF_REPO_TYPE="${repo_type}" HF_LOCAL_DIR="${local_dir}" \
  python - "$@" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

allow_patterns = [arg for arg in sys.argv[1:] if arg]
snapshot_download(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type=os.environ["HF_REPO_TYPE"],
    local_dir=os.environ["HF_LOCAL_DIR"],
    allow_patterns=allow_patterns or None,
    token=os.environ.get("HF_TOKEN") or None,
)
PY
}

copy_artifact_repo() {
  local repo_dir="$1"
  local preferred_subdir="$2"
  local dest="$3"

  if [[ -d "${repo_dir}/${preferred_subdir}" ]]; then
    copy_tree "${repo_dir}/${preferred_subdir}" "${dest}"
  else
    copy_tree "${repo_dir}" "${dest}"
  fi
}

if [[ "${download_data}" -eq 1 ]]; then
  data_repo_dir="${CACHE_DIR}/EqR-data"
  echo "Downloading Hugging Face dataset ${DATA_REPO} into ${data_repo_dir}"
  download_snapshot "${DATA_REPO}" "dataset" "${data_repo_dir}"
  copy_artifact_repo "${data_repo_dir}" "data" "${DATA_DIR}"
  echo "Datasets copied to ${DATA_DIR}"
fi

if [[ "${download_ckpts}" -eq 1 ]]; then
  ckpt_repo_dir="${CACHE_DIR}/EqR-model"
  echo "Downloading Hugging Face model ${CKPT_REPO} into ${ckpt_repo_dir}"
  download_snapshot "${CKPT_REPO}" "model" "${ckpt_repo_dir}" "${ckpt_patterns[@]}"
  if [[ -d "${ckpt_repo_dir}/downloaded_checkpoints" ]]; then
    copy_tree "${ckpt_repo_dir}/downloaded_checkpoints" "${CKPT_DIR}"
  elif [[ -d "${ckpt_repo_dir}/checkpoints" ]]; then
    copy_tree "${ckpt_repo_dir}/checkpoints" "${CKPT_DIR}"
  else
    copy_tree "${ckpt_repo_dir}" "${CKPT_DIR}"
  fi
  echo "Checkpoints copied to ${CKPT_DIR}"
  find "${CKPT_DIR}" -type f \( -name '*.pth' -o -name '*.pt' -o -name '*.ckpt' \) | sort
fi
