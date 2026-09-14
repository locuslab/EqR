#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
suite=${1:?Usage: sweep.sh <eqr-widths|eqr-h42|trm-h42|trm-h44|hrm-h24|eqr-h128-control> [cell]}
cell=${2:-${SLURM_ARRAY_TASK_ID:-0}}
[[ $cell =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid cell: $cell" >&2; exit 2; }
width=42 heads=3 world=1 noise=0.01 h_layers=0
case $suite in
  eqr-widths)
    model=eqr widths=(36 40 44) lrs=(0.001 0.002 0.003 0.004) wds=(0.1)
    ((cell < 12)) || exit 2
    width=${widths[$((cell / 4))]} heads=2
    index=$((cell % 4))
    ;;
  eqr-h42) model=eqr lrs=(0.00075 0.001 0.00125 0.0015) wds=(0.1); index=$cell ;;
  trm-h42) model=trm lrs=(0.0001 0.0002 0.0003 0.0005 0.00075 0.001) wds=(0 0.1 0.5 1); index=$cell ;;
  trm-h44) model=trm width=44 heads=2 lrs=(0.0005 0.001 0.002 0.003 0.004) wds=(0.1); index=$cell ;;
  hrm-h24) model=hrm width=24 h_layers=1 lrs=(0.0005 0.001 0.002) wds=(0 0.01 1); index=$cell ;;
  eqr-h128-control) model=eqr width=128 heads=8 world=4 noise=0.1 lrs=(0.0003) wds=(0.5); index=$cell ;;
  *) echo "Unknown suite: $suite" >&2; exit 2 ;;
esac
((index < ${#lrs[@]} * ${#wds[@]})) || { echo "Cell out of range: $cell" >&2; exit 2; }
lr=${lrs[$((index / ${#wds[@]}))]}
wd=${wds[$((index % ${#wds[@]}))]}
[[ ${NPROC_PER_NODE:-$world} == "$world" ]] || { echo "This suite requires $world ranks to match the recorded run." >&2; exit 2; }
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  [[ ${SLURM_JOB_PARTITION:-} == lowprio && ${SLURM_JOB_NUM_NODES:-} == 1 ]] || {
    echo "Expected partition=lowprio, nodes=1." >&2; exit 2;
  }
fi
export NPROC_PER_NODE=$world DISABLE_COMPILE=1
unset WANDB_RUN_ID WANDB_RESUME
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/outputs/maze_unique_supplement}
seed=${SEED:-0}
run="$model-h$width-maze30-lr$lr-wd$wd-seed$seed"
extra=()
if [[ $model == eqr ]]; then
  extra=(arch.lambda_=0.95 arch.noise_scale="$noise" arch.H_init_std=1.0 arch.L_init_std=1.0)
fi
iter=5000
[[ $suite != eqr-h128-control ]] || iter=1000
cmd=(bash scripts/train.sh "${model}_maze_unique"
  dataset.data_path="${DATA_PATH:-data/maze-30x30-unique-1k}"
  global_batch_size=768 ++gradient_accumulation_steps=1 epochs=150000 ++max_steps=150000
  train_epochs_per_iter="$iter" eval_interval_steps=5000 checkpoint_interval_steps=5000
  lr="$lr" weight_decay="$wd" lr_min_ratio=1.0 lr_warmup_steps=2000 beta1=0.9 beta2=0.95
  puzzle_emb_lr=0.0001 puzzle_emb_weight_decay=1.0 target_q_update_every=4
  arch.hidden_size="$width" arch.num_heads="$heads" arch.H_cycles=3 arch.L_cycles=4
  arch.H_layers="$h_layers" arch.L_layers=1 arch.expansion=4 arch.pos_encodings=rope
  arch.board_height=null arch.board_width=null arch.forward_dtype=bfloat16 arch.mlp_t=false
  arch.puzzle_emb_ndim="$width" arch.puzzle_emb_len=16
  arch.halt_max_steps=16 arch.halt_exploration_prob=0.1 ++arch.no_ACT_continue=true
  ema=true ema_rate=0.999 gradient_checkpoint=false
  ++seed="$seed" ++run_name="$run" ++wandb_mode="${WANDB_MODE:-offline}"
  ++resume=false ++load_checkpoint=null load_weights_only=false "${extra[@]}")
printf 'partition=%s nodes=1 world_size=%s dp=%s local_batch=%s grad_accum=1 global_batch=768 seq_len=900 global_tokens_per_step=691200\n' \
  "${SLURM_JOB_PARTITION:-local}" "$world" "$world" "$((768 / world))"
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  exec "${cmd[@]}"
fi
