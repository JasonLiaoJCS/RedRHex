#!/usr/bin/env bash
set -euo pipefail

# RedRhex 5-stage overnight pipeline.
# Runs Stage1 -> Stage5 sequentially and auto-resumes from previous checkpoint.

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-Template-Redrhex-Direct-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
HEADLESS="${HEADLESS:-1}"
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl/redrhex_wheg}"
RUN_TAG="${RUN_TAG:-nightly_$(date +%Y%m%d_%H%M%S)}"

# Optional GUI preflight before overnight headless run.
PRECHECK_GUI="${PRECHECK_GUI:-0}"
PRECHECK_STAGE="${PRECHECK_STAGE:-1}"
PRECHECK_ENVS="${PRECHECK_ENVS:-64}"
PRECHECK_ITERS="${PRECHECK_ITERS:-120}"

STAGE1_ITERS="${STAGE1_ITERS:-8000}"
STAGE2_ITERS="${STAGE2_ITERS:-8000}"
STAGE3_ITERS="${STAGE3_ITERS:-9000}"
STAGE4_ITERS="${STAGE4_ITERS:-10000}"
STAGE5_ITERS="${STAGE5_ITERS:-12000}"

# Space-separated Hydra overrides.
EXTRA_OVERRIDES_STR="${EXTRA_OVERRIDES_STR:-env.draw_debug_vis=False env.dr_try_physical_material_randomization=False}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/rsl_rl/train_stage_pipeline.sh [options] [-- <extra train.py args>]

Options:
  --python <bin>          Python executable (default: $PYTHON_BIN)
  --task <name>           Isaac task name
  --num_envs <n>          Number of environments
  --headless <0|1>        Use --headless (default: 1)
  --run_tag <tag>         Unique suffix used in run_name per stage
  --precheck_gui <0|1>    Run a short non-headless visual precheck first (default: 0)
  --precheck_stage <n>    Stage index for visual precheck (default: 1)
  --precheck_envs <n>     Num envs for visual precheck (default: 64)
  --precheck_iters <n>    Max iterations for visual precheck (default: 120)
  --s1 <iters>            Stage1 max_iterations
  --s2 <iters>            Stage2 max_iterations
  --s3 <iters>            Stage3 max_iterations
  --s4 <iters>            Stage4 max_iterations
  --s5 <iters>            Stage5 max_iterations
  --extra "<overrides>"   Extra Hydra overrides string
  -h, --help              Show this help

Examples:
  bash scripts/rsl_rl/train_stage_pipeline.sh
  bash scripts/rsl_rl/train_stage_pipeline.sh --run_tag mynight --num_envs 2048 --s5 15000
  bash scripts/rsl_rl/train_stage_pipeline.sh --extra "env.domain_randomization_enable=False"
USAGE
}

ADDITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --num_envs) NUM_ENVS="$2"; shift 2 ;;
    --headless) HEADLESS="$2"; shift 2 ;;
    --run_tag) RUN_TAG="$2"; shift 2 ;;
    --precheck_gui) PRECHECK_GUI="$2"; shift 2 ;;
    --precheck_stage) PRECHECK_STAGE="$2"; shift 2 ;;
    --precheck_envs) PRECHECK_ENVS="$2"; shift 2 ;;
    --precheck_iters) PRECHECK_ITERS="$2"; shift 2 ;;
    --s1) STAGE1_ITERS="$2"; shift 2 ;;
    --s2) STAGE2_ITERS="$2"; shift 2 ;;
    --s3) STAGE3_ITERS="$2"; shift 2 ;;
    --s4) STAGE4_ITERS="$2"; shift 2 ;;
    --s5) STAGE5_ITERS="$2"; shift 2 ;;
    --extra) EXTRA_OVERRIDES_STR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; ADDITIONAL_ARGS+=("$@"); break ;;
    *)
      echo "[ERROR] Unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

IFS=' ' read -r -a EXTRA_OVERRIDES <<< "$EXTRA_OVERRIDES_STR"

mkdir -p "$LOG_ROOT"
mkdir -p logs/rsl_rl/pipeline
PIPELINE_LOG="logs/rsl_rl/pipeline/${RUN_TAG}.log"
echo "[INFO] Pipeline log: $PIPELINE_LOG"
echo "[INFO] RUN_TAG: $RUN_TAG" | tee -a "$PIPELINE_LOG"
echo "[INFO] TASK: $TASK, NUM_ENVS: $NUM_ENVS" | tee -a "$PIPELINE_LOG"
echo "[INFO] ITERS: s1=$STAGE1_ITERS s2=$STAGE2_ITERS s3=$STAGE3_ITERS s4=$STAGE4_ITERS s5=$STAGE5_ITERS" | tee -a "$PIPELINE_LOG"
echo "[INFO] PRECHECK: gui=$PRECHECK_GUI stage=$PRECHECK_STAGE envs=$PRECHECK_ENVS iters=$PRECHECK_ITERS" | tee -a "$PIPELINE_LOG"

LAST_RUN=""
LAST_CKPT=""
LAST_CKPT_ABS=""

latest_run_by_suffix() {
  local suffix="$1"
  ls -td "$LOG_ROOT"/*_"$suffix" 2>/dev/null | head -1 || true
}

latest_ckpt_in_run() {
  local run_dir="$1"
  ls -v "$run_dir"/model_*.pt 2>/dev/null | tail -1 || true
}

run_precheck() {
  if [[ "$PRECHECK_GUI" != "1" ]]; then
    return
  fi

  local precheck_name="${RUN_TAG}_precheck_stage${PRECHECK_STAGE}"
  local cmd=(
    "$PYTHON_BIN" scripts/rsl_rl/train.py
    "--task=${TASK}"
    "--num_envs=${PRECHECK_ENVS}"
    "--max_iterations=${PRECHECK_ITERS}"
    "--run_name=${precheck_name}"
    "env.stage=${PRECHECK_STAGE}"
  )
  for ov in "${EXTRA_OVERRIDES[@]}"; do
    cmd+=("$ov")
  done
  # Force debug visualization ON for the visual sanity check.
  cmd+=("env.draw_debug_vis=True")
  for extra in "${ADDITIONAL_ARGS[@]}"; do
    cmd+=("$extra")
  done

  echo "" | tee -a "$PIPELINE_LOG"
  echo "==================================================" | tee -a "$PIPELINE_LOG"
  echo "[INFO] GUI precheck start (non-headless)" | tee -a "$PIPELINE_LOG"
  echo "[INFO] Command: ${cmd[*]}" | tee -a "$PIPELINE_LOG"
  echo "==================================================" | tee -a "$PIPELINE_LOG"
  "${cmd[@]}" 2>&1 | tee -a "$PIPELINE_LOG"
  echo "[INFO] GUI precheck done. Starting staged overnight training..." | tee -a "$PIPELINE_LOG"
}

run_stage() {
  local stage="$1"
  local iters="$2"
  local load_run="${3:-}"
  local load_ckpt="${4:-}"
  local stage_suffix="${RUN_TAG}_stage${stage}"
  local run_name="${stage_suffix}"

  local cmd=(
    "$PYTHON_BIN" scripts/rsl_rl/train.py
    "--task=${TASK}"
    "--num_envs=${NUM_ENVS}"
    "--max_iterations=${iters}"
    "--run_name=${run_name}"
    "env.stage=${stage}"
  )
  if [[ "$HEADLESS" == "1" ]]; then
    cmd+=(--headless)
  fi
  if [[ -n "$load_run" && -n "$load_ckpt" ]]; then
    cmd+=(--resume "--load_run=${load_run}" "--checkpoint=${load_ckpt}")
  fi
  for ov in "${EXTRA_OVERRIDES[@]}"; do
    cmd+=("$ov")
  done
  for extra in "${ADDITIONAL_ARGS[@]}"; do
    cmd+=("$extra")
  done

  echo "" | tee -a "$PIPELINE_LOG"
  echo "==================================================" | tee -a "$PIPELINE_LOG"
  echo "[INFO] Stage ${stage} start: run_name=${run_name}" | tee -a "$PIPELINE_LOG"
  echo "[INFO] Command: ${cmd[*]}" | tee -a "$PIPELINE_LOG"
  echo "==================================================" | tee -a "$PIPELINE_LOG"

  "${cmd[@]}" 2>&1 | tee -a "$PIPELINE_LOG"

  local run_dir
  run_dir="$(latest_run_by_suffix "$stage_suffix")"
  if [[ -z "$run_dir" ]]; then
    echo "[ERROR] Cannot find run directory for stage ${stage} (suffix=${stage_suffix})." | tee -a "$PIPELINE_LOG"
    exit 1
  fi

  local ckpt_abs
  ckpt_abs="$(latest_ckpt_in_run "$run_dir")"
  if [[ -z "$ckpt_abs" ]]; then
    echo "[ERROR] Cannot find checkpoint in: $run_dir" | tee -a "$PIPELINE_LOG"
    exit 1
  fi

  LAST_RUN="$(basename "$run_dir")"
  LAST_CKPT="$(basename "$ckpt_abs")"
  LAST_CKPT_ABS="$ckpt_abs"

  echo "[INFO] Stage ${stage} done: RUN=${LAST_RUN}, CKPT=${LAST_CKPT}" | tee -a "$PIPELINE_LOG"
}

run_precheck
run_stage 1 "$STAGE1_ITERS"
run_stage 2 "$STAGE2_ITERS" "$LAST_RUN" "$LAST_CKPT"
run_stage 3 "$STAGE3_ITERS" "$LAST_RUN" "$LAST_CKPT"
run_stage 4 "$STAGE4_ITERS" "$LAST_RUN" "$LAST_CKPT"
run_stage 5 "$STAGE5_ITERS" "$LAST_RUN" "$LAST_CKPT"

echo "" | tee -a "$PIPELINE_LOG"
echo "[DONE] 5-stage pipeline complete." | tee -a "$PIPELINE_LOG"
echo "[DONE] FINAL_RUN=$LAST_RUN" | tee -a "$PIPELINE_LOG"
echo "[DONE] FINAL_CKPT=$LAST_CKPT_ABS" | tee -a "$PIPELINE_LOG"
echo "" | tee -a "$PIPELINE_LOG"
echo "Use this checkpoint for eval/play:"
echo "$LAST_CKPT_ABS"
