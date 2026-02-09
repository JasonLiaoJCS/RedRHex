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
START_STAGE="${START_STAGE:-1}"

# Stage handoff behavior (recommended for stability between different command distributions)
RESUME_POLICY_ONLY="${RESUME_POLICY_ONLY:-1}"
RESET_ACTION_STD="${RESET_ACTION_STD:-0.8}"

# Resolve Python interpreter that can import isaaclab.
if ! "$PYTHON_BIN" -c "import isaaclab" >/dev/null 2>&1; then
  FALLBACK_PY="/home/jasonliao/miniconda3/envs/env_isaaclab/bin/python"
  if [[ -x "$FALLBACK_PY" ]] && "$FALLBACK_PY" -c "import isaaclab" >/dev/null 2>&1; then
    echo "[WARN] '$PYTHON_BIN' cannot import isaaclab. Falling back to: $FALLBACK_PY"
    PYTHON_BIN="$FALLBACK_PY"
  fi
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "[WARN] 'rg' not found. Using grep fallback for pipeline log parsing."
fi

# Stability gate: fail fast if a stage is clearly unstable.
STABILITY_GATE="${STABILITY_GATE:-1}"
STABILITY_GATE_STRICT="${STABILITY_GATE_STRICT:-0}"
STAGE1_MIN_EP_LEN="${STAGE1_MIN_EP_LEN:-55}"
STAGE2_MIN_EP_LEN="${STAGE2_MIN_EP_LEN:-45}"
STAGE3_MIN_EP_LEN="${STAGE3_MIN_EP_LEN:-50}"
STAGE4_MIN_EP_LEN="${STAGE4_MIN_EP_LEN:-40}"
STAGE5_MIN_EP_LEN="${STAGE5_MIN_EP_LEN:-50}"
STAGE1_MAX_TERMINATED="${STAGE1_MAX_TERMINATED:-30}"
STAGE2_MAX_TERMINATED="${STAGE2_MAX_TERMINATED:-30}"
STAGE3_MAX_TERMINATED="${STAGE3_MAX_TERMINATED:-30}"
STAGE4_MAX_TERMINATED="${STAGE4_MAX_TERMINATED:-35}"
STAGE5_MAX_TERMINATED="${STAGE5_MAX_TERMINATED:-35}"

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
  --start_stage <1..5>    Start pipeline from this stage (default: 1)
  --resume_policy_only <0|1>
                           Resume stage handoff with policy-only load (default: 1)
  --reset_action_std <v>  Action std reset value when policy-only resume is enabled (default: 0.8)
  --stability_gate <0|1>  Enable stage health gate (default: 1)
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
    --start_stage) START_STAGE="$2"; shift 2 ;;
    --resume_policy_only) RESUME_POLICY_ONLY="$2"; shift 2 ;;
    --reset_action_std) RESET_ACTION_STD="$2"; shift 2 ;;
    --stability_gate) STABILITY_GATE="$2"; shift 2 ;;
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
echo "[INFO] START_STAGE: $START_STAGE" | tee -a "$PIPELINE_LOG"
echo "[INFO] PRECHECK: gui=$PRECHECK_GUI stage=$PRECHECK_STAGE envs=$PRECHECK_ENVS iters=$PRECHECK_ITERS" | tee -a "$PIPELINE_LOG"
echo "[INFO] STABILITY_GATE: $STABILITY_GATE" | tee -a "$PIPELINE_LOG"
echo "[INFO] HANDOFF: resume_policy_only=$RESUME_POLICY_ONLY reset_action_std=$RESET_ACTION_STD" | tee -a "$PIPELINE_LOG"

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

find_last_line() {
  local needle="$1"
  local file="$2"
  local line=""
  if command -v rg >/dev/null 2>&1; then
    line="$(rg --no-messages -F "$needle" "$file" 2>/dev/null | tail -1 || true)"
  fi
  if [[ -z "$line" ]]; then
    line="$(grep -F "$needle" "$file" 2>/dev/null | tail -1 || true)"
  fi
  echo "$line"
}

stage_min_ep_len() {
  local stage="$1"
  case "$stage" in
    1) echo "$STAGE1_MIN_EP_LEN" ;;
    2) echo "$STAGE2_MIN_EP_LEN" ;;
    3) echo "$STAGE3_MIN_EP_LEN" ;;
    4) echo "$STAGE4_MIN_EP_LEN" ;;
    5) echo "$STAGE5_MIN_EP_LEN" ;;
    *) echo "40" ;;
  esac
}

stage_max_terminated() {
  local stage="$1"
  case "$stage" in
    1) echo "$STAGE1_MAX_TERMINATED" ;;
    2) echo "$STAGE2_MAX_TERMINATED" ;;
    3) echo "$STAGE3_MAX_TERMINATED" ;;
    4) echo "$STAGE4_MAX_TERMINATED" ;;
    5) echo "$STAGE5_MAX_TERMINATED" ;;
    *) echo "40" ;;
  esac
}

stage_iters() {
  local stage="$1"
  case "$stage" in
    1) echo "$STAGE1_ITERS" ;;
    2) echo "$STAGE2_ITERS" ;;
    3) echo "$STAGE3_ITERS" ;;
    4) echo "$STAGE4_ITERS" ;;
    5) echo "$STAGE5_ITERS" ;;
    *) echo "0" ;;
  esac
}

validate_stage_health() {
  local stage="$1"
  local stage_log="$2"

  if [[ "$STABILITY_GATE" != "1" ]]; then
    echo "[INFO] Stage ${stage} health gate disabled." | tee -a "$PIPELINE_LOG"
    return
  fi

  local min_ep max_term
  min_ep="$(stage_min_ep_len "$stage")"
  max_term="$(stage_max_terminated "$stage")"

  local ep_line term_line ep_len term_cnt
  ep_line="$(find_last_line "Mean episode length:" "$stage_log")"
  term_line="$(find_last_line "Episode_Termination/terminated:" "$stage_log")"

  # 有些環境下 stage log 可能缺最後幾行，退回到 pipeline 總 log 再抓一次
  if [[ -z "$ep_line" ]]; then
    ep_line="$(find_last_line "Mean episode length:" "$PIPELINE_LOG")"
  fi
  if [[ -z "$term_line" ]]; then
    term_line="$(find_last_line "Episode_Termination/terminated:" "$PIPELINE_LOG")"
  fi

  if [[ -z "$ep_line" && -z "$term_line" ]]; then
    if [[ "$STABILITY_GATE_STRICT" == "1" ]]; then
      echo "[ERROR] Stage ${stage} health gate: missing metrics in $stage_log" | tee -a "$PIPELINE_LOG"
      exit 1
    fi
    echo "[WARN] Stage ${stage} health gate: missing metrics in $stage_log, skip this gate." | tee -a "$PIPELINE_LOG"
    return
  fi

  ep_len=""
  term_cnt=""
  if [[ -n "$ep_line" ]]; then
    ep_len="$(echo "$ep_line" | sed -E 's/.*Mean episode length:[[:space:]]*([0-9.]+).*/\1/')"
  else
    echo "[WARN] Stage ${stage} health: missing 'Mean episode length' in $stage_log. Skipping this check." | tee -a "$PIPELINE_LOG"
  fi
  if [[ -n "$term_line" ]]; then
    term_cnt="$(echo "$term_line" | sed -E 's/.*Episode_Termination\/terminated:[[:space:]]*([0-9.]+).*/\1/')"
  else
    echo "[WARN] Stage ${stage} health: missing 'Episode_Termination/terminated' in $stage_log. Skipping this check." | tee -a "$PIPELINE_LOG"
  fi

  echo "[INFO] Stage ${stage} health: mean_episode_length=${ep_len:-N/A}, terminated=${term_cnt:-N/A}" | tee -a "$PIPELINE_LOG"
  echo "[INFO] Stage ${stage} thresholds: min_episode_length=${min_ep}, max_terminated=${max_term}" | tee -a "$PIPELINE_LOG"

  if [[ -n "$ep_len" ]]; then
    if ! awk -v x="$ep_len" -v y="$min_ep" 'BEGIN{exit !(x+0 >= y+0)}'; then
      echo "[ERROR] Stage ${stage} unstable: mean_episode_length=${ep_len} < ${min_ep}" | tee -a "$PIPELINE_LOG"
      exit 1
    fi
  fi
  if [[ -n "$term_cnt" ]]; then
    if ! awk -v x="$term_cnt" -v y="$max_term" 'BEGIN{exit !(x+0 <= y+0)}'; then
      echo "[ERROR] Stage ${stage} unstable: terminated=${term_cnt} > ${max_term}" | tee -a "$PIPELINE_LOG"
      exit 1
    fi
  fi

  echo "[INFO] Stage ${stage} health gate PASS." | tee -a "$PIPELINE_LOG"
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
    if [[ "$RESUME_POLICY_ONLY" == "1" ]]; then
      cmd+=(--resume_policy_only "--reset_action_std=${RESET_ACTION_STD}")
    fi
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

  local stage_log="logs/rsl_rl/pipeline/${RUN_TAG}_stage${stage}.log"
  "${cmd[@]}" 2>&1 | tee "$stage_log" | tee -a "$PIPELINE_LOG"

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

  validate_stage_health "$stage" "$stage_log"

  echo "[INFO] Stage ${stage} done: RUN=${LAST_RUN}, CKPT=${LAST_CKPT}" | tee -a "$PIPELINE_LOG"
}

if ! [[ "$START_STAGE" =~ ^[1-5]$ ]]; then
  echo "[ERROR] --start_stage must be an integer in [1,5], got: $START_STAGE" | tee -a "$PIPELINE_LOG"
  exit 1
fi

if [[ "$START_STAGE" -eq 1 ]]; then
  run_precheck
else
  local_prev_stage=$((START_STAGE - 1))
  local_prev_suffix="${RUN_TAG}_stage${local_prev_stage}"
  local_prev_run_dir="$(latest_run_by_suffix "$local_prev_suffix")"
  if [[ -z "$local_prev_run_dir" ]]; then
    echo "[ERROR] Cannot resume from stage ${START_STAGE}: previous stage run not found for suffix '${local_prev_suffix}'" | tee -a "$PIPELINE_LOG"
    exit 1
  fi
  local_prev_ckpt_abs="$(latest_ckpt_in_run "$local_prev_run_dir")"
  if [[ -z "$local_prev_ckpt_abs" ]]; then
    echo "[ERROR] Cannot resume from stage ${START_STAGE}: no checkpoint found in ${local_prev_run_dir}" | tee -a "$PIPELINE_LOG"
    exit 1
  fi
  LAST_RUN="$(basename "$local_prev_run_dir")"
  LAST_CKPT="$(basename "$local_prev_ckpt_abs")"
  LAST_CKPT_ABS="$local_prev_ckpt_abs"
  echo "[INFO] Resuming pipeline from stage ${START_STAGE} using ${LAST_CKPT_ABS}" | tee -a "$PIPELINE_LOG"
fi

for stage in $(seq "$START_STAGE" 5); do
  iters="$(stage_iters "$stage")"
  if [[ "$stage" -eq 1 ]]; then
    run_stage "$stage" "$iters"
  else
    run_stage "$stage" "$iters" "$LAST_RUN" "$LAST_CKPT"
  fi
done

echo "" | tee -a "$PIPELINE_LOG"
echo "[DONE] 5-stage pipeline complete." | tee -a "$PIPELINE_LOG"
echo "[DONE] FINAL_RUN=$LAST_RUN" | tee -a "$PIPELINE_LOG"
echo "[DONE] FINAL_CKPT=$LAST_CKPT_ABS" | tee -a "$PIPELINE_LOG"
echo "" | tee -a "$PIPELINE_LOG"
echo "Use this checkpoint for eval/play:"
echo "$LAST_CKPT_ABS"
