# RedRhex 終極操作指南

更新日期：2026-04-16

這份文件是 RedRhex 專案目前唯一的操作手冊，目標不是講理論，而是讓一個第一次碰這個專案的人也能知道：

- 要先做什麼檢查
- 怎麼開始 train
- 怎麼用 play 看模型
- 怎麼匯出 policy
- 怎麼跑五階段 curriculum
- 怎麼用 teacher / distillation
- 怎麼做 smoke test、command sweep、除錯

如果你要看「這次改革背後參考了哪些論文、理論、程式修改細節、reward 設計、驗證結果」，請看：

- `docs/2026_Midterm.md`
- `docs/redrhex_improvement_strategy_full.md`
- `docs/redrhex_forwardfast_professor_report.md`

這份文件只負責把操作流程講清楚。

---

## 1. 你先要知道的事

目前專案有兩個主要 task：

| 用途 | Task ID | 說明 |
|---|---|---|
| 完整能力訓練 | `Template-Redrhex-Direct-v0` | 主任務，包含 rough terrain、history actor、asymmetric critic、teacher obs、symmetry augmentation、actuator randomization、fault injection 等改革 |
| 快速直走收斂 | `Template-Redrhex-ForwardFast-Direct-v0` | forward-only 快速版，適合現場快速迭代與 sim2real 直走調參 |

目前 `train.py` / `play.py` 都支援三種 agent 入口：

| `--agent` 值 | 用途 | 可否直接當部署學生策略 |
|---|---|---|
| `rsl_rl_cfg_entry_point` | 一般 PPO，actor 吃 `policy + history`，critic 吃 privileged obs | 可以，這是最常用的部署路線 |
| `rsl_rl_teacher_cfg_entry_point` | privileged teacher PPO，actor/critic 都吃 `teacher` obs | 不建議直接當真機部署策略，主要用來做 teacher 上界或蒸餾 |
| `rsl_rl_distillation_cfg_entry_point` | teacher-student distillation | 可以，這是把 teacher 壓縮成可部署 student 的路線 |

最重要的實務觀念：

- `rsl_rl_cfg_entry_point` 已經是改革後的主力設定，不是舊版 baseline。
- 現在的主 task 內建 history、critic privileged obs、teacher obs、對稱增強、rough terrain、故障注入與 actuator randomization，不需要另外開很多旗標才會生效。
- `play.py` 現在每次載入 checkpoint 都會自動匯出 `policy.pt` 和 `policy.onnx` 到該 run 的 `exported/` 目錄。
- `play.py` 和 `eval_command_sweep.py` 會自動幫你把錯誤的 checkpoint 參數修正成真正的 `model_*.pt`。就算你誤傳資料夾、`events.out.tfevents...` 或 `exported/policy.pt`，它也會盡量 fallback 到同 run 裡最新的訓練 checkpoint。

---

## 2. 先做環境準備

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

建議第一次操作前先做一次最小檢查：

```bash
bash -n scripts/rsl_rl/train_stage_pipeline.sh

python -m py_compile \
  scripts/rsl_rl/train.py \
  scripts/rsl_rl/play.py \
  scripts/rsl_rl/eval_command_sweep.py \
  scripts/rsl_rl/validate_reform_stack.py \
  scripts/rsl_rl/validate_distillation_stack.py
```

如果你只是想先確認新改革的訓練堆疊能不能跑，再加做這個 smoke test：

```bash
python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 32
```

這個腳本會檢查：

- 環境能否成功 reset / rollout
- `policy/history/critic/teacher` observation 維度是否一致
- reward 和 observation 是否有 NaN / Inf
- rough terrain / fault injection / actuator scaling 是否真的在跑

---

## 3. 最短成功路徑

如果你完全是第一次使用這個專案，推薦照這個順序：

1. 跑 `validate_reform_stack.py`，確認環境與 observation stack 正常。
2. 選一條訓練路線：
   - 想先快速拿到直走模型：用 `Template-Redrhex-ForwardFast-Direct-v0`
   - 想直接做完整能力模型：用 `Template-Redrhex-Direct-v0`
3. 開 TensorBoard 看訓練。
4. 用 `play.py` 載入最新 checkpoint。
5. 從 `exported/policy.onnx` 或 `exported/policy.pt` 拿部署檔。

---

## 4. 日誌、checkpoint、輸出檔會放在哪裡

所有 RSL-RL 訓練都會寫到：

```text
logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
```

常見 experiment root：

| 模式 | experiment root |
|---|---|
| 主 task PPO | `logs/rsl_rl/redrhex_wheg/` |
| ForwardFast PPO | `logs/rsl_rl/redrhex_forward_fast/` |
| 主 task teacher PPO | `logs/rsl_rl/redrhex_wheg_teacher/` |
| ForwardFast teacher PPO | `logs/rsl_rl/redrhex_forward_fast_teacher/` |
| 主 task distillation | `logs/rsl_rl/redrhex_wheg_distill/` |
| ForwardFast distillation | `logs/rsl_rl/redrhex_forward_fast_distill/` |

每個 run 內通常有：

- `model_*.pt`：真正可拿來 resume / play / eval 的訓練 checkpoint
- `events.out.tfevents...`：TensorBoard 日誌，不是模型
- `params/env.yaml`、`params/agent.yaml`：該次訓練實際用到的設定
- `exported/policy.pt`：JIT 匯出
- `exported/policy.onnx`：ONNX 匯出

實務上要記住：

- 能拿來 `--checkpoint` 的主角永遠是 `model_*.pt`
- `events.out.tfevents...` 不能拿去 `play.py`
- `exported/policy.pt` 不是訓練 checkpoint，不能拿來 resume 訓練

---

## 5. 最常用的訓練指令

## 5.1 主 task 一般 PPO

這是目前最推薦的主線訓練入口。它用 deployable actor 設計：

- actor：`policy + history`
- critic：`policy + history + critic privileged obs`

也就是說，訓練時 critic 會看到比較多資訊，但最後部署時 actor 不需要 teacher-only 的特權資訊。

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name wheg_locomotion_reform_v1
```

如果你要把 env 數量拉高：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 4096 \
  --max_iterations 2500 \
  --run_name wheg_locomotion_reform_v1_big
```

## 5.2 ForwardFast 快速直走 PPO

這條路線適合：

- 現場想快速得到可上機的直走 policy
- reward / 物理參數剛調完，想快點看到行為變化
- 不想一開始就訓完整 mixed locomotion

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-ForwardFast-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 1500 \
  --run_name forward_fast_reform_v1
```

如果 GPU 夠，可以加到 4096 env：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-ForwardFast-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 4096 \
  --max_iterations 1500 \
  --run_name forward_fast_reform_v1_big
```

## 5.3 主 task 五階段 curriculum

如果你要完整能力整合，推薦直接使用五階段 pipeline，而不是手動一段一段拚。

這裡有兩個原本指南很重要、現在也仍然成立的觀念：

- 五階段不是五個互不相干的模型，而是同一個 policy 連續微調
- 目的不是把所有技能一次混著硬學，而是降低 forward / lateral / diagonal / yaw 彼此干擾

你可以把它理解成：

1. Stage1 先把直走打穩
2. Stage2 再把側移獨立拉起來
3. Stage3 再學 `vx + vy` 的耦合控制
4. Stage4 單獨練 yaw，避免旋轉技能干擾其他技能
5. Stage5 最後再整合前四段

五個 stage 的意義：

| Stage | 重點 |
|---|---|
| `env.stage=1` | forward-only，先把直走穩定性打底 |
| `env.stage=2` | lateral-only，讓側移獨立成型 |
| `env.stage=3` | diagonal-only，讓 `vx + vy` 組合動作成型 |
| `env.stage=4` | yaw-only，先把原地旋轉單獨練穩 |
| `env.stage=5` | mixed，整合前四段 |

補充一個很重要的訓練邏輯：

- Stage5 不是暴力覆蓋前四段
- 目前環境裡有前進防遺忘機制，會盡量保護 Stage1 已建立的穩定直走能力，再慢慢把 lateral / diagonal / yaw 拉上來

### 一鍵跑完整 pipeline

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --num_envs 4096 \
  --s1 8000 \
  --s2 8000 \
  --s3 9000 \
  --s4 10000 \
  --s5 12000
```

### 先做 GUI 預檢再放整晚

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --precheck_gui 1 \
  --precheck_stage 1 \
  --precheck_envs 64 \
  --precheck_iters 120 \
  --num_envs 4096 \
  --s1 8000 \
  --s2 8000 \
  --s3 9000 \
  --s4 10000 \
  --s5 12000
```

### pipeline 會幫你做什麼

- 每個 stage 自動接前一段 checkpoint
- 預設使用 full resume，也就是 policy、optimizer、iteration continuity 都會接續
- 自動做 stage health gate
- 寫 pipeline log 到 `logs/rsl_rl/pipeline/<run_tag>.log`
- 最後會印出 `FINAL_CKPT=...`

### 中途斷掉後從某 stage 接著跑

注意：如果你要從 stage 3 或 stage 4 接著跑，`--run_tag` 必須和前一次一致，因為 pipeline 會用它去找前一段 run。

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --start_stage 3 \
  --num_envs 4096 \
  --s1 8000 \
  --s2 8000 \
  --s3 9000 \
  --s4 10000 \
  --s5 12000
```

### 若你真的要用 policy-only handoff

這通常只適合做實驗，不建議當正式 curriculum。

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_policy_only \
  --resume_policy_only 1 \
  --reset_action_std 0.8
```

它的效果是：

- 只載入 policy 權重
- optimizer state 不接續
- iteration 會重置，所以 `model_*.pt` 編號會重新開始

## 5.4 手動指定 stage 的一般訓練

如果你不要 pipeline，也可以直接在 `train.py` 後面加 Hydra override：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 8000 \
  --run_name stage1_manual \
  env.stage=1 \
  env.draw_debug_vis=False
```

如果你要手動接上一段：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 8000 \
  --run_name stage2_manual \
  --resume \
  --load_run <上一段 run 名稱> \
  --checkpoint model_xxxxx.pt \
  env.stage=2
```

如果你是在做手動逐段 debug，原本文件裡常用的兩個額外 override 仍然有用：

```bash
env.draw_debug_vis=False \
env.dr_try_physical_material_randomization=False
```

用途是：

- `env.draw_debug_vis=False`：減少不必要視覺除錯負擔
- `env.dr_try_physical_material_randomization=False`：在某些手動除錯情境下，先把物理材質隨機化關掉，讓問題更容易重現與定位

---

## 6. Teacher 與 Distillation 操作

## 6.1 Teacher PPO

teacher 版本是為了訓練上界與之後做蒸餾。它的 actor/critic 都吃 `teacher` privileged obs。

重要提醒：

- teacher 很適合在模擬裡追求更高學習效率或當蒸餾來源
- teacher 不等於真機可直接部署策略
- 真機或 sim2real 部署仍應優先使用一般 PPO student 或 distillation student

### 主 task teacher

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name wheg_privileged_teacher_v1
```

### ForwardFast teacher

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-ForwardFast-Direct-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 1500 \
  --run_name forward_fast_privileged_teacher_v1
```

## 6.2 Distillation 先做 smoke test

現在專案已經把 distillation runner 接好了，但蒸餾流程比一般 PPO 更容易因 checkpoint 路徑規則而卡住，所以推薦先做 smoke test。

### 一次檢查 reform stack + teacher + distillation

```bash
python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 32 \
  --runner_smoke \
  --distill_smoke \
  --runner_steps 8 \
  --distill_steps 8 \
  --log_dir /tmp/redrhex_reform_smoke
```

這會做三件事：

- random rollout smoke
- 1 iteration PPO smoke
- teacher PPO + distillation smoke

輸出檔常見位置：

- `/tmp/redrhex_validate_stats.json`
- `/tmp/redrhex_reform_smoke/teacher/teacher_smoke.pt`
- `/tmp/redrhex_reform_smoke/distill/distill_smoke.pt`

### 已有 teacher checkpoint，單獨檢查 distillation

```bash
python scripts/rsl_rl/validate_distillation_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 8 \
  --teacher_ckpt /absolute/path/to/model_xxxxx.pt \
  --log_dir /tmp/redrhex_reform_distill
```

## 6.3 正式做 distillation

### Step 1：先訓練 teacher

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name wheg_privileged_teacher_v1
```

### Step 2：確認 teacher checkpoint

```bash
TEACHER_RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg_teacher/* | head -1)")
TEACHER_CKPT=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg_teacher/$TEACHER_RUN/model_*.pt | tail -1)")
echo "TEACHER_RUN=$TEACHER_RUN"
echo "TEACHER_CKPT=$TEACHER_CKPT"
```

### Step 3：讓 distillation runner 找得到 teacher run

這一步很重要。現在 `train.py` 在 distillation 模式下，會用 distill config 的 `experiment_name` 當 resume 根目錄：

- 主 task distill：`logs/rsl_rl/redrhex_wheg_distill/`
- ForwardFast distill：`logs/rsl_rl/redrhex_forward_fast_distill/`

也就是說，你不能假設 teacher run 放在 `redrhex_wheg_teacher/` 時，`--agent rsl_rl_distillation_cfg_entry_point --resume --load_run <teacher_run>` 一定會直接找到。

最簡單的做法是建立軟連結，讓 teacher run 同時出現在 distill experiment root 下面：

```bash
mkdir -p logs/rsl_rl/redrhex_wheg_distill
ln -s ../redrhex_wheg_teacher/$TEACHER_RUN logs/rsl_rl/redrhex_wheg_distill/$TEACHER_RUN
```

如果是 ForwardFast distill，對應改成：

```bash
mkdir -p logs/rsl_rl/redrhex_forward_fast_distill
ln -s ../redrhex_forward_fast_teacher/$TEACHER_RUN logs/rsl_rl/redrhex_forward_fast_distill/$TEACHER_RUN
```

### Step 4：開始 distillation

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_distillation_cfg_entry_point \
  --resume \
  --load_run "$TEACHER_RUN" \
  --checkpoint "$TEACHER_CKPT" \
  --headless \
  --num_envs 2048 \
  --max_iterations 800 \
  --run_name wheg_student_distill_v1
```

ForwardFast 版本只要把 task 換成 `Template-Redrhex-ForwardFast-Direct-v0` 即可。

### Step 5：播放 distillation student

```bash
STUDENT_RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg_distill/* | head -1)")
STUDENT_CKPT=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg_distill/$STUDENT_RUN/model_*.pt | tail -1)")

python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_distillation_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --load_run "$STUDENT_RUN" \
  --checkpoint "$STUDENT_CKPT"
```

`play.py` 已經有處理 distillation runner，會自動用 `student_obs_normalizer` 匯出 student policy。

---

## 7. TensorBoard 怎麼看

最簡單的開法：

```bash
tensorboard --logdir logs/rsl_rl --port 6006 --bind_all
```

常先看這幾類：

- `Train/mean_reward`
- `Train/mean_episode_length`
- `Episode_Termination/terminated`
- `Episode_Reward/rew_fall`
- 各種 `diag_*` 診斷指標

對主 task 而言，現在常見的診斷重點是：

- episode length 有沒有穩定上升
- terminated 有沒有下降
- base height / tilt 相關診斷有沒有長期惡化
- forward / lateral / diagonal / yaw 是否都真的有學起來，而不是只剩一種技能

如果你跑的是五階段 curriculum，原本指南裡每個 stage 的觀察重點也值得保留：

### 五階段各 stage 建議觀察

全階段共通：

- `Train/mean_episode_length`：要上升且穩定
- `Train/mean_reward`：整體往上
- `Episode_Termination/terminated`：下降
- `Episode_Reward/rew_fall`：接近 0
- `Episode_Reward/diag_base_height`：不要長期崩掉

Stage1：

- `diag_cmd_vx` 應大於 0，`diag_cmd_vy/wz` 應接近 0
- `rew_tracking`、`rew_forward_gait` 應上升
- `diag_forward_duty_ema` 建議往穩定 tripod 節奏收斂

Stage2：

- `diag_lateral_fsm_state` 最好能進到 `2`，也就是 `LATERAL_STEP`
- `rew_lateral_speed_deficit` 往 0 靠近
- `diag_lateral_vel` 絕對值上升且方向正確

Stage3：

- `rew_diag_sign` 轉正且穩定
- `diag_vel_error` 下降

Stage4：

- `rew_yaw_track` 上升
- `diag_wz_error` 下降
- `diag_roll_rms`、`diag_pitch_rms` 不要爆掉

Stage5：

- forward / lateral / diagonal / yaw 四類指標都要有反應
- 不能只剩其中一種技能表現好

---

## 8. Play、匯出、鍵盤控制

## 8.1 播放最新 checkpoint

### 主 task PPO

```bash
RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/* | head -1)")
CKPT=$(ls -v logs/rsl_rl/redrhex_wheg/$RUN/model_*.pt | tail -1)

python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --load_run "$RUN" \
  --checkpoint "$CKPT"
```

### ForwardFast PPO

```bash
RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_forward_fast/* | head -1)")
CKPT=$(ls -v logs/rsl_rl/redrhex_forward_fast/$RUN/model_*.pt | tail -1)

python scripts/rsl_rl/play.py \
  --task Template-Redrhex-ForwardFast-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --load_run "$RUN" \
  --checkpoint "$CKPT"
```

你也可以直接傳絕對路徑：

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --checkpoint /absolute/path/to/model_xxxxx.pt
```

如果你是從五階段 pipeline 完成後要抓最終模型，原本手冊這段仍然很好用：

```bash
PIPE_LOG=$(ls -t logs/rsl_rl/pipeline/*.log | head -1)
FINAL_CKPT=$(grep -F "[DONE] FINAL_CKPT=" "$PIPE_LOG" | tail -1 | sed 's/.*FINAL_CKPT=//')
echo "PIPE_LOG=$PIPE_LOG"
echo "FINAL_CKPT=$FINAL_CKPT"
basename "$FINAL_CKPT"
```

如果 `basename "$FINAL_CKPT"` 不是 `model_xxxxx.pt`，就不要直接拿去 play。

## 8.2 Play 的實用行為

- 若 `--checkpoint` 是資料夾，會自動抓其中最新 `model_*.pt`
- 若 `--checkpoint` 指到 `events.out.tfevents...` 或 `exported/policy.pt`，會自動嘗試 fallback 到相鄰的訓練 checkpoint
- 若 run 名稱像 `..._stage4`，`play.py` 會自動把 `env.stage` 推成 4
- 若你不想讓它自動推 stage，可以加 `--disable_auto_stage_from_checkpoint`
- 每次 `play.py` 成功載入模型後，都會自動匯出：
  - `exported/policy.pt`
  - `exported/policy.onnx`

## 8.3 只匯出，不進入 play 迴圈

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 1 \
  --disable_keyboard_control \
  --checkpoint /absolute/path/to/model_xxxxx.pt \
  --export_policy
```

## 8.4 關閉鍵盤控制

如果你想看模型在環境命令採樣下自然表現，不要手動切指令：

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 64 \
  --disable_keyboard_control \
  --checkpoint /absolute/path/to/model_xxxxx.pt
```

## 8.5 開啟鍵盤控制時可用的初始命令

`--initial_command` 可選：

- `forward`
- `backward`
- `left`
- `right`
- `diag_left`
- `diag_right`
- `yaw_ccw`
- `yaw_cw`
- `stop`

初次使用建議：

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --checkpoint /absolute/path/to/model_xxxxx.pt
```

## 8.6 鍵盤對應

鍵盤控制是切換模式，不是按住才有作用。按一次就會維持該命令，直到你按下一個按鍵。

| 按鍵 | 命令 |
|---|---|
| `W` | 前進 |
| `S` | 後退 |
| `A` | 左移 |
| `D` | 右移 |
| `T` | 左前 |
| `R` | 右前 |
| `F` | 左後 |
| `G` | 右後 |
| `Q` | 逆時針旋轉 |
| `E` | 順時針旋轉 |
| `Space` | 停止 |

注意：

- Isaac Sim 視窗必須是焦點視窗
- 如果你發現按鍵沒反應，先點一下 Isaac Sim 視窗再試

## 8.7 Headless 錄影

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --video \
  --video_length 1000 \
  --num_envs 64 \
  --disable_keyboard_control \
  --checkpoint /absolute/path/to/model_xxxxx.pt
```

---

## 9. 評估與驗收

## 9.1 `validate_reform_stack.py`

這是改革後最基礎的健康檢查。

### 只做 random rollout smoke

```bash
python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 32
```

### 加做 1 iteration PPO smoke

```bash
python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 32 \
  --runner_smoke \
  --runner_steps 8
```

### 加做 teacher + distillation smoke

```bash
python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 32 \
  --runner_smoke \
  --distill_smoke \
  --runner_steps 8 \
  --distill_steps 8 \
  --log_dir /tmp/redrhex_reform_smoke
```

## 9.2 `eval_command_sweep.py`

這個腳本適合做量化驗收，不是只靠肉眼看 play。

### 主 task 最終 mixed 驗收

```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 256 \
  --checkpoint /absolute/path/to/model_xxxxx.pt \
  --eval_profile stage5
```

如果你要沿用原本那套比較嚴格的 Stage5 驗收門檻，可以直接用這個版本：

```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 256 \
  --checkpoint /absolute/path/to/model_xxxxx.pt \
  --eval_profile stage5 \
  --warmup_steps 120 \
  --sweep_steps 600 \
  --accept_duration_s 2.0 \
  --accept_vx_abs 0.15 \
  --accept_vy_abs 0.15 \
  --accept_wz_abs 0.40 \
  --accept_lin_ratio 0.55 \
  --accept_wz_ratio 0.55 \
  --accept_yaw_tilt_bound 0.60 \
  --accept_yaw_tilt_ratio 0.70 \
  --accept_diag_sign_ratio 0.70 \
  --accept_diag_component_ratio 0.50 \
  --accept_max_fall_rate 0.20 \
  --accept_skill_pass_ratio 0.60 \
  --accept_overall_pass_ratio 0.70
```

### 輸出 CSV

```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 256 \
  --checkpoint /absolute/path/to/model_xxxxx.pt \
  --eval_profile stage5 \
  --csv logs/rsl_rl/redrhex_wheg/eval_command_sweep.csv
```

`--eval_profile` 常用值：

- `stage1`
- `stage2`
- `stage3`
- `stage4`
- `stage5`
- `full`

---

## 10. 恢復訓練與常用參數

## 10.1 一般 resume

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --resume \
  --load_run <run 名稱> \
  --checkpoint model_xxxxx.pt
```

## 10.2 policy-only resume

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --resume \
  --resume_policy_only \
  --reset_action_std 0.8 \
  --load_run <run 名稱> \
  --checkpoint model_xxxxx.pt
```

## 10.3 常用 train 參數

| 參數 | 用途 |
|---|---|
| `--task` | 選 task |
| `--agent` | 選 PPO / teacher / distill 入口 |
| `--num_envs` | 環境數量 |
| `--max_iterations` | 訓練迭代數 |
| `--resume` | 從既有 checkpoint 繼續 |
| `--load_run` | 要從哪個 run 繼續 |
| `--checkpoint` | 指定 `model_*.pt` |
| `--resume_policy_only` | 只載 policy，不載 optimizer |
| `--reset_action_std` | 搭配 policy-only 時重新設 action std |
| `env.stage=<1..5>` | 指定主 task 的 curriculum stage |

---

## 11. 這次改革後，操作上有什麼實際變化

這一段專門講「你現在操作這個專案時，和舊版相比，哪些地方不同」。

### 11.1 現在不是只有單一 observation

環境現在會同時提供：

- `policy`
- `history`
- `critic`
- `teacher`

這帶來的操作差異是：

- 一般 PPO 直接用 `rsl_rl_cfg_entry_point`
- teacher PPO 要改成 `rsl_rl_teacher_cfg_entry_point`
- distillation 要改成 `rsl_rl_distillation_cfg_entry_point`

### 11.2 現在有 asymmetric critic

所以你不需要另外手改 train 腳本來分 actor/critic input，配置已經在 agent cfg 中接好。

### 11.3 現在有 symmetry augmentation

這已經在 PPO cfg 中啟用，不需要你額外傳旗標。

### 11.4 現在有 rough terrain、actuator randomization、fault injection

這些已經在主 task 環境裡。代表：

- 主 task 的訓練難度比舊版平地 baseline 更高
- 但理論上更接近真機部署需求
- 如果你只是想先快速看到「會往前走」，請先用 ForwardFast

### 11.5 `play.py` 現在會自動匯出 policy

所以現在一般流程可以簡化成：

1. train
2. play 一次
3. 直接去該 run 的 `exported/` 取 `policy.onnx` 或 `policy.pt`

不需要再另外寫一支 export script。

---

## 12. 最容易踩的坑

## 12.1 把 TensorBoard 檔案當 checkpoint

錯誤示範：

- `events.out.tfevents...`
- `params/agent.yaml`
- `exported/policy.pt`

正確做法：

- 傳 `model_*.pt`

## 12.2 `--checkpoint` 只給檔名，但沒給 `--load_run`

如果你只傳：

```bash
--checkpoint model_2500.pt
```

那通常還要搭配：

```bash
--load_run <run 名稱>
```

否則就直接傳完整絕對路徑。

## 12.3 pipeline 續跑時換了 `--run_tag`

如果你要 `--start_stage 3` 或 `--start_stage 4`，`--run_tag` 必須和前一次一致，否則 pipeline 找不到前一段 run。

## 12.4 `env.stage=1` 寫法錯了

Hydra 覆寫一定要寫：

```bash
env.stage=1
```

不是：

```bash
stage=1
```

## 12.5 鍵盤沒反應

先確認：

- 你沒有加 `--disable_keyboard_control`
- Isaac Sim 視窗有被點到
- 你不是在 headless 模式

## 12.6 `omni.platforminfo` 或 CPU 告警很多

這類訊息很多時候只是平台偵測或環境告警，不一定是主因。先優先檢查：

- 是否一出生就 `terminated`
- `Episode_Reward/diag_base_height` 是否崩掉
- 是否是 checkpoint 路徑傳錯

## 12.7 teacher 和 student 混用

記住這個原則：

- 要一般部署：優先用 `rsl_rl_cfg_entry_point`
- 要做蒸餾上界：先訓 `rsl_rl_teacher_cfg_entry_point`
- 要可部署的蒸餾學生：用 `rsl_rl_distillation_cfg_entry_point`

## 12.8 distillation 找不到 teacher checkpoint

先檢查三件事：

1. `TEACHER_RUN` 是否真的存在
2. `TEACHER_CKPT` 是否真的是 `model_*.pt`
3. 該 run 是否對 distill experiment root 可見

最穩的做法通常是先建立軟連結，再啟動 distillation。

## 12.9 系統沒有 `rg`

原本這曾經是 pipeline 會中斷的原因之一。現在 `train_stage_pipeline.sh` 已經有 `grep` fallback，所以不會因為少了 `rg` 就直接掛掉，但若你自己寫外部腳本，還是建議優先確認 `rg` 是否可用。

---

## 13. 建議工作流

### 情境 A：第一次接觸專案

```text
validate_reform_stack.py
-> ForwardFast train
-> TensorBoard
-> play.py
-> exported/policy.onnx
```

### 情境 B：要做完整 locomotion 能力

```text
GUI precheck
-> 5-stage pipeline
-> eval_command_sweep.py
-> play.py
-> exported/policy.onnx
```

### 情境 C：要研究 teacher-student

```text
teacher PPO
-> validate_distillation_stack.py
-> distillation train
-> play distill student
-> exported/policy.onnx
```

---

## 14. 最後的實務建議

- 先確定 `validate_reform_stack.py` 能過，再開長訓練。
- 要快速直走，就先用 ForwardFast，不要一開始就硬打完整 mixed task。
- 要正式完整能力，優先跑五階段 pipeline，不要只靠單段 mixed 硬練。
- 要做 sim2real 部署，優先使用一般 PPO student 或 distillation student。
- 每次找到可用模型後，記得去對應 run 目錄確認 `exported/policy.onnx` 是否已經生成。
- 如果你看到行為很好但數值驗收不清楚，補跑 `eval_command_sweep.py`，不要只靠目視判斷。
- 一個很實用的節奏是：先 GUI precheck 2 到 5 分鐘，再做短迭代快篩，確認不是出生即死後，再放長訓練或整晚 pipeline。

這份文件未來若再有 train / play / export / validation 流程變更，應優先更新這裡。
