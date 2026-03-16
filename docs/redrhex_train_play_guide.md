# RedRhex 五階段訓練整合指南（Train + Explainer + Eval + Play）

> 這份文件已整併原本兩份：
> - `docs/redrhex_train_play_guide.md`
> - `docs/redrhex_stage_training_explainer.md`

## 模式總覽（先選一種）

| 模式 | 適用情境 | 核心目的 | 入口章節 |
|---|---|---|---|
| 快速訓練（ForwardFast） | 實驗室現場反覆調參、要快 | 盡快得到可上機直走 policy | A 區 |
| 一般訓練（五階段） | 要完整技能（前進/側移/斜向/旋轉） | 最終整合能力與泛化 | B 區 |

閱讀建議：
- 只要快速上機：先看 `A`，最後看 `C`（FAQ）
- 要完整能力訓練：看 `B`，最後看 `C`（FAQ）

---

## 導言：一般訓練為什麼從單段改成五階段

你原本是一次混合訓練所有技能（forward/lateral/diagonal/yaw）。
現在改成 **5-stage curriculum**，目的是降低技能互相干擾：

1. Stage1: Forward-only（先把直走練穩）
2. Stage2: Lateral-only（把側移獨立練出來）
3. Stage3: Diagonal-only（融合 vx+vy）
4. Stage4: Yaw-only（先把原地旋轉單獨練穩）
5. Stage5: Mixed（整合前四段，持續潤化）

這 5 段不是 5 個模型，而是 **同一個 policy 連續微調**：每段都接續上一段 checkpoint。

---

## 共用 0) 先做一致性檢查（兩種模式都建議）

先確認你目前腳本可正常解析（避免半夜才發現指令不相容）：

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab

bash -n scripts/rsl_rl/train_stage_pipeline.sh
python -m py_compile \
  scripts/rsl_rl/play.py \
  scripts/rsl_rl/eval_command_sweep.py \
  source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py
```

---

## A) 快速訓練流程（ForwardFast，現場 Sim2Real）

## A1) 最快直接執行（Train / TensorBoard / Play）

以下是現場最常用的最小流程，直接複製即可。

### Step A: 開始訓練（ForwardFast）

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab

python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-ForwardFast-Direct-v0 \
  --headless \
  --num_envs=2048 \
  --max_iterations=1500 \
  --run_name=forward_fast_trial_a
```

### Step B: 開 TensorBoard 監看訓練曲線

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab

tensorboard --logdir . --port 6006 --bind_all
```

開瀏覽器：
- 本機：`http://localhost:6006`
- 遠端機器：`http://<你的主機IP>:6006`

建議先看這幾條：
- `Train/mean_reward`
- `Train/mean_episode_length`
- `Episode_Reward/rew_fall`
- `Episode_Reward/rew_tracking`

### Step C: 找最新 checkpoint 並播放（Play）

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab

RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_forward_fast/* | head -1)")
CKPT=$(ls -v logs/rsl_rl/redrhex_forward_fast/$RUN/model_*.pt | tail -1)
echo "RUN=$RUN"
echo "CKPT=$CKPT"

python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-ForwardFast-Direct-v0 \
  --num_envs=64 \
  --disable_keyboard_control \
  --load_run="$RUN" \
  --checkpoint="$CKPT"
```

---

## A2) 快速訓練設定說明（現場 Sim2Real：直走專用）

已新增一個 forward-only 快速收斂 task：
- Task ID: `Template-Redrhex-ForwardFast-Direct-v0`
- 特色：
  - 只訓練前進（`stage=1` 固定）
  - reward 權重集中在 forward progress + velocity tracking
  - 弱化 lateral/diagonal/yaw shaping，減少干擾
  - domain randomization 改為窄範圍（保留基本 sim2real 魯棒）
  - 關閉隨機推擠，讓 TensorBoard 更快進入平穩段

快速開訓（建議先用 2048 env 進行現場迭代）：

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-ForwardFast-Direct-v0 \
  --headless \
  --num_envs=2048 \
  --max_iterations=1500 \
  --run_name=forward_fast_trial_a
```

如果 GPU 足夠，再拉到 4096：

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-ForwardFast-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=1500 \
  --run_name=forward_fast_trial_b
```

訓練後直接匯出 ONNX（play.py 會自動輸出 `exported/policy.onnx`）：

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-ForwardFast-Direct-v0 \
  --headless \
  --disable_keyboard_control \
  --load_run=<你的 run 資料夾> \
  --checkpoint=<model_xxxxx.pt>
```

---

## A3) 2026-03-16 第一輪品質修正（歷程）：解決「六腳同轉、趴地死亡循環」

你回報的症狀是：
- TensorBoard 看起來有收斂，但上機後六隻腳常一起轉動
- 機身很快觸地，被判定死亡
- 形成「倒地 -> reset -> 再倒地」循環

這類問題的本質通常是「學到可拿分但不可用的策略」。  
這次修正不是只加重某一個 reward，而是同時調整「控制邏輯 + reward gate + termination + PPO」。

### A3.1) 控制器核心修正（最關鍵）

檔案：`source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py`

1. Forward bias 的著地/擺動判斷改成 **duty-cycle 時間制**：
- 由 `desired_cycle < duty_target` 判斷（`duty_target` 由 `forward_duty_target` 讀取）
- 避免固定角度窗造成六腳同速旋轉，提升 tripod 交替穩定性

2. 新增「健康姿態 gating」：
- 當機身過低或傾斜過大時，會關閉主要正向獎勵
- 避免策略靠趴地抖動拿到 tracking/progress 分數

### A3.2) ForwardFast 參數修正（環境層）

檔案：`source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`  
類別：`RedrhexForwardFastEnvCfg`

重點更新：
- 命令帶寬再縮窄：`lin_vel_x_range=[0.20, 0.32]`
- 動作更保守：`main_drive_vel_scale=5.8`、`stage_forward_policy_drive_residual_scale=[0.03]`
- 起步更平順：`stage_action_warmup_steps=[120]`
- 倒地判定更嚴格：`max_tilt_magnitude=0.75`、`fall_height_threshold=0.11`
- 獎勵防作弊開關：`gate_positive_rewards_when_unhealthy=True`
- 站姿目標提高：`target_base_height=0.15`
- 移除「只會轉腿也有分」：`leg_moving=0.0`

### A3.3) PPO 穩定化修正（演算法層）

檔案：`source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_ppo_cfg.py`  
類別：`PPORunnerForwardFastCfg`

重點更新：
- `num_steps_per_env=32`
- `init_noise_std=0.30`
- `learning_rate=4.0e-4`
- `entropy_coef=0.0010`
- `desired_kl=0.008`

目的是降低早期高噪聲亂探索，讓策略更快進入「可走、可穩」區間。

### A3.4) 這輪修正後要看哪幾條 TensorBoard

優先順序：
1. `Train/mean_episode_length`：應該上升，且不是只靠 timeout 漂亮
2. `Episode_Termination/terminated`：要持續下降
3. `Episode_Reward/rew_fall`：絕對值要往 0 靠近
4. `Episode_Reward/diag_base_height`：不能長期下滑
5. `Episode_Reward/rew_forward_gait` + `diag_forward_duty_ema`：要維持 tripod 節奏

如果 `mean_reward` 上升，但 `terminated` 不降、`rew_fall` 很差，代表是「假收斂」，不能上機。

> 註：A3 是第一輪修正歷程。若你要跑目前最新版，請看下面 A4。

---

## A4) 2026-03-16 第二輪修正（目前推薦）：回到 Stage1 成功邏輯再加速

你最新回報是：
- 比第一輪更不容易動
- 起步後仍會六腳同轉並趴地死亡

這代表第一輪修正「太保守」，把策略活動度壓掉了。  
因此第二輪策略是：**以原本五階段 Stage1 成功設定為底，只保留必要的加速訓練改動**。

### A4.1) 核心回調（與原 Stage1 對齊）

1. 控制器 stance 判定回到原本穩定邏輯：
- `desired_in_stance = self._in_stance_phase(desired_phase)`
- 不再使用 duty-time stance 控制（保留 duty/tripod reward 診斷）

2. ForwardFast 參數改為 Stage1 風格：
- `curriculum_stage_scales=[0.05]`（沿用 stage1 低隨機擾動）
- `lin_vel_x_range=[0.22, 0.42]`
- `command_resample_on_timer=False`
- `main_drive_vel_scale=8.0`
- `stage_forward_policy_drive_residual_scale=[0.10]`
- `stage_action_warmup_steps=[30]`（從 120 回到 30）

3. 終止條件放回 Stage1 穩定區：
- `stage1_min_base_height=0.03`
- `stage1_body_contact_height_threshold=0.06`
- `stage_body_contact_tilt_threshold=[1.80]`
- `stage_termination_grace_steps=[120]`
- `gate_positive_rewards_when_unhealthy=False`

4. PPO 恢復探索能力：
- `num_steps_per_env=24`
- `init_noise_std=0.55`
- `entropy_coef=0.0035`
- `learning_rate=5.0e-4`
- `desired_kl=0.01`

### A4.2) 這樣改的原因

第一輪的「低殘差 + 長暖機 + 嚴終止 + 低探索」組合，雖然能壓住暴衝，但也容易讓 policy 學成「不太動」。  
第二輪改法是回到你已驗證過的 Stage1 穩定策略空間，讓「會走」先成立，再用 forward-only 任務縮短總訓練時間。

### A4.3) 目前推薦重訓指令

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-ForwardFast-Direct-v0 \
  --headless \
  --num_envs=2048 \
  --max_iterations=1500 \
  --run_name=forward_fast_stage1ref_v1
```

### A4.4) 第二輪修正後先看這些 TensorBoard

1. `Train/mean_episode_length`：不能再卡在極短回合  
2. `Episode_Termination/terminated`：應下降  
3. `Episode_Reward/rew_forward_gait`：應持續上升  
4. `Episode_Reward/diag_base_height`：要穩，不要長期下滑  
5. `Episode_Reward/diag_cmd_vx` vs `diag_forward_vel`：追蹤要更貼近

---

## B) 一般訓練流程（五階段完整能力）

## B1) 每個 Stage 在做什麼

## B1-Stage1（Forward-only）
- 命令分布：`vx > 0, vy = 0, wz = 0`
- 硬限制：ABAD 鎖住
- 主要目標：穩定前進 + tripod 節奏 + 不倒地

## B1-Stage2（Lateral-only）
- 命令分布：`vy != 0, vx ~= 0, wz ~= 0`
- 硬限制：main-drive lock/soft-lock
- 流程：`GO_TO_STAND -> LATERAL_STEP`
- 主要目標：側移速度要起來，不是站著抖動

## B1-Stage3（Diagonal-only）
- 命令分布：`vx > 0 且 vy != 0, wz ~= 0`
- 控制：main-drive + ABAD 都開放
- 主要目標：vx/vy 方向同時正確（不能只會一邊斜走）

## B1-Stage4（Yaw-only）
- 命令分布：`wz != 0, vx ~= 0, vy ~= 0`
- 控制：允許 per-leg signed main-drive（可反轉）
- 主要目標：原地旋轉穩定，避免「掀機身作弊」

## B1-Stage5（Mixed）
- 命令分布：FWD/LAT/DIAG/YAW 混合
- 主要目標：整合技能且保留 Stage1 的穩定直走能力

---

## B2) 現在程式如何保留「穩定直走」不被後續洗掉

目前環境已加入前進防遺忘機制：
- Forward 模式殘差上限（cap）避免 policy 大幅破壞前進 bias
- stage-specific main-drive residual scale（各 stage 強度不同）
- Stage5 仍保留 forward 專屬 reward multiplier

你可以把 Stage5 理解成：
- 不是暴力覆蓋前四段
- 是在保護 forward 基底下，把 lateral/diag/yaw 慢慢拉上來

---

## B3) 訓練前建議（先肉眼確認）

## B3.1) 前置

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

## B3.2) 建議先做 GUI 預檢（非 headless）

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag precheck_a \
  --precheck_gui 1 \
  --precheck_stage 1 \
  --precheck_envs 64 \
  --precheck_iters 120 \
  --num_envs 512 \
  --s1 8000 --s2 8000 --s3 9000 --s4 10000 --s5 12000
```

確認「不是一出生就觸地死亡」後，再放整晚 headless。

---

## B4) 一鍵跑完五階段（推薦）

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --num_envs 4096 \
  --s1 8000 --s2 8000 --s3 9000 --s4 10000 --s5 12000
```

若你想先看 2~5 分鐘 GUI 再放整晚：

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --precheck_gui 1 \
  --precheck_stage 1 \
  --precheck_envs 64 \
  --precheck_iters 120 \
  --num_envs 4096 \
  --s1 8000 --s2 8000 --s3 9000 --s4 10000 --s5 12000
```

補充：
- pipeline 會自動串接 stage checkpoint
- 預設使用「完整 resume」（policy + optimizer + iteration），是**真正 curriculum 接續學習**
- 因此 stage2~stage5 的 `model_*.pt` 數字會延續成長，不會每段從 0 重來
- 現在即使系統沒有 `rg`，也會 fallback `grep`，不會因缺 `rg` 直接中斷
- 會輸出：`FINAL_CKPT=.../model_xxxxx.pt`

快速檢查「是否真的接續」：

```bash
RUN1=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage1* | head -1)")
RUN2=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage2* | head -1)")
S1=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN1/model_*.pt | tail -1)")
S2=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN2/model_*.pt | tail -1)")
echo "S1=$S1"
echo "S2=$S2"
# 正常 full-resume：S2 的數字應 > S1
```

## B4.1) 若中斷後續跑

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --start_stage 2 \
  --num_envs 4096 \
  --s1 8000 --s2 8000 --s3 9000 --s4 10000 --s5 12000
```

---

## B5) 手動逐段跑（可精細控制）

## B5-Stage1（手動）

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=8000 \
  --run_name=stage1 \
  env.stage=1 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

```bash
RUN1=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage1* | head -1)")
CKPT1=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN1/model_*.pt | tail -1)")
echo "RUN1=$RUN1"
echo "CKPT1=$CKPT1"
```

## B5-Stage2（接 Stage1）

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=8000 \
  --run_name=stage2 \
  --resume \
  --load_run="$RUN1" \
  --checkpoint="$CKPT1" \
  env.stage=2 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

## B5-Stage3（接 Stage2）

```bash
RUN2=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage2* | head -1)")
CKPT2=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN2/model_*.pt | tail -1)")

python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=9000 \
  --run_name=stage3 \
  --resume \
  --load_run="$RUN2" \
  --checkpoint="$CKPT2" \
  env.stage=3 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

## B5-Stage4（接 Stage3）

```bash
RUN3=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage3* | head -1)")
CKPT3=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN3/model_*.pt | tail -1)")

python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=10000 \
  --run_name=stage4 \
  --resume \
  --load_run="$RUN3" \
  --checkpoint="$CKPT3" \
  env.stage=4 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

## B5-Stage5（接 Stage4）

```bash
RUN4=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage4* | head -1)")
CKPT4=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN4/model_*.pt | tail -1)")

python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=12000 \
  --run_name=stage5 \
  --resume \
  --load_run="$RUN4" \
  --checkpoint="$CKPT4" \
  env.stage=5 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

若你要刻意使用 policy-only（會讓每段迭代號重新從 0 開始，不建議做正式 curriculum）：

```bash
python scripts/rsl_rl/train.py ... --resume --resume_policy_only --reset_action_std=0.8 ...
```

```bash
RUN5=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage5* | head -1)")
FINAL_CKPT=$(ls -v logs/rsl_rl/redrhex_wheg/$RUN5/model_*.pt | tail -1)
echo "FINAL_CKPT=$FINAL_CKPT"
```

---

## B6) TensorBoard 每階段要看什麼

## B6-全階段共通
- `Train/mean_episode_length`：上升且穩定
- `Train/mean_reward`：整體往上
- `Episode_Termination/terminated`：下降
- `Episode_Reward/rew_fall`：接近 0（少摔）
- `Episode_Reward/diag_base_height`：不要長期崩掉

## B6-Stage1
- `diag_cmd_vx` > 0，`diag_cmd_vy/wz` ~ 0
- `rew_tracking`、`rew_forward_gait` 上升
- `diag_forward_duty_ema` 靠近 0.65

## B6-Stage2
- `diag_lateral_fsm_state` 能進到 2（LATERAL_STEP）
- `rew_lateral_speed_deficit` 往 0 靠近（少負）
- `diag_lateral_vel` 絕對值上升且方向正確

## B6-Stage3
- `rew_diag_sign` 轉正且穩定
- `diag_vel_error` 下降

## B6-Stage4
- `rew_yaw_track` 上升
- `diag_wz_error` 下降
- `diag_roll_rms`、`diag_pitch_rms` 不爆

## B6-Stage5
- forward/lateral/diag/yaw 四類指標都要有反應
- 不能只剩一種技能表現好

---

## B7) 驗收（Command Sweep）

## B7.1) 最終混合驗收（Stage5）

```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=256 \
  --checkpoint="$FINAL_CKPT" \
  --eval_profile=stage5 \
  --warmup_steps=120 \
  --sweep_steps=600 \
  --accept_duration_s=2.0 \
  --accept_vx_abs=0.15 \
  --accept_vy_abs=0.15 \
  --accept_wz_abs=0.40 \
  --accept_lin_ratio=0.55 \
  --accept_wz_ratio=0.55 \
  --accept_yaw_tilt_bound=0.60 \
  --accept_yaw_tilt_ratio=0.70 \
  --accept_diag_sign_ratio=0.70 \
  --accept_diag_component_ratio=0.50 \
  --accept_max_fall_rate=0.20 \
  --accept_skill_pass_ratio=0.60 \
  --accept_overall_pass_ratio=0.70
```

可輸出 CSV：

```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=256 \
  --checkpoint="$FINAL_CKPT" \
  --eval_profile=stage5 \
  --warmup_steps=120 \
  --sweep_steps=600 \
  --accept_duration_s=2.0 \
  --accept_vx_abs=0.15 \
  --accept_vy_abs=0.15 \
  --accept_wz_abs=0.40 \
  --accept_lin_ratio=0.55 \
  --accept_wz_ratio=0.55 \
  --accept_yaw_tilt_bound=0.60 \
  --accept_yaw_tilt_ratio=0.70 \
  --accept_diag_sign_ratio=0.70 \
  --accept_diag_component_ratio=0.50 \
  --accept_max_fall_rate=0.20 \
  --accept_skill_pass_ratio=0.60 \
  --accept_overall_pass_ratio=0.70 \
  --csv logs/rsl_rl/redrhex_wheg/${RUN5}/eval_command_sweep.csv
```

---

## B8) Play（播放最終模型）

先確定抓到的是模型檔，不是 TensorBoard 事件檔：

```bash
RUN5=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage5* | head -1)")
FINAL_CKPT=$(ls -v "logs/rsl_rl/redrhex_wheg/$RUN5"/model_*.pt | tail -1)
echo "RUN5=$RUN5"
echo "FINAL_CKPT=$FINAL_CKPT"
basename "$FINAL_CKPT"   # 必須是 model_xxxxx.pt
```

建議先用 `stop` 起步，再用鍵盤切換方向（較安全）：

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-Direct-v0 \
  --num_envs=64 \
  --initial_command=stop \
  --checkpoint="$FINAL_CKPT"
```

Headless 錄影：

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --video \
  --video_length=1000 \
  --num_envs=64 \
  --initial_command=stop \
  --checkpoint="$FINAL_CKPT"
```

若你要先排除鍵盤控制干擾（直接看模型在訓練命令分佈下的行為）：

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-Direct-v0 \
  --num_envs=64 \
  --disable_keyboard_control \
  --checkpoint="$FINAL_CKPT"
```

註：
- `play.py` 會自動從 checkpoint 路徑推斷 `env.stage`（例如 `..._stage4/...` 會自動用 stage4 設定）
- 若你要手動固定 stage，可加 `--disable_auto_stage_from_checkpoint` 再用 Hydra 覆寫（例：`env.stage=5`）
- `play.py` 現在若你誤傳 `events.out.tfevents...`，會自動嘗試改抓同資料夾 `model_*.pt`

注意：`--checkpoint` 只能是一個完整路徑，不要把兩段路徑黏在一起。

檢查檔案是否正確：

```bash
test -f "$FINAL_CKPT" && echo "checkpoint exists"
basename "$FINAL_CKPT"   # 應該是 model_xxxxx.pt
```

如果你是從 pipeline log 取得最終模型，也請用這種方式：

```bash
PIPE_LOG=$(ls -t logs/rsl_rl/pipeline/*.log | head -1)
FINAL_CKPT=$(grep -F "[DONE] FINAL_CKPT=" "$PIPE_LOG" | tail -1 | sed 's/.*FINAL_CKPT=//')
echo "FINAL_CKPT=$FINAL_CKPT"
```

---

## C) 共用常見問題（FAQ）

## Q1: Stage1 訓練完就中斷（`rg: 指令找不到`）
- 原因：舊 pipeline 健康檢查解析依賴 `rg`
- 現況：已改為 `rg` 不存在時自動 fallback `grep`

## Q2: `stage=1` 覆寫失敗
- 要用 `env.stage=1`（不是 `stage=1`）

## Q3: `play.py` 找不到 checkpoint
- 請先 `echo "$FINAL_CKPT"` 確認只是一條路徑
- 再把同一條路徑傳給 `--checkpoint`
- 若看到 `FileNotFoundError: Unable to find the file: model_xxxx.pt`
  表示你只傳了檔名；請改傳完整路徑（例如 `logs/rsl_rl/.../model_1199.pt`）

## Q4: `omni.platforminfo` CPU 錯誤很多
- 多數情況是平台偵測告警，不是主因
- 優先看：是否一出生就 `terminated`、是否 base height 崩掉

## Q5: `play.py` 報錯 `UnpicklingError: invalid load key, 'H'`
- 原因：`--checkpoint` 指到非模型檔（通常是 `events.out.tfevents...`）
- 修正：
  1. 重新設定 `FINAL_CKPT` 為 `model_*.pt`
  2. `echo "$FINAL_CKPT"` 確認只是一條路徑
  3. `basename "$FINAL_CKPT"` 必須是 `model_xxxxx.pt`
  4. 現在 `play.py/eval_command_sweep.py` 已內建 fallback，但仍建議手動確認

---

## C1) 建議訓練節奏（避免再浪費整晚）

1. 先跑 GUI precheck（2~5 分鐘）
2. 再跑 Stage1 500~1000 iter 快篩，確認非出生即死
3. 通過後再放整晚全 pipeline
4. 隔天先跑 `eval_command_sweep.py` 再決定是否繼續加訓

---

## D) ForwardFast 改版完整說明（教授報告版）

本章是「給教授看的工程說明版」，內容與目前程式碼同步（更新日期：2026-03-16）。

## D1) 問題定義（Sim2Real 現場真實痛點）

實驗室流程不是一次訓練完就成功，而是：
1. 先在模擬訓練 policy。  
2. 匯出 ONNX 上機。  
3. 發現行為偏差後，回頭調 reward/物理參數。  
4. 再訓練、再上機，反覆迭代。  

原本最大瓶頸是「每輪重訓太久」，尤其在現場調參時效率非常差。

---

## D2) 這次做了哪些事情（總覽）

本次改版分成兩個階段：

1. **ForwardFast 路徑建立**  
- 新增 forward-only task 與專用 env/PPO 設定，目標是快速收斂。

2. **品質修正（解決假收斂）**  
- 針對你回報的「六腳同轉、趴地死亡循環」做控制器與 reward 機制修正，確保不只是曲線好看，而是真的能走。

---

## D3) 程式碼改動清單（檔案層級）

1. `source/RedRhex/RedRhex/tasks/direct/redrhex/__init__.py`  
- 新增 task：`Template-Redrhex-ForwardFast-Direct-v0`

2. `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`  
- 新增 `RedrhexForwardFastEnvCfg`（forward-only 快速訓練配置）
- 補上 play 相容參數（舊 checkpoint 在新版控制器下更穩定）

3. `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py`  
- 修正 forward bias 相位邏輯（duty-time scheduling）
- 新增健康姿態 reward gate（避免趴地拿分）

4. `source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_ppo_cfg.py`  
- 新增 `PPORunnerForwardFastCfg` 並做穩定化調整

5. `scripts/rsl_rl/play.py`  
- 新增 checkpoint 路徑相容：`--load_run <run> --checkpoint model_xxxx.pt` 可自動組出完整路徑  
- 失配檔案時自動 fallback 到 `model_*.pt`

6. 文件更新  
- `docs/redrhex_train_play_guide.md`（本文件）
- `docs/redrhex_forwardfast_professor_report.md`（獨立報告）

---

## D4) ForwardFast 主要設計（做法與理由）

### D4.1) 任務降維（只保留直走）

關鍵設定（`RedrhexForwardFastEnvCfg`）：
- `stage=1`
- `curriculum_auto_progress=False`
- `lin_vel_x_range=[0.20, 0.32]`
- `lin_vel_y_range=[0.0, 0.0]`
- `ang_vel_z_range=[0.0, 0.0]`

理由：
- 現場目標是「先打通直走」，不是一次解完整四技能。
- 把命令分布縮窄，能顯著提高樣本效率。

### D4.2) 控制器改成更穩定的 tripod 節奏

在 `redrhex_env.py`，forward bias 的 `desired_in_stance` 由固定角窗改為 duty-time：
- `desired_cycle < duty_target`

理由：
- 直接按照 duty 比例（例如 65/35）去排時間，較不容易出現六腳同相亂轉。

### D4.3) 加入「健康狀態才給正向獎勵」

新增：
- `gate_positive_rewards_when_unhealthy=True`
- `reward_gate_min_base_height=0.105`
- `reward_gate_max_body_tilt=0.70`

機制：
- 當 base 高度太低或 body tilt 太大時，forward/tracking/gait 等正向獎勵會被 gate。

理由：
- 防止策略學到「趴地抖腿也有分」的 reward hacking。

### D4.4) 終止與倒地懲罰提早介入

關鍵設定：
- `max_tilt_magnitude=0.75`
- `body_contact_height_threshold=0.090`
- `body_contact_tilt_threshold=0.72`
- `termination_grace_steps=4`
- `fall=-24.0`
- `fall_height_threshold=0.11`
- `fall_tilt_threshold=0.75`

理由：
- 讓低價值軌跡更快結束，把訓練預算留給可行步態。

### D4.5) 動作激進度下修，起步更平順

關鍵設定：
- `main_drive_vel_scale=5.8`
- `main_drive_residual_scale=0.07`
- `stage_forward_policy_drive_residual_scale=[0.03]`
- `stage_action_warmup_steps=[120]`

理由：
- 降低 reset 後大動作衝擊，減少起步翻倒。

---

## D5) PPO 設定（快速收斂但不失穩）

`PPORunnerForwardFastCfg`：
- `max_iterations=1500`
- `num_steps_per_env=32`
- `init_noise_std=0.30`
- `learning_rate=4.0e-4`
- `entropy_coef=0.0010`
- `desired_kl=0.008`

理由：
- 降低初期過度探索與策略抖動。
- 用更平穩更新把「會走」放在「只看曲線快升」之前。

---

## D6) 為什麼這樣改（方法論）

我們不是單純「加獎勵」，而是按優先順序做三層修正：

1. **控制器層**：先保證產生合理 gait 的可行空間。  
2. **獎勵層**：再用 gate 阻止不健康行為拿分。  
3. **優化層**：最後調 PPO，避免在壞策略附近亂晃。  

這比單純堆高某個 reward weight 更穩，也比較符合你希望的「盡量保留原始獎勵精神」。

---

## D7) 現場操作 SOP（Train / TensorBoard / Play）

### 訓練

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-ForwardFast-Direct-v0 \
  --headless \
  --num_envs=2048 \
  --max_iterations=1500 \
  --run_name=forward_fast_trial_a
```

### TensorBoard

```bash
tensorboard --logdir . --port 6006 --bind_all
```

### Play（可直接吃完整 checkpoint 路徑）

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-ForwardFast-Direct-v0 \
  --num_envs=64 \
  --disable_keyboard_control \
  --load_run="$RUN" \
  --checkpoint="$CKPT"
```

---

## D8) 驗證指標（教授可直接看）

最重要不是「有收斂」，而是「收斂到可上機行為」。

建議看：
1. `Train/mean_episode_length`：應穩定上升。  
2. `Episode_Termination/terminated`：要下降。  
3. `Episode_Reward/rew_fall`：負值幅度要收斂。  
4. `Episode_Reward/diag_base_height`：不能長期下滑。  
5. `Episode_Reward/rew_forward_gait` + `diag_forward_duty_ema`：tripod 節奏要穩。  

判讀原則：
- 若 `mean_reward` 上升但 `terminated` 不降，屬於假收斂，不可直接上機。

---

## D9) 目前狀態與限制

已完成：
1. ForwardFast task + env + PPO + play 相容修正皆已落地。  
2. `play.py` 的 checkpoint 常見錯誤已處理（basename / event 檔誤傳）。  
3. 文件與指令已同步到目前程式版本。  

限制：
1. ForwardFast 是現場快迭代工具，不是多技能最終模型。  
2. 仍需實機回饋資料持續微調。  

---

## D10) 報告給教授的建議講法（可直接拿去用）

1. 問題：現場 Sim2Real 需要高頻迭代，原多技能訓練週期太慢。  
2. 解法：建立 ForwardFast 任務，把問題降維到 forward-only。  
3. 成果：訓練週期縮短，且新增反作弊機制，避免「趴地也算收斂」。  
4. 核心技術：Stage1-reference stance 控制 + 健康姿態 gate（功能保留）+ 穩定化 PPO。  
5. 下一步：白天 ForwardFast 校參，夜間再接五階段整合。  

---

## D11) 對應獨立報告檔

詳細版請看：
- `docs/redrhex_forwardfast_professor_report.md`

## D12) 第二輪更新（v2.1，現在請以這版為準）

你最新回報「更不好動、起步後仍六腳同轉趴地」後，我們做了第二輪回調。  
這一輪的核心不是再加懲罰，而是 **回到原本五階段 Stage1 已驗證成功的控制骨架**，只保留 ForwardFast 的加速訓練優勢。

最終重點：
1. 控制器 stance 判定回退到原 Stage1：
- `desired_in_stance = self._in_stance_phase(desired_phase)`

2. ForwardFast 參數回到 Stage1 風格：
- `curriculum_stage_scales=[0.05]`
- `lin_vel_x_range=[0.22, 0.42]`
- `main_drive_vel_scale=8.0`
- `stage_forward_policy_drive_residual_scale=[0.10]`
- `stage_action_warmup_steps=[30]`

3. 終止條件從過嚴回到穩定區：
- `stage1_min_base_height=0.03`
- `stage1_body_contact_height_threshold=0.06`
- `stage_body_contact_tilt_threshold=[1.80]`
- `stage_termination_grace_steps=[120]`

4. PPO 恢復探索能力：
- `num_steps_per_env=24`
- `init_noise_std=0.55`
- `entropy_coef=0.0035`
- `learning_rate=5.0e-4`
- `desired_kl=0.01`

目前推薦的重訓入口：

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-ForwardFast-Direct-v0 \
  --headless \
  --num_envs=2048 \
  --max_iterations=1500 \
  --run_name=forward_fast_stage1ref_v1
```
