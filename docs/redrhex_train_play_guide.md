# RedRhex 五階段訓練整合指南（Train + Explainer + Eval + Play）

> 這份文件已整併原本兩份：
> - `docs/redrhex_train_play_guide.md`
> - `docs/redrhex_stage_training_explainer.md`

## 1) 為什麼從單段改成五階段

你原本是一次混合訓練所有技能（forward/lateral/diagonal/yaw）。
現在改成 **5-stage curriculum**，目的是降低技能互相干擾：

1. Stage1: Forward-only（先把直走練穩）
2. Stage2: Lateral-only（把側移獨立練出來）
3. Stage3: Diagonal-only（融合 vx+vy）
4. Stage4: Yaw-only（先把原地旋轉單獨練穩）
5. Stage5: Mixed（整合前四段，持續潤化）

這 5 段不是 5 個模型，而是 **同一個 policy 連續微調**：每段都接續上一段 checkpoint。

---

## 2) 每個 Stage 在做什麼

## Stage1（Forward-only）
- 命令分布：`vx > 0, vy = 0, wz = 0`
- 硬限制：ABAD 鎖住
- 主要目標：穩定前進 + tripod 節奏 + 不倒地

## Stage2（Lateral-only）
- 命令分布：`vy != 0, vx ~= 0, wz ~= 0`
- 硬限制：main-drive lock/soft-lock
- 流程：`GO_TO_STAND -> LATERAL_STEP`
- 主要目標：側移速度要起來，不是站著抖動

## Stage3（Diagonal-only）
- 命令分布：`vx > 0 且 vy != 0, wz ~= 0`
- 控制：main-drive + ABAD 都開放
- 主要目標：vx/vy 方向同時正確（不能只會一邊斜走）

## Stage4（Yaw-only）
- 命令分布：`wz != 0, vx ~= 0, vy ~= 0`
- 控制：允許 per-leg signed main-drive（可反轉）
- 主要目標：原地旋轉穩定，避免「掀機身作弊」

## Stage5（Mixed）
- 命令分布：FWD/LAT/DIAG/YAW 混合
- 主要目標：整合技能且保留 Stage1 的穩定直走能力

---

## 3) 現在程式如何保留「穩定直走」不被後續洗掉

目前環境已加入前進防遺忘機制：
- Forward 模式殘差上限（cap）避免 policy 大幅破壞前進 bias
- stage-specific main-drive residual scale（各 stage 強度不同）
- Stage5 仍保留 forward 專屬 reward multiplier

你可以把 Stage5 理解成：
- 不是暴力覆蓋前四段
- 是在保護 forward 基底下，把 lateral/diag/yaw 慢慢拉上來

---

## 4) 訓練前建議（先肉眼確認）

## 前置

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

## 建議先做 GUI 預檢（非 headless）

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

## 5) 一鍵跑完五階段（推薦）

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --num_envs 4096 \
  --s1 8000 --s2 8000 --s3 9000 --s4 10000 --s5 12000
```

補充：
- pipeline 會自動串接 stage checkpoint
- 現在即使系統沒有 `rg`，也會 fallback `grep`，不會因缺 `rg` 直接中斷
- 會輸出：`FINAL_CKPT=.../model_xxxxx.pt`

## 若中斷後續跑

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --start_stage 2 \
  --num_envs 4096 \
  --s1 8000 --s2 8000 --s3 9000 --s4 10000 --s5 12000
```

---

## 6) 手動逐段跑（可精細控制）

## Stage1

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

## Stage2（接 Stage1）

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=8000 \
  --run_name=stage2 \
  --resume \
  --resume_policy_only \
  --reset_action_std=0.8 \
  --load_run="$RUN1" \
  --checkpoint="$CKPT1" \
  env.stage=2 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

## Stage3（接 Stage2）

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
  --resume_policy_only \
  --reset_action_std=0.8 \
  --load_run="$RUN2" \
  --checkpoint="$CKPT2" \
  env.stage=3 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

## Stage4（接 Stage3）

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
  --resume_policy_only \
  --reset_action_std=0.8 \
  --load_run="$RUN3" \
  --checkpoint="$CKPT3" \
  env.stage=4 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

## Stage5（接 Stage4）

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
  --resume_policy_only \
  --reset_action_std=0.8 \
  --load_run="$RUN4" \
  --checkpoint="$CKPT4" \
  env.stage=5 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

```bash
RUN5=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage5* | head -1)")
FINAL_CKPT=$(ls -v logs/rsl_rl/redrhex_wheg/$RUN5/model_*.pt | tail -1)
echo "FINAL_CKPT=$FINAL_CKPT"
```

---

## 7) TensorBoard 每階段要看什麼

## 全階段共通
- `Train/mean_episode_length`：上升且穩定
- `Train/mean_reward`：整體往上
- `Episode_Termination/terminated`：下降
- `Episode_Reward/rew_fall`：接近 0（少摔）
- `Episode_Reward/diag_base_height`：不要長期崩掉

## Stage1
- `diag_cmd_vx` > 0，`diag_cmd_vy/wz` ~ 0
- `rew_tracking`、`rew_forward_gait` 上升
- `diag_forward_duty_ema` 靠近 0.65

## Stage2
- `diag_lateral_fsm_state` 能進到 2（LATERAL_STEP）
- `rew_lateral_speed_deficit` 往 0 靠近（少負）
- `diag_lateral_vel` 絕對值上升且方向正確

## Stage3
- `rew_diag_sign` 轉正且穩定
- `diag_vel_error` 下降

## Stage4
- `rew_yaw_track` 上升
- `diag_wz_error` 下降
- `diag_roll_rms`、`diag_pitch_rms` 不爆

## Stage5
- forward/lateral/diag/yaw 四類指標都要有反應
- 不能只剩一種技能表現好

---

## 8) 驗收（Command Sweep）

## 最終混合驗收（Stage5）

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

## 9) Play（播放最終模型）

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-Direct-v0 \
  --num_envs=64 \
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
  --checkpoint="$FINAL_CKPT"
```

注意：`--checkpoint` 只能是一個完整路徑，不要把兩段路徑黏在一起。

---

## 10) 常見問題（你最近遇過的）

## Q1: Stage1 訓練完就中斷（`rg: 指令找不到`）
- 原因：舊 pipeline 健康檢查解析依賴 `rg`
- 現況：已改為 `rg` 不存在時自動 fallback `grep`

## Q2: `stage=1` 覆寫失敗
- 要用 `env.stage=1`（不是 `stage=1`）

## Q3: `play.py` 找不到 checkpoint
- 請先 `echo "$FINAL_CKPT"` 確認只是一條路徑
- 再把同一條路徑傳給 `--checkpoint`

## Q4: `omni.platforminfo` CPU 錯誤很多
- 多數情況是平台偵測告警，不是主因
- 優先看：是否一出生就 `terminated`、是否 base height 崩掉

---

## 11) 建議訓練節奏（避免再浪費整晚）

1. 先跑 GUI precheck（2~5 分鐘）
2. 再跑 Stage1 500~1000 iter 快篩，確認非出生即死
3. 通過後再放整晚全 pipeline
4. 隔天先跑 `eval_command_sweep.py` 再決定是否繼續加訓
