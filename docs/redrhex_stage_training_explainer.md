# RedRhex 分段訓練說明（Stage1~Stage5）

## 1) 你從「一次混合訓練」改成了什麼

你原本是單階段直接混合 forward/lateral/diagonal/yaw。現在改成 5 階段 curriculum：

1. Stage1：只練 forward。
2. Stage2：只練 lateral。
3. Stage3：只練 diagonal。
4. Stage4：只練 yaw。
5. Stage5：混合所有技能做整合與潤化。

這不是 5 個獨立模型，而是同一個 policy 連續微調。每一段都用上一段 checkpoint 接續。

---

## 2) 每個 Stage 的責任

## Stage1（Forward-only）
- 目標：先把前進穩定走好。
- 規則：`vx>0, vy=0, wz=0`。
- 硬限制：ABAD 鎖住。
- 重點：tripod 節奏、前進追蹤、不倒地。

## Stage2（Lateral-only）
- 目標：把側移能力獨立學出來，不被前進策略吞掉。
- 規則：`vy!=0, vx≈0, wz≈0`。
- 硬限制：main-drive lock/soft-lock。
- 額外流程：`GO_TO_STAND -> LATERAL_STEP`。

## Stage3（Diagonal-only）
- 目標：學會 `vx + vy` 同時成立時的合成動作。
- 規則：`vx>0 且 vy!=0, wz≈0`。
- 特徵：主驅動 + ABAD 一起用，強化斜向符號正確性（左前/右前）。

## Stage4（Yaw-only）
- 目標：把原地旋轉單獨練穩。
- 規則：`wz!=0, vx≈0, vy≈0`。
- 核心：允許每腿正反轉（signed main-drive），學 differential/skid-steer 旋轉。
- 重點：`wz` 追蹤 + roll/pitch 穩定 + 抑制「掀機身作弊」。

## Stage5（Mixed integration）
- 目標：整合前 4 段能力到同一個 command-conditioned policy。
- 不是暴力拼接：
1. 載入 Stage4 checkpoint。
2. 混合命令分布（FWD/LAT/DIAG/YAW）。
3. 維持 mode gating 硬限制。
4. 持續 PPO 更新，消除技能互相干擾。

---

## 3) 這次程式上的關鍵改動（對外可講）

1. 命令採樣改成 5-stage（stage1~stage5），每段 command 分布分離。
2. `_apply_action()` 改成 signed main-drive mapping（允許反轉），yaw 可 differential。
3. lateral FSM：`NORMAL -> GO_TO_STAND -> LATERAL_STEP`，含 timeout 防卡死。
4. lateral 加了速度不足懲罰，不再「站著不動也能過」。
5. diagonal 加了符號一致性獎勵（`vx/vy` 方向要對）。
6. yaw 加強穩定項與反作弊項，並在大傾斜時降低 yaw 驅動輸出。
7. base height reward 改成站姿附近目標，而不是舊的過低目標。
8. fall 判定收緊，避免翻倒資料污染訓練。

---

## 4) 完整串接指令（每段接續上一段）

## 前置

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

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
  --load_run="$RUN1" \
  --checkpoint="$CKPT1" \
  env.stage=2 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

```bash
RUN2=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage2* | head -1)")
CKPT2=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN2/model_*.pt | tail -1)")
```

## Stage3（接 Stage2）

```bash
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

```bash
RUN3=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage3* | head -1)")
CKPT3=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN3/model_*.pt | tail -1)")
```

## Stage4（接 Stage3）

```bash
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

```bash
RUN4=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage4* | head -1)")
CKPT4=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN4/model_*.pt | tail -1)")
```

## Stage5（接 Stage4，最終整合）

```bash
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

```bash
RUN5=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage5* | head -1)")
FINAL_CKPT=$(ls -v logs/rsl_rl/redrhex_wheg/$RUN5/model_*.pt | tail -1)
```

---

## 4.1) 一次自動串完五個 Stage（建議夜跑）

如果你不想手動一段一段接，直接跑：

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh
```

自訂迭代數：

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --s1 8000 --s2 8000 --s3 9000 --s4 10000 --s5 12000
```

這支腳本會自動：
1. 跑 Stage1。
2. 自動抓 Stage1 最後 `model_*.pt`。
3. `--resume` 接續到 Stage2。
4. 重複到 Stage5。
5. 最後印出 `FINAL_CKPT` 給你直接評估/播放。

---

## 5) 評估程式怎麼用與怎麼判讀

## 指令

新版評估腳本有 `--eval_profile`，可直接對應五階段：
- `stage1` 前進
- `stage2` 側移
- `stage3` 斜向
- `stage4` 旋轉
- `stage5` 最終混合
- `full` 全部命令一次掃描

### 範例 A：只驗 Stage4（Yaw-only）

```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=256 \
  --checkpoint="$CKPT4" \
  --eval_profile=stage4 \
  --warmup_steps=120 \
  --sweep_steps=600 \
  --accept_duration_s=2.0 \
  --accept_wz_abs=0.35 \
  --accept_wz_ratio=0.55 \
  --accept_yaw_tilt_bound=0.60 \
  --accept_yaw_tilt_ratio=0.70 \
  --accept_yaw_lin_leak=0.18 \
  --accept_max_fall_rate=0.20 \
  --accept_skill_pass_ratio=0.70 \
  --accept_overall_pass_ratio=0.70
```

### 範例 B：最終 Stage5 混合驗收

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

## 重點欄位
- `Skill Acceptance`：先看 PASS/FAIL（left/right/diag/yaw）。
- `Skill-level Pass Ratio`：看每個 skill 的 pass ratio 是否過線。
- `Overall Acceptance`：看 `command_pass_ratio` 與 `min_skill_pass_ratio` 是否都達標。
- `tracking.mean|vx-vx_cmd| / |vy-vy_cmd| / |wz-wz_cmd|`：越低越好。
- `stability.fall_rate`：要接近 0。
- `stability.roll_rms / pitch_rms`：過高代表在翻滾。
- `forward.stance_fraction_abs_err_to_0.65`：越低越好。
- `forward.swing_to_stance_speed_ratio`：應明顯 > 1。

---

## 6) 最終播放

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-Direct-v0 \
  --num_envs=64 \
  --checkpoint="$FINAL_CKPT"
```

Headless + 錄影：

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --video \
  --video_length=1000 \
  --num_envs=64 \
  --checkpoint="$FINAL_CKPT"
```
