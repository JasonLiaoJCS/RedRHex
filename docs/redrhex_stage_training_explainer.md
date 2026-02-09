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

## 2.1) 每個 Stage 的 TensorBoard 驗收重點（你要看的 tag）

先看這 3 個共通原則，不然很容易誤判：
- 很多 `rew_*` 是「模式啟動才會有值」。例如你在 Stage1 看 `rew_yaw_track` 幾乎是 0，這是正常。
- 先看「趨勢」再看「絕對值」：同一組 scale 下，往正向收斂比單點數值更重要。
- 最後一定用 `eval_command_sweep.py` 做客觀驗收；TensorBoard 只負責訓練中監控。

### 共通必看（所有 Stage 都要看）
- `Train/mean_episode_length`：要明顯上升並穩住。若長期接近 1~20，通常是出生即死或 reward 爆炸。
- `Train/mean_reward`：應該從亂飄逐漸變穩。若一路往負無限或劇烈震盪，優先查終止與 action 映射。
- `Episode_Termination/terminated`：要下降。若長期很高，代表跌倒/觸地問題嚴重。
- `Episode_Reward/rew_fall`：理想是接近 0（少觸發）；若持續大負值，表示還在大量摔倒。
- `Episode_Reward/diag_base_height`：要接近站姿高度（目前設定目標約 0.30m 附近），不要一路掉到 0.1 以下。

### Stage1（Forward-only）要看什麼
命令是否正確：
- `Episode_Reward/diag_cmd_vx` 應為正值（約在 `stage1_forward_vx_range`）。
- `Episode_Reward/diag_cmd_vy`、`Episode_Reward/diag_cmd_wz` 應接近 0。
核心追蹤：
- `Episode_Reward/rew_tracking` 應上升並穩定。
- `Episode_Reward/diag_forward_vel` 應與 `diag_cmd_vx` 同號，且差距縮小（可搭配 `diag_vel_error` 下降）。
前進步態先驗（最關鍵）：
- `Episode_Reward/rew_forward_prior_coherence`、`rew_forward_prior_antiphase`、`rew_forward_prior_duty`、`rew_forward_prior_vel_ratio`、`rew_forward_prior_overlap` 應整體轉正並提升。
- `Episode_Reward/diag_forward_duty_ema` 目標靠近 0.65（常見可接受帶：0.50~0.75，越接近越好）。
- `Episode_Reward/diag_forward_vel_ratio_proxy` 應明顯 > 1，代表 swing 比 stance 快（越高越符合快擺慢撐）。
失敗警訊：
- `rew_tracking` 長期低迷 + `terminated` 高。
- `diag_forward_duty_ema` 長期偏離（例如 <0.35 或 >0.9）。
- `rew_stall` 長期負值很大（有命令卻不動）。

### Stage2（Lateral-only）要看什麼
命令是否正確：
- `Episode_Reward/diag_cmd_vy` 應非 0（正負都會出現）。
- `Episode_Reward/diag_cmd_vx`、`Episode_Reward/diag_cmd_wz` 應接近 0。
FSM 是否正常：
- `Episode_Reward/diag_lateral_fsm_state` 應從 `GO_TO_STAND(1)` 進到 `LATERAL_STEP(2)`，後期平均值應偏向 2。
- `Episode_Reward/diag_pose_error` 應下降（進入 LATERAL_STEP 前要變小）。
- `Episode_Reward/diag_contact_count` 在側移時不要長期掉到 0~1。
側移能力是否變積極：
- `Episode_Reward/rew_lateral_speed_deficit` 應「往 0 靠近」（這個值通常是負，越不負越好）。
- `Episode_Reward/diag_lateral_vel` 要和 `diag_cmd_vy` 同號，且絕對值變大。
- `Episode_Reward/rew_mode`（在 lateral 段）應提升。
鎖主驅動是否生效：
- `Episode_Reward/diag_masked_action_norm_main` 應維持低值（硬鎖接近 0，soft-lock 也不應長期很大）。
失敗警訊：
- 卡在 `diag_lateral_fsm_state ≈ 1` 很久（GO_TO_STAND 出不去）。
- `rew_lateral_speed_deficit` 長期大幅負值 + `diag_lateral_vel` 接近 0（站著不動）。

### Stage3（Diagonal-only）要看什麼
命令是否正確：
- `Episode_Reward/diag_cmd_vx` 應為正，`diag_cmd_vy` 應有正負切換，`diag_cmd_wz` 接近 0。
斜向組合能力：
- `Episode_Reward/rew_diag_sign` 要轉正並穩定（代表 vx/vy 方向符號正確）。
- `Episode_Reward/rew_mode`（diag 段）應提升。
- `Episode_Reward/diag_vel_error` 應下降。
干擾軸抑制：
- `Episode_Reward/rew_axis_suppression` 不要越來越負（過負表示未命令軸漏動嚴重）。
失敗警訊：
- `rew_diag_sign` 長期 <= 0（常見是只會某一側，例如只會 diag_left）。
- `diag_lateral_vel` 或 `diag_forward_vel` 長期某一軸太小，導致不像真正斜走。

### Stage4（Yaw-only）要看什麼
命令是否正確：
- `Episode_Reward/diag_cmd_wz` 應非 0（正負都會出現）。
- `Episode_Reward/diag_cmd_vx`、`diag_cmd_vy` 應接近 0。
旋轉是否真的成立：
- `Episode_Reward/rew_yaw_track` 應上升。
- `Episode_Reward/diag_actual_wz` 應和 `diag_cmd_wz` 同號，且差距縮小（`diag_wz_error` 下降）。
穩定性（避免「掀機身作弊」）：
- `Episode_Reward/rew_yaw_stability`、`rew_yaw_cheat` 通常是負值，理想是逐步往 0 靠近。
- `Episode_Reward/diag_roll_rms`、`diag_pitch_rms` 要壓低（建議先看是否能壓到 <0.35 rad）。
- `Episode_Reward/diag_yaw_slip_proxy` 要下降（降低平移滑移）。
- `Episode_Reward/diag_base_height` 不可長期崩到很低。
失敗警訊：
- `rew_yaw_track` 低 + `rew_yaw_cheat` 很負 + `terminated` 高，通常就是一轉就翻。

### Stage5（Mixed integration）要看什麼
先確認「不是只剩單一技能」：
- `Episode_Reward/rew_forward_gait`、`rew_lateral_speed_deficit`、`rew_diag_sign`、`rew_yaw_track` 這四組都要有活動，不應長期只剩一組有反應。
綜合能力：
- `Episode_Reward/rew_tracking` 應維持高位。
- `Episode_Reward/rew_mode` 應為正向貢獻（或至少比 stage4 初期更好）。
- `Episode_Reward/rew_fall`、`Episode_Termination/terminated` 要壓低。
- `Episode_Reward/rew_smooth` 不要極端負值（避免抖動控制）。
失敗警訊：
- 某技能相關 tag 完全貼地（接近 0 或持續惡化），表示混訓時被其他技能「覆蓋」。
- `rew_tracking` 看似高，但 `rew_mode` 長期很差，代表可能靠錯誤運動模式偷分。

### 你可以直接照抄的 Stage 成功判定（TensorBoard 快速版）
- Stage1 成功：`rew_tracking` 穩定上升、`rew_forward_gait` 為正且成長、`diag_forward_duty_ema` 靠近 0.65、`terminated` 下降。
- Stage2 成功：`diag_lateral_fsm_state` 多數時間在 2、`rew_lateral_speed_deficit` 趨近 0、`diag_lateral_vel` 顯著上升且方向正確。
- Stage3 成功：`rew_diag_sign` 穩定為正、`diag_vel_error` 下降、`rew_mode` 提升。
- Stage4 成功：`rew_yaw_track` 上升、`diag_wz_error` 下降、`rew_yaw_stability/rew_yaw_cheat` 往 0 靠近、`terminated` 不爆量。
- Stage5 成功：四類技能 tag 都活著，且 `rew_tracking` 高、`rew_fall` 低、`terminated` 低，最後 eval profile=`stage5/full` 能過線。

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
