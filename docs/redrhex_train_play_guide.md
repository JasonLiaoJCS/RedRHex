# RedRhex 五階段訓練 / 驗收 / 播放指南

## 0) 先記住三件事
- Hydra 覆寫要用 `env.xxx`，例如 `env.stage=3`。
- 分段訓練要「接續」必須帶：`--resume --load_run --checkpoint`。
- 訓練權重檔是 `model_*.pt`，不是固定 `model.pt`。

---

## 1) 前置

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

---

## 2) Stage1：Forward-only

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

---

## 3) Stage2：Lateral-only（接 Stage1）

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
echo "RUN2=$RUN2"
echo "CKPT2=$CKPT2"
```

---

## 4) Stage3：Diagonal-only（接 Stage2）

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
echo "RUN3=$RUN3"
echo "CKPT3=$CKPT3"
```

---

## 5) Stage4：Yaw-only（接 Stage3）

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
echo "RUN4=$RUN4"
echo "CKPT4=$CKPT4"
```

---

## 6) Stage5：Mixed skills（接 Stage4，最終整合）

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
echo "RUN5=$RUN5"
echo "FINAL_CKPT=$FINAL_CKPT"
```

---

## 7) `model_*.pt` 在哪裡

每段訓練都在：

```text
logs/rsl_rl/redrhex_wheg/<run_name>/model_*.pt
```

快速找最新：

```bash
find logs/rsl_rl/redrhex_wheg -type f -name "model_*.pt" | sort | tail -1
```

---

## 8) 一次自動跑完 Stage1~5（夜跑模式）

你不用再手動每段抓 checkpoint。直接用：

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh
```

自訂範例：

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --num_envs 4096 \
  --s1 8000 --s2 8000 --s3 9000 --s4 10000 --s5 12000
```

跑完會輸出 `FINAL_CKPT`，並寫 log 到：

```text
logs/rsl_rl/pipeline/<run_tag>.log
```

---

## 9) 驗收評估（Command Sweep）

新版本 `eval_command_sweep.py` 支援 `--eval_profile`，可直接對應 5 階段：
- `stage1`: 只測前進
- `stage2`: 只測側移
- `stage3`: 只測斜向
- `stage4`: 只測原地旋轉
- `stage5`: 最終混合技能（預設）
- `full`: 一次掃 stage1~5 全部命令

### Stage 專屬驗收（範例：Stage3）

```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=256 \
  --checkpoint="$CKPT3" \
  --eval_profile=stage3 \
  --warmup_steps=120 \
  --sweep_steps=600 \
  --accept_duration_s=2.0 \
  --accept_diag_sign_ratio=0.75 \
  --accept_diag_component_ratio=0.55 \
  --accept_skill_pass_ratio=0.70 \
  --accept_overall_pass_ratio=0.70
```

### 最終混合驗收（Stage5 checkpoint）

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

可加 CSV：

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

## 10) Play 最終模型

GUI 觀察：

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

`play.py` 會輸出：

```text
logs/rsl_rl/redrhex_wheg/<run>/exported/policy.pt
logs/rsl_rl/redrhex_wheg/<run>/exported/policy.onnx
```
