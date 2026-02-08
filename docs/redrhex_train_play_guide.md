# RedRhex 分段訓練與播放完整流程（Hydra 版）

## 0) 重點
- Hydra override 必須用 `env.<key>`，例如 `env.stage=1`。
- 你之前報錯 `Could not override 'stage'` 是因為用了 `stage=1`（頂層 key）。
- 分段訓練要串接 checkpoint：`stage1 -> stage2 -> stage3 -> stage4`。

## 1) 前置
```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

## 2) Stage 1 (Forward)
```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage1 \
  env.stage=1 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

取得 Stage1 產生的 run / checkpoint：
```bash
RUN1=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage1* | head -1)")
CKPT1=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN1/model_*.pt | tail -1)")
echo "$RUN1"
echo "$CKPT1"
```

## 3) Stage 2 (Lateral) 接續 Stage1
```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage2 \
  --resume \
  --load_run="$RUN1" \
  --checkpoint="$CKPT1" \
  env.stage=2 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False

RUN2=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage2* | head -1)")
CKPT2=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN2/model_*.pt | tail -1)")
```

## 4) Stage 3 (Yaw) 接續 Stage2
```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage3 \
  --resume \
  --load_run="$RUN2" \
  --checkpoint="$CKPT2" \
  env.stage=3 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False

RUN3=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage3* | head -1)")
CKPT3=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN3/model_*.pt | tail -1)")
```

## 5) Stage 4 (Mixed) 接續 Stage3
```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage4 \
  --resume \
  --load_run="$RUN3" \
  --checkpoint="$CKPT3" \
  env.stage=4 \
  env.draw_debug_vis=False \# RedRhex 分段訓練與播放完整流程（Hydra 版）

## 0) 重點
- Hydra override 必須用 `env.<key>`，例如 `env.stage=1`。
- 你之前報錯 `Could not override 'stage'` 是因為用了 `stage=1`（頂層 key）。
- 分段訓練要串接 checkpoint：`stage1 -> stage2 -> stage3 -> stage4`。

## 1) 前置
```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

## 2) Stage 1 (Forward)
```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage1 \
  env.stage=1 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

取得 Stage1 產生的 run / checkpoint：
```bash
RUN1=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage1* | head -1)")
CKPT1=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN1/model_*.pt | tail -1)")
echo "$RUN1"
echo "$CKPT1"
```

## 3) Stage 2 (Lateral) 接續 Stage1
```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage2 \
  --resume \
  --load_run="$RUN1" \
  --checkpoint="$CKPT1" \
  env.stage=2 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False

RUN2=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage2* | head -1)")
CKPT2=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN2/model_*.pt | tail -1)")
```

## 4) Stage 3 (Yaw) 接續 Stage2
```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage3 \
  --resume \
  --load_run="$RUN2" \
  --checkpoint="$CKPT2" \
  env.stage=3 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False

RUN3=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage3* | head -1)")
CKPT3=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN3/model_*.pt | tail -1)")
```

## 5) Stage 4 (Mixed) 接續 Stage3
```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  env.dr_try_physical_material_randomization=False

RUN4=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage4* | head -1)")
FINAL_CKPT=$(ls -v logs/rsl_rl/redrhex_wheg/$RUN4/model_*.pt | tail -1)
echo "$FINAL_CKPT"
```

## 6) Play（載入最終模型）
```bash# RedRhex 分段訓練與播放完整流程（Hydra 版）

## 0) 重點
- Hydra override 必須用 `env.<key>`，例如 `env.stage=1`。
- 你之前報錯 `Could not override 'stage'` 是因為用了 `stage=1`（頂層 key）。
- 分段訓練要串接 checkpoint：`stage1 -> stage2 -> stage3 -> stage4`。

## 1) 前置
```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

## 2) Stage 1 (Forward)
```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage1 \
  env.stage=1 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

取得 Stage1 產生的 run / checkpoint：
```bash
RUN1=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage1* | head -1)")
CKPT1=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN1/model_*.pt | tail -1)")
echo "$RUN1"
echo "$CKPT1"
```

## 3) Stage 2 (Lateral) 接續 Stage1
```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage2 \
  --resume \
  --load_run="$RUN1" \
  --checkpoint="$CKPT1" \
  env.stage=2 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False

RUN2=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage2* | head -1)")
CKPT2=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN2/model_*.pt | tail -1)")
```

## 4) Stage 3 (Yaw) 接續 Stage2
```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage3 \
  --resume \
  --load_run="$RUN2" \
  --checkpoint="$CKPT2" \
  env.stage=3 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False

RUN3=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage3* | head -1)")
CKPT3=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN3/model_*.pt | tail -1)")
```

## 5) Stage 4 (Mixed) 接續 Stage3
```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
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

## 7) 驗收評估
```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=256 \
  --checkpoint="$FINAL_CKPT" \
  --warmup_steps=120 \
  --sweep_steps=600 \
  --accept_duration_s=2.0 \
  --accept_vy_abs=0.15 \
  --accept_wz_abs=0.40 \
  --accept_yaw_tilt_bound=0.60 \
  --accept_diag_sign_ratio=0.70
```

## 8) `model.pt` / `model_*.pt` 在哪裡
- 訓練 checkpoints：`logs/rsl_rl/redrhex_wheg/<run>/model_*.pt`
- Play 匯出：`logs/rsl_rl/redrhex_wheg/<run>/exported/policy.pt`、`policy.onnx`

快速找最新 checkpoint：
```bash
find logs/rsl_rl/redrhex_wheg -type f -name "model_*.pt" | sort | tail -1
```

## 9) 跟「沒分段訓練」的差異
- 沒分段：一開始就混合命令，常被 forward 主導，`vy/wz` 容易學不起來。
- 分段：先學單技能（FWD/LAT/YAW），最後 mixed，收斂更穩定。
- 關鍵是是否有 `--resume --load_run --checkpoint` 串接前一階段權重。
