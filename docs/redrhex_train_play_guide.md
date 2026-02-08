# RedRhex 分段訓練 / 播放完整流程（可直接複製）

## 0. 先講結論
- 你的任務使用 Hydra，**環境參數要寫 `env.xxx`**，例如 `env.stage=1`。
- 分段訓練要真的「接續」，每一段都要帶：
  - `--resume`
  - `--load_run=<前一段 run 資料夾名>`
  - `--checkpoint=<前一段 checkpoint 檔名>`
- 訓練時主要檔案是 `model_*.pt`（不是固定叫 `model.pt`）。
- `play.py` 載入後會另外輸出：`exported/policy.pt`、`exported/policy.onnx`。

---

## 1. 前置

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

可先確認任務可被找到：

```bash
python scripts/rsl_rl/train.py --task=Template-Redrhex-Direct-v0 --help
```

---

## 2. Stage 1（Forward-only）

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

### 2.1 取得 Stage 1 的 run 名稱與最後 checkpoint

```bash
RUN1=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage1* | head -1)")
CKPT1=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN1/model_*.pt | tail -1)")

echo "RUN1=$RUN1"
echo "CKPT1=$CKPT1"
```

---

## 3. Stage 2（Lateral-only，接續 Stage 1）

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
```

```bash
RUN2=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage2* | head -1)")
CKPT2=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN2/model_*.pt | tail -1)")

echo "RUN2=$RUN2"
echo "CKPT2=$CKPT2"
```

---

## 4. Stage 3（Yaw-only，接續 Stage 2）

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
```

```bash
RUN3=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage3* | head -1)")
CKPT3=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN3/model_*.pt | tail -1)")

echo "RUN3=$RUN3"
echo "CKPT3=$CKPT3"
```

---

## 5. Stage 4（Mixed-skills，接續 Stage 3）

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
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

```bash
RUN4=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage4* | head -1)")
FINAL_CKPT=$(ls -v logs/rsl_rl/redrhex_wheg/$RUN4/model_*.pt | tail -1)

echo "RUN4=$RUN4"
echo "FINAL_CKPT=$FINAL_CKPT"
```

---

## 6. model 檔案到底在哪

### 6.1 訓練 checkpoint（給 train resume / play 載入）

位置：

```text
logs/rsl_rl/redrhex_wheg/<run資料夾>/model_*.pt
```

例如：

```text
logs/rsl_rl/redrhex_wheg/2026-02-08_15-06-43_wheg_locomotion_v3/model_3000.pt
```

### 6.2 Play 匯出模型

`play.py` 跑完會輸出：

```text
logs/rsl_rl/redrhex_wheg/<run資料夾>/exported/policy.pt
logs/rsl_rl/redrhex_wheg/<run資料夾>/exported/policy.onnx
```

### 6.3 一鍵找最新 checkpoint

```bash
find logs/rsl_rl/redrhex_wheg -type f -name "model_*.pt" | sort | tail -1
```

---

## 7. 分段訓練完如何 Play

### 7.1 GUI 觀察（可鍵盤控制）

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-Direct-v0 \
  --num_envs=64 \
  --checkpoint="$FINAL_CKPT"
```

### 7.2 Headless 錄影片

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --video \
  --video_length=1000 \
  --num_envs=64 \
  --checkpoint="$FINAL_CKPT"
```

---

## 8. 驗收評估（Command Sweep）

先確認 `FINAL_CKPT` 不是空字串（避免腳本自動回退去抓錯 run）：

```bash
echo "$FINAL_CKPT"
test -n "$FINAL_CKPT" && test -f "$FINAL_CKPT" || { echo "FINAL_CKPT 無效"; exit 1; }
```

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

如果你怕 shell 變數失效，建議直接在同一行先抓最新 stage4 checkpoint 再評估：

```bash
RUN4=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage4* | head -1)")
FINAL_CKPT=$(ls -v logs/rsl_rl/redrhex_wheg/$RUN4/model_*.pt | tail -1)
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

如果要輸出表格：

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
  --accept_diag_sign_ratio=0.70 \
  --csv logs/rsl_rl/redrhex_wheg/${RUN4}/eval_command_sweep.csv
```

---

## 9. 跟「沒分段」有何不同
- 沒分段：通常一開始就混合命令，forward 容易主導，`vy / wz` 技能學得慢或失敗。
- 分段：先學 FWD，再 LAT，再 YAW，最後 MIXED，穩定性通常較好。
- 但**只有加 `--resume --load_run --checkpoint` 才是接續**；沒加就是新模型重訓。

---

## 10. 常見錯誤與修正

### 錯誤 1：`Could not override 'stage'`
- 原因：用了 `stage=1`。
- 修正：改 `env.stage=1`。

### 錯誤 2：每個 stage 都從頭訓練
- 原因：忘記 `--resume/--load_run/--checkpoint`。
- 修正：第 2 段起都要帶這三個參數。

### 錯誤 3：找不到 checkpoint
- 先列目錄檢查：

```bash
ls -td logs/rsl_rl/redrhex_wheg/* | head
find logs/rsl_rl/redrhex_wheg -maxdepth 2 -type f -name "model_*.pt" | sort | tail -20
```

### 錯誤 4：`UnicodeEncodeError: surrogates not allowed`（rsl_rl 寫 git diff 時）
- 現在 `scripts/rsl_rl/train.py` 已預設關閉 code-state git diff 存檔，不會再因為這個錯誤中斷。
- 若你想強制保存 git diff，可手動加 `--store_code_state`，但若 repo 內有非 UTF-8 檔名仍可能報同錯誤。
