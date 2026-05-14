# RedRHex 新 Ubuntu 操作指令

更新日期：2026-04-25

這份文件記錄這台 Ubuntu 上 RedRHex + Isaac Sim + Isaac Lab 的實際操作流程。這台機器的路徑和舊電腦不同，請以本文件為準。

## 1. 這台機器的固定路徑

| 項目 | 路徑 / 名稱 |
|---|---|
| RedRHex repo | `/home/lab_user1/Py/RedRHex` |
| Isaac Lab | `/home/lab_user1/isaac_lab_ws/IsaacLab` |
| Isaac Lab launcher | `/home/lab_user1/isaac_lab_ws/IsaacLab/isaaclab.sh` |
| Isaac Sim | `/home/lab_user1/isaacsim` |
| 正確 conda env | `env_isaaclab_bin` |
| 不要用的 env | `base`, `env_isaaclab` |
| 任務名稱 | `Template-Redrhex-Direct-v0` |
| RSL-RL train script | `scripts/rsl_rl/train.py` |
| RSL-RL play script | `scripts/rsl_rl/play.py` |
| 訓練輸出 | `logs/rsl_rl/redrhex_wheg/<timestamp>_wheg_locomotion/` |

目前版本：

- Isaac Lab repo version: `2.3.2`
- Isaac Sim version: `5.1.0-rc.19`
- Python env: `/home/lab_user1/miniconda3/envs/env_isaaclab_bin`
- GPU: NVIDIA GeForce RTX 5080, 16 GB VRAM

## 2. 每次開新終端機先做這段

```bash
export REDRHEX_ROOT=/home/lab_user1/Py/RedRHex
export ISAACLAB_ROOT=/home/lab_user1/isaac_lab_ws/IsaacLab
export ISAACSIM_ROOT=/home/lab_user1/isaacsim

source /home/lab_user1/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab_bin

# 有些非互動終端會出現 "tabs: terminal type 'dumb' cannot reset tabs"，加這行可避免。
export TERM=xterm

cd "$REDRHEX_ROOT"
```

重點：請從 RedRHex repo 根目錄執行 train/play。這樣 log 會存在 `RedRHex/logs/...`，不會跑去 IsaacLab repo 裡。

## 3. 第一次 setup 或換機器後要做

### 3.1 安裝 RedRHex package

```bash
cd /home/lab_user1/Py/RedRHex
python -m pip install -e source/RedRhex
```

這一步會讓 `import RedRhex.tasks` 成功，Isaac Lab 才能註冊 `Template-Redrhex-Direct-v0`。

### 3.2 安裝並拉 Git LFS 資源

這個 repo 的 `RedRhex.usd`、產生出的 robot USD、以及 STL mesh 資源走 Git LFS。若沒有拉 LFS，檔案會只有 pointer 內容，Isaac Sim 會打不開。

```bash
conda install -n env_isaaclab_bin -c conda-forge git-lfs -y
conda activate env_isaaclab_bin

cd /home/lab_user1/Py/RedRHex
git lfs install --local
git lfs pull
```

確認 `RedRhex.usd` 和 repo-local robot USD 是真檔案：

```bash
ls -lh RedRhex.usd
file RedRhex.usd
ls -lh test_7_description/test_7_description/urdf/test_7/test_7.usd
```

正常應該看到大約 `19K`，而且 `file` 顯示 `USD crate`。

## 4. 驗證環境是否接好

### 4.1 確認 Isaac Lab launcher 使用正確 Python

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py --help
```

輸出裡應該看到：

```text
[INFO] Using python from: /home/lab_user1/miniconda3/envs/env_isaaclab_bin/bin/python
```

### 4.2 列出 RedRHex task

```bash
$ISAACLAB_ROOT/isaaclab.sh -p -u scripts/list_envs.py
```

應該看到：

```text
Template-Redrhex-Direct-v0
```

注意：不要用單純的 `python scripts/rsl_rl/train.py` 當標準指令。這版 binary Isaac Sim 需要透過 `isaaclab.sh -p` 接好 Isaac Sim / USD / pxr 的 Python 路徑。

## 5. 開始 train

### 5.1 最小 smoke test

先用 4 個環境、1 次 iteration 測試程式能不能建場景：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 4 \
  --max_iterations 1 \
  --headless \
  --device cuda:0
```

### 5.2 小規模 debug train

用來看 reward、動作、觀測有沒有明顯錯誤：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 64 \
  --max_iterations 100 \
  --headless \
  --device cuda:0
```

### 5.3 正式 train

這台 GPU 是 RTX 5080 16 GB，建議先從 `1024` 或 `2048` 開始。若 VRAM 還夠，再提高到 `4096`。

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 2048 \
  --max_iterations 2000 \
  --headless \
  --device cuda:0
```

如果要開 GUI 看畫面，不加 `--headless`，並把環境數降小：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 64 \
  --max_iterations 100 \
  --device cuda:0
```

## 6. 從 checkpoint 繼續 train

先找最近的 run：

```bash
ls -td logs/rsl_rl/redrhex_wheg/* | head
```

找某個 run 裡的模型：

```bash
ls -lh logs/rsl_rl/redrhex_wheg/<run_name>/model_*.pt
```

接著繼續訓練：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 2048 \
  --max_iterations 2000 \
  --headless \
  --device cuda:0 \
  --resume \
  --checkpoint logs/rsl_rl/redrhex_wheg/<run_name>/model_<iteration>.pt
```

範例：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 2048 \
  --max_iterations 2000 \
  --headless \
  --device cuda:0 \
  --resume \
  --checkpoint logs/rsl_rl/redrhex_wheg/2026-04-25_12-00-00_wheg_locomotion/model_1000.pt
```

## 7. 開始 play

### 7.1 GUI 播放單一模型

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 1 \
  --device cuda:0 \
  --checkpoint logs/rsl_rl/redrhex_wheg/<run_name>/model_<iteration>.pt
```

### 7.2 播放最新 run 的最新 checkpoint

```bash
RUN=$(ls -td logs/rsl_rl/redrhex_wheg/* | head -1)
CKPT=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
echo "$CKPT"

$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 1 \
  --device cuda:0 \
  --checkpoint "$CKPT"
```

`play.py` 會自動把 policy 匯出到 checkpoint 同一個 run 底下：

```text
logs/rsl_rl/redrhex_wheg/<run_name>/exported/policy.pt
logs/rsl_rl/redrhex_wheg/<run_name>/exported/policy.onnx
```

### 7.3 Headless 錄影播放

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 1 \
  --headless \
  --video \
  --video_length 600 \
  --device cuda:0 \
  --checkpoint logs/rsl_rl/redrhex_wheg/<run_name>/model_<iteration>.pt
```

影片會輸出到：

```text
logs/rsl_rl/redrhex_wheg/<run_name>/videos/play/
```

## 8. TensorBoard

```bash
cd /home/lab_user1/Py/RedRHex
tensorboard --logdir logs/rsl_rl --port 6006
```

瀏覽器打開：

```text
http://localhost:6006
```

如果是遠端機器，請用 SSH port forwarding 或遠端桌面開瀏覽器。

## 8.1 Training Panel

這個 branch 有一個本機 training panel，可以用表單啟動 train、查看 reward/tweakable files、整理 training history notes，並一鍵啟動 TensorBoard / play checkpoint。

本機使用：

```bash
cd /home/lab_user1/Py/RedRHex
source /home/lab_user1/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab_bin
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

瀏覽器打開：

```text
http://127.0.0.1:8080
```

LAN 使用：

```bash
python -m tools.training_panel --host 0.0.0.0 --port 8080
```

SSH tunnel 使用：

```bash
ssh -L 8080:127.0.0.1:8080 user@host
```

V1 是 read-only reward/config browser：只顯示檔案、說明、reward scale index，不直接改 source code。

## 9. 目前這台機器已驗證到哪裡

已完成：

- `env_isaaclab_bin` 可 import Isaac Lab / Isaac Sim / Torch CUDA。
- `python -m pip install -e source/RedRhex` 已完成。
- `git-lfs` 已裝在 `env_isaaclab_bin`。
- `RedRhex.usd` 已從 Git LFS 拉成真 USD 檔。
- `test_7_description/` 已放進 repo，並已用 Isaac Lab URDF converter 產生 repo-local `test_7_description/test_7_description/urdf/test_7/test_7.usd`。
- `RedRhex.usd` 目前使用 repo-relative reference，不再依賴 `/home/lab_user1/Py/Downloads/b/...`。
- `scripts/list_envs.py` 可列出 `Template-Redrhex-Direct-v0`。
- `scripts/rsl_rl/train.py` 已修掉 `omni.log.warn` 在 Isaac Sim 5.1 下的相容性問題。
- 2026-05-14 已驗證第 5.1 節 smoke test 可完成：4 envs、1 iteration、headless、`cuda:0`。

注意：目前 repo-local `test_7_description` URDF expose 9 個 active joints，因此 RL config 已調整成 3 main drive + 3 ABAD + 3 damper：action space = 6，observation space = 35。若之後換回完整 6 腿/18 joint USD，需要同步恢復 joint mapping、action space、observation space。

## 10. 常見錯誤

### `ModuleNotFoundError: No module named 'isaaclab'`

通常是還在 `base` 或錯 env。重做：

```bash
source /home/lab_user1/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab_bin
cd /home/lab_user1/Py/RedRHex
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py --help
```

### `ModuleNotFoundError: No module named 'pxr'`

不要直接用 `python` 跑 train/play。改用：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py ...
```

### `RedRhex.usd` 打不開或只有 130 bytes

Git LFS 還沒拉：

```bash
conda activate env_isaaclab_bin
cd /home/lab_user1/Py/RedRHex
git lfs pull
```

### `tabs: terminal type 'dumb' cannot reset tabs`

```bash
export TERM=xterm
```

### CUDA out of memory

把 `--num_envs` 降低，例如：

```bash
--num_envs 2048
--num_envs 1024
--num_envs 512
```

## 11. 最短版每日流程

```bash
export REDRHEX_ROOT=/home/lab_user1/Py/RedRHex
export ISAACLAB_ROOT=/home/lab_user1/isaac_lab_ws/IsaacLab
source /home/lab_user1/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab_bin
export TERM=xterm
cd "$REDRHEX_ROOT"

$ISAACLAB_ROOT/isaaclab.sh -p -u scripts/list_envs.py

$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 2048 \
  --max_iterations 2000 \
  --headless \
  --device cuda:0
```
