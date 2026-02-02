# RedRhex 訓練指令大全

## 專案根目錄
```
/home/jasonliao/RedRhex/RedRhex
```

## 目錄
- [重要檔案位置](#重要檔案位置)
- [環境設置](#環境設置)
- [訓練指令](#訓練指令)
- [播放/測試指令](#播放測試指令)
- [TensorBoard 監控](#tensorboard-監控)
- [Git 版本控制](#git-版本控制)
- [常用參數說明](#常用參數說明)
- [完整範例](#完整範例)

---

## 重要檔案位置

### 核心程式碼
| 檔案 | 路徑 | 說明 |
|------|------|------|
| 環境定義 | `/home/jasonliao/RedRhex/RedRhex/source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py` | 獎勵函數、動作應用、觀測 |
| 環境配置 | `/home/jasonliao/RedRhex/RedRhex/source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py` | 超參數、機器人配置 |
| 訓練腳本 | `/home/jasonliao/RedRhex/RedRhex/scripts/rsl_rl/train.py` | 啟動訓練 |
| 播放腳本 | `/home/jasonliao/RedRhex/RedRhex/scripts/rsl_rl/play.py` | 測試模型 |
| CLI 參數 | `/home/jasonliao/RedRhex/RedRhex/scripts/rsl_rl/cli_args.py` | 命令行參數定義 |

### 訓練輸出
| 檔案類型 | 路徑 | 說明 |
|----------|------|------|
| 訓練日誌目錄 | `/home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/` | 所有訓練記錄 |
| 模型檔案 | `/home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/redrhex_*/YYYY-MM-DD_HH-MM-SS_*/model_*.pt` | 保存的模型權重 |
| TensorBoard 日誌 | `/home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/redrhex_*/YYYY-MM-DD_HH-MM-SS_*/events.*` | 訓練曲線數據 |
| Hydra 輸出 | `/home/jasonliao/RedRhex/RedRhex/outputs/YYYY-MM-DD/HH-MM-SS/` | 配置快照 |

### 機器人資源
| 檔案 | 路徑 | 說明 |
|------|------|------|
| URDF/USD 模型 | `/home/jasonliao/RedRhex/RedRhex/source/RedRhex/RedRhex/assets/` | 機器人 3D 模型 |
| 機器人配置 | `/home/jasonliao/RedRhex/RedRhex/source/RedRhex/config/` | 機器人參數配置 |

### 文檔
| 檔案 | 路徑 | 說明 |
|------|------|------|
| 本指令文檔 | `/home/jasonliao/RedRhex/RedRhex/docs/COMMANDS.md` | 訓練指令大全 |
| 專案說明 | `/home/jasonliao/RedRhex/RedRhex/README.md` | 專案介紹 |
| RedRhex 說明 | `/home/jasonliao/RedRhex/RedRhex/README_REDRHEX.md` | RedRhex 專案說明 |

---

## 環境設置

### 啟動 Conda 虛擬環境
```bash
conda activate env_isaaclab
```

### 退出虛擬環境
```bash
conda deactivate
```

### 查看已安裝的環境
```bash
conda env list
```

### 安裝專案依賴（首次設置）
```bash
cd /home/jasonliao/RedRhex/RedRhex
pip install -e source/RedRhex
```

---

## 訓練指令

### 基本訓練指令
```bash
python scripts/rsl_rl/train.py --task=Template-Redrhex-Direct-v0
```

### 常用訓練參數組合

#### 無頭模式訓練（推薦，速度最快）
```bash
python scripts/rsl_rl/train.py --task=Template-Redrhex-Direct-v0 --num_envs=4096 --max_iterations=5000 --headless
```

#### 有 GUI 訓練（可視化訓練過程）
```bash
python scripts/rsl_rl/train.py --task=Template-Redrhex-Direct-v0 --num_envs=256
```

#### 少量環境快速測試
```bash
python scripts/rsl_rl/train.py --task=Template-Redrhex-Direct-v0 --num_envs=64 --max_iterations=100
```

#### 從 checkpoint 繼續訓練
```bash
python scripts/rsl_rl/train.py --task=Template-Redrhex-Direct-v0 --num_envs=4096 --max_iterations=10000 --headless --resume --checkpoint=/path/to/model.pt
```

---

## 播放/測試指令

### 基本播放指令
```bash
python scripts/rsl_rl/play.py --task=Template-Redrhex-Direct-v0 --checkpoint=/path/to/model.pt
```

### 單環境播放（推薦觀察）
```bash
python scripts/rsl_rl/play.py --task=Template-Redrhex-Direct-v0 --num_envs=1 --checkpoint=/path/to/model.pt
```

### 播放最新訓練的模型
```bash
# 找到最新的 log 目錄
ls -lt /home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/redrhex_*/

# 播放該目錄中的模型（範例）
python /home/jasonliao/RedRhex/RedRhex/scripts/rsl_rl/play.py \
    --task=Template-Redrhex-Direct-v0 \
    --num_envs=1 \
    --checkpoint=/home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/redrhex_wheg/2026-02-02_17-46-42_wheg_locomotion/model_3800.pt
```

### 多環境播放
```bash
python scripts/rsl_rl/play.py --task=Template-Redrhex-Direct-v0 --num_envs=16 --checkpoint=/path/to/model.pt
```

---

## TensorBoard 監控

### 啟動 TensorBoard
```bash
# 監控特定訓練
tensorboard --logdir=/home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/redrhex_wheg/2026-02-02_17-46-42_wheg_locomotion

# 監控所有訓練
tensorboard --logdir=/home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/

# 指定端口
tensorboard --logdir=/home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/ --port=6006
```

### 在瀏覽器中打開
```
http://localhost:6006
```

---

## Git 版本控制

### 查看狀態
```bash
git status
git branch
git remote -v
```

### 創建新分支並推送
```bash
# 創建並切換到新分支
git checkout -b feature/your-feature-name

# 添加修改的文件
git add /home/jasonliao/RedRhex/RedRhex/source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py
git add /home/jasonliao/RedRhex/RedRhex/source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py

# 提交
git commit -m "feat: Your commit message"

# 推送到遠端
git push -u origin feature/your-feature-name
```

### GitHub 倉庫位置
```
https://github.com/JasonLiaoJCS/RedRHex.git
```

### 切換分支
```bash
git checkout main
git checkout feature/velocity-tracking-visualization
```

### 拉取最新代碼
```bash
git pull origin main
```

---

## 常用參數說明

### 訓練參數 (train.py)

| 參數 | 說明 | 預設值 | 範例 |
|------|------|--------|------|
| `--task` | 任務名稱 | 必填 | `Template-Redrhex-Direct-v0` |
| `--num_envs` | 並行環境數量 | 配置文件定義 | `4096` |
| `--max_iterations` | 最大訓練迭代次數 | 配置文件定義 | `5000` |
| `--headless` | 無頭模式（無 GUI） | False | 直接加上即可 |
| `--resume` | 從 checkpoint 繼續訓練 | False | 直接加上即可 |
| `--checkpoint` | checkpoint 文件路徑 | - | `/path/to/model.pt` |
| `--seed` | 隨機種子 | 配置文件定義 | `42` |

### 播放參數 (play.py)

| 參數 | 說明 | 預設值 | 範例 |
|------|------|--------|------|
| `--task` | 任務名稱 | 必填 | `Template-Redrhex-Direct-v0` |
| `--num_envs` | 環境數量 | 1 | `1` |
| `--checkpoint` | 模型文件路徑 | 必填 | `/path/to/model.pt` |

### 環境數量建議

| 用途 | 建議 num_envs | 說明 |
|------|---------------|------|
| 快速測試 | 64-256 | 驗證代碼正確性 |
| 正式訓練 | 2048-4096 | 最佳訓練效率 |
| 播放觀察 | 1-4 | 方便觀察單個機器人 |
| 批量評估 | 16-64 | 統計多個機器人表現 |

---

## 完整範例

### 範例 1：完整訓練流程
```bash
# 1. 啟動虛擬環境
conda activate env_isaaclab

# 2. 進入專案目錄
cd /home/jasonliao/RedRhex/RedRhex

# 3. 開始訓練（無頭模式）
python /home/jasonliao/RedRhex/RedRhex/scripts/rsl_rl/train.py \
    --task=Template-Redrhex-Direct-v0 \
    --num_envs=4096 \
    --max_iterations=5000 \
    --headless

# 4. 另開終端，啟動 TensorBoard 監控
tensorboard --logdir=/home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/

# 5. 訓練完成後播放測試（替換 YYYY-MM-DD_HH-MM-SS 為實際時間戳）
python /home/jasonliao/RedRhex/RedRhex/scripts/rsl_rl/play.py \
    --task=Template-Redrhex-Direct-v0 \
    --num_envs=1 \
    --checkpoint=/home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/redrhex_wheg/YYYY-MM-DD_HH-MM-SS_wheg_locomotion/model_5000.pt
```

### 範例 2：快速迭代測試
```bash
# 快速測試代碼修改是否有效（少量環境、少量迭代）
python /home/jasonliao/RedRhex/RedRhex/scripts/rsl_rl/train.py \
    --task=Template-Redrhex-Direct-v0 \
    --num_envs=256 \
    --max_iterations=100
```

### 範例 3：長時間訓練
```bash
# 使用 nohup 在背景運行（即使關閉終端也繼續）
cd /home/jasonliao/RedRhex/RedRhex
nohup python scripts/rsl_rl/train.py \
    --task=Template-Redrhex-Direct-v0 \
    --num_envs=4096 \
    --max_iterations=10000 \
    --headless > train.log 2>&1 &

# 查看訓練日誌
tail -f /home/jasonliao/RedRhex/RedRhex/train.log
```

### 範例 4：Git 工作流
```bash
cd /home/jasonliao/RedRhex/RedRhex

# 修改代碼後推送
git checkout -b feature/new-reward-function
git add source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py
git commit -m "feat: Add new reward function for lateral movement"
git push -u origin feature/new-reward-function
```

---

## 目錄結構

```
/home/jasonliao/RedRhex/RedRhex/
│
├── docs/
│   └── COMMANDS.md                    # 本指令文檔
│
├── logs/
│   └── rsl_rl/
│       └── redrhex_*/                 # 訓練日誌和模型
│           └── YYYY-MM-DD_HH-MM-SS_*/
│               ├── model_*.pt         # 保存的模型權重
│               └── events.*           # TensorBoard 日誌
│
├── outputs/
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/                  # Hydra 配置輸出
│
├── scripts/
│   └── rsl_rl/
│       ├── train.py                   # 訓練腳本
│       ├── play.py                    # 播放腳本
│       └── cli_args.py                # CLI 參數定義
│
└── source/
    └── RedRhex/
        ├── pyproject.toml             # 專案配置
        ├── setup.py                   # 安裝腳本
        ├── config/                    # 機器人配置
        └── RedRhex/
            ├── assets/                # 機器人 URDF/USD 模型
            └── tasks/
                └── direct/
                    └── redrhex/
                        ├── __init__.py
                        ├── redrhex_env.py      # ⭐ 環境定義（獎勵函數）
                        └── redrhex_env_cfg.py  # ⭐ 配置文件（超參數）
```

---

## 故障排除

### 常見問題

1. **CUDA out of memory**
   ```bash
   # 減少環境數量
   python scripts/rsl_rl/train.py --task=Template-Redrhex-Direct-v0 --num_envs=1024 --headless
   ```

2. **找不到模塊**
   ```bash
   # 重新安裝專案
   pip install -e source/RedRhex
   ```

3. **渲染問題**
   ```bash
   # 使用無頭模式
   python scripts/rsl_rl/train.py --task=Template-Redrhex-Direct-v0 --headless
   ```

4. **查看 GPU 使用情況**
   ```bash
   nvidia-smi
   watch -n 1 nvidia-smi  # 每秒更新
   ```

---

## 快速參考卡

```bash
# ============ 常用路徑 ============
# 專案根目錄
cd /home/jasonliao/RedRhex/RedRhex

# 環境定義（獎勵函數）
/home/jasonliao/RedRhex/RedRhex/source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py

# 環境配置（超參數）
/home/jasonliao/RedRhex/RedRhex/source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py

# 訓練日誌
/home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/

# ============ 常用指令 ============
# 虛擬環境
conda activate env_isaaclab

# 訓練
python /home/jasonliao/RedRhex/RedRhex/scripts/rsl_rl/train.py \
    --task=Template-Redrhex-Direct-v0 \
    --num_envs=4096 \
    --max_iterations=5000 \
    --headless

# 播放
python /home/jasonliao/RedRhex/RedRhex/scripts/rsl_rl/play.py \
    --task=Template-Redrhex-Direct-v0 \
    --num_envs=1 \
    --checkpoint=<MODEL_PATH>

# TensorBoard
tensorboard --logdir=/home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/

# 查看最新訓練目錄
ls -lt /home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/redrhex_*/
```
