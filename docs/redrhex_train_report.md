# RedRhex 強化學習訓練與獎勵系統改進報告

## 1. 專案背景
RedRhex 是一款六足機器人，採用 RHex-style whegs（輪腿），目標是透過強化學習（RL）訓練其自主移動能力。訓練環境基於 Isaac Lab/Sim，使用 DirectRLEnv 與 PPO 演算法，並行 4096 個環境。

## 2. 問題起點與成因
### 2.1 初始問題
- **錯誤回報**：訓練過程中出現 `UnboundLocalError: cannot access local variable 'abad_right_mean'`。
- **行為異常**：機器人訓練後極度被動，幾乎不移動，尤其在橫向/前進方向。
- **獎勵系統**：原始獎勵函數過於複雜，懲罰項過多，導致學習困難與 reward hacking（獎勵作弊）。

### 2.2 問題成因分析
- **ABAD 計算時機錯誤**：導致變數未初始化。
- **獎勵項目衝突**：多重懲罰導致 agent 採取極端保守策略。
- **獎勵分散**：獎勵分散於多處，難以追蹤與調整。

## 3. 參考資料與最佳實踐
- 參考頂尖大學 RL 機器人論文（如 CMU, MIT, Stanford）
- 研究 reward shaping、reward hacking 問題
- TensorBoard reward trend 分析

## 4. 主要修正流程與邏輯
### 4.1 Bug 修正
- **ABAD 計算提前**：將 ABAD 相關計算移至 reward 函數前段，確保變數初始化。
- **程式檔案**：`redrhex_env.py`

### 4.2 行為優化：橫向移動兩階段
- **需求**：橫向移動時，先讓腿回到初始位置，再鎖定主驅動關節。
- **實作**：
  - 新增兩階段 lateral movement：Preparation（P 控制器回初始）→ Execution（鎖定主驅動，ABAD 步進）
  - **程式檔案**：`redrhex_env.py`

### 4.3 獎勵系統大幅重構
- **目標**：簡化獎勵、提升 velocity tracking、減少懲罰、增加腿部動作獎勵、集中權重、方便 ablation study。
- **步驟**：
  1. **移除/關閉多餘懲罰項**：如能耗、碰撞、過度扭力等。
  2. **提升 velocity tracking reward**：鼓勵機器人主動移動。
  3. **新增腿部動作獎勵**：鼓勵 whegs 積極運動。
  4. **獎勵權重集中管理**：所有權重移至 config，方便調整。
  5. **獎勵項目 logging**：每項獎勵都記錄到 TensorBoard，方便分析。
  6. **程式檔案**：
     - `redrhex_env.py`：重構 `_get_rewards()`，移除多餘項目，新增 logging
     - `redrhex_env_cfg.py`：集中獎勵權重，新增腿部動作獎勵

### 4.4 訓練驗證與結果
- **訓練腳本**：PPO 設定不變，環境改為新 reward。
- **TensorBoard 分析**：
  - mean reward 由負轉正
  - 機器人行為更積極，移動明顯
  - 各獎勵項目趨勢清楚可追蹤

## 5. 舊程式與新程式差異
| 項目 | 舊程式 | 新程式 |
|------|--------|--------|
| ABAD 計算 | reward 內部，易出錯 | reward 前段，穩定 |
| 橫向移動 | 單階段，容易卡死 | 兩階段，動作流暢 |
| 獎勵項目 | 多懲罰，分散，難調 | 精簡，集中，易調整 |
| 權重管理 | 分散各處 | 集中 config |
| logging | 部分項目 | 全部項目 |
| 行為表現 | 被動，reward hacking | 積極，學習明顯 |

## 6. 為什麼要這麼做？
- **解決 bug**：避免訓練中斷
- **提升學習效率**：減少 reward hacking，讓 agent 學到正確行為
- **方便 ablation study**：可系統性關閉/調整獎勵項目，分析影響
- **提升團隊協作**：程式結構清楚，logging 完整，方便溝通

## 7. 具體成果與好處
- 訓練初始化成功，mean reward 由負轉正
- 機器人行為更積極，移動明顯
- reward 項目可追蹤，方便 debug 與 ablation
- 程式結構更清楚，團隊易於維護

## 8. 未來建議
- 進行 ablation study：系統性關閉各獎勵項目，分析行為影響
- 持續優化 reward，參考最新 RL 論文
- 強化 logging 與可視化，提升 debug 能力

## 9. 參考資料
- CMU, MIT, Stanford RL 機器人論文
- RL reward shaping 相關研究
- TensorBoard reward trend 分析

---

> 本報告詳細記錄了 RedRhex RL 訓練流程、程式修正、獎勵系統重構、行為優化與訓練結果，供團隊成員參考。

---

## 10. 最重要執行指令（Train / TensorBoard / Play）

若你要看「從啟動虛擬環境開始，到選 Task、選 Agent、決定要不要 teacher/student、怎麼跑五階段 curriculum、怎麼 play 與 export」的完整新手版流程，請直接看：

- `docs/redrhex_train_play_guide.md`

本報告這一節只保留最精簡的 Train / TensorBoard / Play 範例。

## 10.0 目前建議的訓練決策流程

這裡先把一件最容易誤會的事情講清楚：

- 直接跑 `python scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0 ...`
- 不等於五階段 curriculum

目前主 task 的預設是單段 mixed stage 訓練，所以：

- `train.py` 直接跑主 task = 單段 Stage5 mixed training
- `train_stage_pipeline.sh` = 真正的五階段 curriculum training

因此現在的建議順序是：

1. **第一次上手 / 先把系統跑通**
- `Task`：`Template-Redrhex-Direct-v0`
- `Agent`：`rsl_rl_cfg_entry_point`
- 直接跑 `train.py`

2. **想快速看直走結果**
- `Task`：`Template-Redrhex-ForwardFast-Direct-v0`
- `Agent`：`rsl_rl_cfg_entry_point`
- 直接跑 `train.py`

3. **只想穩定往前走，但不要 ForwardFast**
- `Task`：`Template-Redrhex-Direct-v0`
- 額外加 Hydra override：`env.stage=1`
- `Agent`：
  - 一般 PPO：`rsl_rl_cfg_entry_point`
  - teacher：`rsl_rl_teacher_cfg_entry_point`
  - student distillation：`rsl_rl_distillation_cfg_entry_point`
- 用途：保留主 task 的正式 reward / observation / teacher-student 堆疊，但只訓 forward-only

4. **想做最正式的完整 locomotion 主線**
- `Task`：`Template-Redrhex-Direct-v0`
- `Agent`：`rsl_rl_cfg_entry_point`
- 使用 `train_stage_pipeline.sh`

5. **想追求更高上限或做研究**
- 先有穩定的一般 PPO baseline
- 再 train teacher
- 最後再做 student distillation

一句話總結：

- 新手第一步不是 teacher / student
- 最常用主線是 `rsl_rl_cfg_entry_point`
- 最正式完整路線仍然是五階段 curriculum

### 10.0.1 補充：正式 forward-only（非 ForwardFast）怎麼做

如果你的目標不是「越快收斂越好」，而是：

- 不想訓練 lateral / diagonal / yaw
- 只想把直走訓到更穩
- 想保留主 task 的完整 reward 與觀測堆疊
- 想繼續用 teacher / student

那最適合的不是 `ForwardFast`，而是：

- `Task`：`Template-Redrhex-Direct-v0`
- Hydra override：`env.stage=1`

這條路線的意思是：

- 仍然使用主 task 的正式架構
- 只是把 curriculum 固定在 Stage1 forward-only
- 主 task 的節能 reward 仍然生效
- teacher / distillation 入口也都能直接沿用

最常用三個指令如下。

**A. forward-only baseline PPO**

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab

python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name forward_stage1_student \
  env.stage=1
```

**B. forward-only privileged teacher**

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab

python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name forward_stage1_teacher \
  env.stage=1
```

**C. forward-only student distillation**

先找 teacher checkpoint：

```bash
TEACHER_RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg_teacher/* | head -1)")
TEACHER_CKPT=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg_teacher/$TEACHER_RUN/model_*.pt | tail -1)")
echo "TEACHER_RUN=$TEACHER_RUN"
echo "TEACHER_CKPT=$TEACHER_CKPT"
```

再建立 distill 讀取 teacher checkpoint 的連結：

```bash
mkdir -p logs/rsl_rl/redrhex_wheg_distill
ln -s ../redrhex_wheg_teacher/$TEACHER_RUN logs/rsl_rl/redrhex_wheg_distill/$TEACHER_RUN
```

最後跑 distillation：

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
  --run_name forward_stage1_distill \
  env.stage=1
```

### 10.0.2 補充：為什麼訓練畫面看起來像被彈飛

這也是目前最常被誤會的現象之一。

如果你直接跑主 task：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500
```

那你目前跑到的其實是：

- 單段 Stage5 mixed training
- rough terrain
- domain randomization
- terrain curriculum
- push randomization
- fault injection

所以早期畫面常常會：

- 往不同方向衝
- 翻滾
- 看起來像被彈出去

這不一定表示訓練壞掉。  
很多時候只是因為你現在看的不是 forward-only，而是高難度 mixed-skill 模式。

另外，這個環境的 policy 輸出是 residual action，不是完全從零開始。  
環境內部還會疊加 command-conditioned gait / drive bias，所以即使 policy 還沒學好，機器人也已經會有明顯動作。

因此這個畫面更正確的解讀通常是：

- 早期控制很醜
- mixed task 很兇
- 但不一定是 PhysX 爆炸

如果你要快速判斷是不是模式太激進，最簡單的 sanity check 是：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 256 \
  --max_iterations 100 \
  --run_name debug_stage1_forward \
  env.stage=1
```

如果這樣就正常很多，通常代表你原本看到的主要是 Stage5 mixed task 的可視化副作用，而不是核心訓練流程壞掉。

### 10.1 Train（ForwardFast）

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab

python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-ForwardFast-Direct-v0 \
  --headless \
  --num_envs=2048 \
  --max_iterations=1200 \
  --run_name=forward_fast_trial_a
```

### 10.2 TensorBoard

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab

tensorboard --logdir . --port 6006 --bind_all
```

瀏覽器打開：
- `http://localhost:6006`

### 10.3 Play（讀取最新 checkpoint）

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
