# RedRhex ForwardFast 專題報告（教授版）

日期：2026-03-16  
專案：RedRhex Sim2Real 現場快速迭代  
版本：ForwardFast v2.1（回歸 Stage1 成功邏輯 + anti-collapse 保留）

---

## 0) 報告摘要（先講結論）

本次專題的核心不是「把曲線做得好看」，而是建立一條可以在實驗室現場高頻迭代、快速重訓、快速上機驗證的 Sim2Real 流程。

今天完成了兩個階段：
1. 建立 `ForwardFast`（forward-only）快速訓練管線，解決每次重訓等待太久的問題。  
2. 針對「六腳同轉、趴地死亡循環」做 anti-collapse 修正，並在第二輪回到 Stage1 成功邏輯，避免策略過度保守。

一句話總結：
我們把「大而慢的一次性訓練」轉成「可在現場反覆小步快跑的工程流程」，且已補上品質保證機制。

---

## 1) 起點：為什麼要做這件事

我們在實驗室要做的是 Sim2Real，不是只在模擬器內得高分。

實際工作流程是：
1. 在 Isaac Lab 訓練 policy。  
2. 匯出 `ONNX` 丟到實機。  
3. 觀察實機與模擬差異。  
4. 回到模擬調整 reward/物理參數。  
5. 重新訓練，再上機。  

這是一個「反覆調參 + 反覆驗證」的閉環流程。

最大瓶頸：
- 每次重訓都很久。  
- 現場調參時，一次錯誤就要等很久才能驗證下一版。  
- 原本同時訓練直走、橫走、斜走、旋轉，對今天「只要先打通直走」這個目標來說負擔過重。

所以今天的出發點是：
先把直走單獨做成「快速收斂 + 快速上機」版本，讓現場迭代可行。

---

## 2) 目標定義（今天要達成什麼）

今天目標分三層：
1. 訓練速度：建立 forward-only 快速收斂流程。  
2. 訓練品質：不是只看 reward 上升，而是要能走、能站、少倒地。  
3. 工程流程：形成可重複 SOP（Train -> TensorBoard -> Play/ONNX -> 上機 -> 回調）。

具體要求：
- 現場調參後可快速重訓。  
- TensorBoard 早期快速上升，後期快速平穩。  
- 上機行為與模擬目標一致，不是趴地抖腿式假收斂。

---

## 3) 方法總覽（策略怎麼定）

我們採用「問題降維 + 品質守門」策略。

第一步：問題降維
- 從多技能任務降維成 forward-only。  
- 命令分布縮窄到直走區間。  
- reward 聚焦直走核心 KPI。

第二步：品質守門
- 避免不健康姿態也能拿分。  
- 提早終止低價值倒地軌跡。  
- 調整 PPO 探索強度，避免在壞策略附近震盪。

---

## 4) 今天實際做了什麼（從 0 到 1 再到 2，再到 2.1）

## 4.1 階段 A：先建立 ForwardFast 管線（v1）

新增一條平行路徑，不破壞原本五階段訓練：
1. 新 task：`Template-Redrhex-ForwardFast-Direct-v0`  
2. 新環境配置：`RedrhexForwardFastEnvCfg`  
3. 新 PPO 配置：`PPORunnerForwardFastCfg`  
4. 新手冊：把 train / play / tensorboard 指令補齊

這一步解決的是「速度」問題。

## 4.2 階段 B：現場回報品質問題（假收斂）

回報症狀：
1. 曲線看起來收斂快。  
2. 但實機或播放會出現六腳同步轉動。  
3. 機身觸地後反覆死亡重置。  

判讀：
這是「會拿分但不會走」的假收斂，不符合部署需求。

## 4.3 階段 C：anti-collapse 品質修正（v2）

修正原則不是單加重懲罰，而是分三層處理：
1. 控制器層：先讓 tripod/duty-cycle 的可行空間正確。  
2. 獎勵層：不健康姿態不給主要正向分。  
3. 優化層：降低探索躁動，提升訓練穩定性。

## 4.4 階段 D：第二輪回調（v2.1，參考原 Stage1）

v2 雖然抑制了崩潰，但現場觀察到「太保守、起步不動」問題。  
因此第二輪採用「回到原 Stage1 成功參數骨架，再保留 forward-only 加速」策略：
1. 控制器 stance 判定回到角度相位窗（與原 Stage1 一致）。  
2. 放寬過嚴終止條件，避免過早重置。  
3. 恢復探索（提高 `init_noise_std` 與 `entropy_coef`）。  
4. 將暖機步數從 120 降回 30，避免 episode 大部分時間都在低動作幅度。

---

## 5) 程式碼修改明細（What + Why）

## 5.1 任務註冊層

檔案：`source/RedRhex/RedRhex/tasks/direct/redrhex/__init__.py`

修改內容：
1. 註冊 `Template-Redrhex-ForwardFast-Direct-v0`。  
2. 綁定 `RedrhexForwardFastEnvCfg` 與 `PPORunnerForwardFastCfg`。

原因：
- 讓現場直接切 task 就能進入快訓練模式。  
- 不影響舊任務 `Template-Redrhex-Direct-v0`。

## 5.2 環境配置層（ForwardFast）

檔案：`source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`  
類別：`RedrhexForwardFastEnvCfg`

最終版關鍵參數（v2.1）：
1. 訓練結構
- `episode_length_s = 30`
- `stage = 1`
- `curriculum_auto_progress = False`

2. 命令分布（只保留直走）
- `lin_vel_x_range = [0.22, 0.42]`
- `lin_vel_y_range = [0.0, 0.0]`
- `ang_vel_z_range = [0.0, 0.0]`
- `command_resample_on_timer = False`
- `command_resample_time = 6.0`

3. 控制設定（回到 Stage1 成功風格）
- `main_drive_vel_scale = 8.0`
- `forward_phase_lock_gain = 1.2`
- `main_drive_residual_scale = 0.22`
- `stage_forward_policy_drive_residual_scale = [0.10]`
- `stage_action_warmup_steps = [30]`

4. 終止條件（回到 Stage1 較穩定區間）
- `max_tilt_magnitude = 1.55`
- `stage_max_tilt_magnitude = [1.82]`
- `stage1_min_base_height = 0.03`
- `stage1_body_contact_height_threshold = 0.06`
- `stage_body_contact_tilt_threshold = [1.80]`
- `stage_termination_grace_steps = [120]`

5. reward gate（保留功能，但 v2.1 預設關閉）
- `gate_positive_rewards_when_unhealthy = False`
- `reward_gate_min_base_height = 0.105`
- `reward_gate_max_body_tilt = 0.70`

6. reward 方向（v2.1）
- `forward_progress = 5.5`
- `velocity_tracking = 4.5`
- `forward_prior_coherence = 1.2`
- `forward_prior_antiphase = 1.2`
- `target_base_height = 0.12`
- `height_low_penalty = 1.2`
- `leg_moving = 0.35`
- `fall = -8.0`

原因總結：
- 讓策略更快學到「穩定直走」，不是「快速亂轉」。

## 5.3 環境控制邏輯層（核心）

檔案：`source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py`

修改一（v2 試驗）：forward bias 曾改為 duty-time stance 判定
- 由 `desired_cycle < duty_target` 決定是否 stance。

修改一最終（v2.1）：回到原 Stage1 角度相位窗
- `desired_in_stance = self._in_stance_phase(desired_phase)`

最終目的：
- 和原本已成功的 Stage1 控制機制一致，優先保證「先走起來」。

修改二：reward 健康門控（healthy_gate，功能保留）
- 用 `base_height` + `body_tilt` 判斷姿態健康。  
- 不健康時關閉 `rew_forward / rew_tracking / rew_forward_gait / rew_leg_moving` 等主要正向項。

目前策略：
- 程式功能保留，但 ForwardFast v2.1 預設關閉，避免過早壓制探索動作。

## 5.4 PPO 層

檔案：`source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_ppo_cfg.py`  
類別：`PPORunnerForwardFastCfg`

最終版參數（v2.1）：
- `max_iterations = 1500`
- `num_steps_per_env = 24`
- `init_noise_std = 0.55`
- `learning_rate = 5.0e-4`
- `entropy_coef = 0.0035`
- `desired_kl = 0.01`

原因：
- 恢復必要探索，避免策略卡在「低動作、不前進」區域。  
- 保留快速收斂速度，但先確保 locomotion 活性。

## 5.5 Play/checkpoint 工具相容修正

檔案：`scripts/rsl_rl/play.py`

修改內容：
1. 支援 `--load_run <run> --checkpoint model_xxxx.pt` 自動拼接完整路徑。  
2. 若誤傳非訓練 checkpoint，會自動 fallback 到 `model_*.pt`。

解決問題：
- `FileNotFoundError: Unable to find the file: model_1199.pt`  
- 把 `events.out.tfevents...` 誤當 checkpoint

---

## 6) 現場流程 SOP（How）

## 6.1 訓練（ForwardFast）

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-ForwardFast-Direct-v0 \
  --headless \
  --num_envs=2048 \
  --max_iterations=1500 \
  --run_name=forward_fast_stage1ref_v1
```

## 6.2 TensorBoard 監看

```bash
tensorboard --logdir . --port 6006 --bind_all
```

## 6.3 播放與匯出 ONNX

```bash
RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_forward_fast/* | head -1)")
CKPT=$(ls -v logs/rsl_rl/redrhex_forward_fast/$RUN/model_*.pt | tail -1)

python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-ForwardFast-Direct-v0 \
  --num_envs=64 \
  --disable_keyboard_control \
  --load_run="$RUN" \
  --checkpoint="$CKPT"
```

---

## 7) 如何判斷「真的成功」而不是假收斂

必要指標（需同時滿足）：
1. `Train/mean_episode_length`：上升且穩定。  
2. `Episode_Termination/terminated`：下降。  
3. `Episode_Reward/rew_fall`：負值幅度收斂。  
4. `Episode_Reward/diag_base_height`：維持健康高度。  
5. `Episode_Reward/rew_forward_gait`、`diag_forward_duty_ema`：tripod 節奏穩定。

判讀原則：
- 只看 `mean_reward` 不夠。  
- 如果 reward 上升但終止率高，仍不能上機。

---

## 8) 今天遇到的問題與對應修正

問題 1：快收斂但不會走（六腳同轉、趴地）
- 第一輪修正：duty-time stance + 健康姿態 gate + 嚴終止。  
- 第二輪修正（最終）：回到 Stage1 stance 控制 + 放寬終止 + 提高探索。

問題 2：checkpoint 路徑常誤傳
- 修正：`play.py` 自動補路徑與 fallback。

問題 3：現場重訓速度不足
- 修正：forward-only 管線 + 集中 reward + 穩定化 PPO。

---

## 9) 目前成果、限制與下一步

已完成：
1. ForwardFast 任務與配置全落地。  
2. anti-collapse 品質修正落地，且完成第二輪 Stage1-reference 回調。  
3. 文件與 SOP 同步完成。  

限制：
1. ForwardFast 是現場快迭代工具，不是多技能最終部署模型。  
2. 仍需根據實機回饋持續調整。

下一步：
1. 白天用 ForwardFast 快速校參。  
2. 夜間回到五階段做整合訓練。  
3. 建立固定報告格式（參數版本、TB 曲線、上機影片、結論）。

---

## 10) 給教授的結論句

本專題已把 Sim2Real 從「慢速大訓練」轉為「可現場快速重訓、快速驗證、可控品質」的工程流程，並且針對假收斂問題完成可重現的技術修正。
