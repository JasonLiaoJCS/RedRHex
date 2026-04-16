# RedRHex 2026 期中報告

**日期**: 2026-04-16  
**主題**: 依據 `docs/redrhex_improvement_strategy_full.md` 與最新可查文獻，對 RedRHex 強化學習訓練堆疊進行大幅重構  
**本次重點**: 我不只做了文獻整理，也直接改了程式，並且用 smoke validation 把 `環境 -> PPO -> privileged teacher -> student distillation` 這條訓練鏈跑通。

---

## 1. 這次我做了什麼

這次工作不是單純調 reward，而是把 RedRHex 的訓練架構往近五年腿足機器人 SOTA 的方向重構。核心改革分成七件事：

1. 把地形從單一平地改成 **rough / wave / stairs / boxes 混合 terrain generator + curriculum**。  
2. 把主驅動與 ABAD 致動器從過度理想化的設定，改成 **更接近真機的 DCMotor 飽和模型**。  
3. 把 observation 架構從單幀 proprio，擴成 **當前 policy obs + 短歷史 history + privileged critic + teacher obs**。  
4. 把 domain randomization 從 env 級別平均擾動，升級成 **腿級別 actuator randomization + 單腿故障注入**。  
5. 加入 **左右對稱資料增強**，讓六足形態的對稱性真正進入 PPO 訓練。  
6. 補上 **privileged teacher PPO config + distillation config**，讓 Miki / RMA / DreamWaQ / CTS 這條 teacher-student 路線在 RedRHex 變成可執行流程。  
7. 新增 **`validate_reform_stack.py`** 與 **`validate_distillation_stack.py`**，建立修改後的閉環驗證流程。

本次不是只停留在想法。我已完成以下驗證：

- 環境 reset / step 正常。
- 新 observation group 維度正確。
- terrain generator 真正啟用。
- 腿級 fault injection 真正啟用。
- PPO runner smoke test 成功。
- privileged teacher PPO smoke test 成功。
- student distillation smoke test 成功。

我必須誠實說明：  
這次已完成的是 **訓練堆疊重構與工程驗證**，不是完整長時間收斂 benchmark。也就是說，我已經把會影響 sim-to-real 與魯棒性的關鍵基礎設施接好並驗證它能跑，但「最終性能提升多少」仍需要正式長訓練與對照實驗來量化。

---

## 2. 我怎麼從文獻推導出這次改革

下面這些文獻與專案頁，都是我根據 `docs/redrhex_improvement_strategy_full.md` 再向外查證後，整理成對 RedRHex 最有價值的工程結論。  
我會明確區分：

- **文獻本身真的做了什麼**
- **我對 RedRHex 的工程推論**
- **我這次已經實作了哪些部分**

### 2.1 ETH RSL — Learning Robust Perceptive Locomotion (Miki et al., 2022)

**來源**

- Science Robotics DOI: [10.1126/scirobotics.abk2822](https://www.science.org/doi/10.1126/scirobotics.abk2822)
- Project page: [https://leggedrobotics.github.io/rl-perceptiveloco/](https://leggedrobotics.github.io/rl-perceptiveloco/)
- ETH News: [How robots learn to hike](https://ethz.ch/en/news-and-events/eth-news/news/2022/01/how-robots-learn-to-hike.html)

**動機**

這篇工作的核心痛點是：  
只靠 proprioception 的 locomotion 很穩，但面對野外複雜地形太保守；只靠深度感測的 perception locomotion 又會受遮蔽、漂移、霧氣、雪地與非剛性地面的影響。

**理論背景**

本質上是 POMDP 問題。機器人對真實環境的觀測永遠不完整，所以需要把短時間內的 proprioception 與 exteroception 整合成一個 latent belief。Miki 這篇的重要觀念是：  
**訓練時給 teacher 更多資訊，部署時讓 student 靠不完整且含噪的觀測近似 teacher。**

**他們怎麼做**

- 先訓 privileged teacher policy。
- 再把 student 蒸餾到 noisy elevation map + proprio。
- 在 student 階段主動加入 sensor failure、drift、遮蔽與 domain randomization。

**結果**

這篇最有名的是 ANYmal 的野外長距離徒步與 DARPA SubT 場景表現。它證明了 teacher-student 不是 paper trick，而是真的能把訓練期的 privileged information 轉化成部署期的強健控制策略。

**對 RedRHex 的啟發**

我對這篇的工程推論是：  
RedRHex 目前沒有完整深度相機與 elevation map pipeline，所以最值得抄的不是感知模組本身，而是 **teacher-student 訓練哲學**。

**我這次已落地的部分**

- 新增 `teacher` observation group，讓 privileged teacher 有獨立輸入。
- 新增 distillation config，讓後續可以真的跑兩階段 teacher-student。
- 加入 observation noise 與 latency randomization，模擬 student 階段的感測退化。

---

### 2.2 ETH RSL — ANYmal Parkour (Hoeller et al., 2024)

**來源**

- Science Robotics DOI: [10.1126/scirobotics.adi7566](https://www.science.org/doi/10.1126/scirobotics.adi7566)
- ETH News: [ANYmal can do parkour and walk across rubble](https://ethz.ch/en/news-and-events/eth-news/news/2024/03/anymal-can-do-parkour-and-walk-across-rubble.html)
- arXiv preprint: [https://arxiv.org/abs/2306.14874](https://arxiv.org/abs/2306.14874)

**動機**

單一 locomotion policy 很難同時學會走、跳、爬、跨越與蹲伏。不同技能的 reward 與控制結構互相衝突，容易 mode collapse。

**理論背景**

這篇代表的是 **hierarchical RL + multi-skill decomposition**。  
上層做 navigation / skill selection，下層做 skill-specific locomotion。

**他們怎麼做**

- 先分技能各自訓練。
- 再做 distillation。
- 再用高層 policy 決定什麼時候切換技能。

**對 RedRHex 的啟發**

這篇不是要我現在立刻做完整 MoE，而是提醒我：  
**地形多樣性、課程設計、技能分層，都是為了避免 policy 被單一平地目標綁死。**

**我這次已落地的部分**

- 先把平地改成混合 rough terrain + curriculum。
- 保留 `teacher` 與 distillation 路徑，未來若要做 skill-specific teacher，架構上已經接得上。

**這次沒有直接實作的部分**

- 我這次沒有上完整 MoE skill library，因為目前任務仍以基礎 locomotion 為主，沒有 jump / climb / crouch 的單獨任務定義。

---

### 2.3 UC Berkeley / CMU — RMA: Rapid Motor Adaptation (Kumar et al., 2021)

**來源**

- arXiv: [https://arxiv.org/abs/2107.04034](https://arxiv.org/abs/2107.04034)
- Project page: [https://ashish-kmr.github.io/rma-legged-robots/](https://ashish-kmr.github.io/rma-legged-robots/)

**動機**

真機部署時最常發生的問題不是 reward 寫得不夠漂亮，而是：  
負重變了、地面摩擦變了、馬達老化了、某條腿弱掉了。  
如果 policy 不能快速推斷「現在的外在條件跟訓練名義值不一樣」，就很難真的穩。

**理論背景**

RMA 的核心是：  
**domain randomization 不只要讓 policy 被動 robust，更要讓 policy 從歷史觀測中主動估計 extrinsics。**

**他們怎麼做**

- 訓練期把摩擦、payload、馬達強度等 extrinsics 隨機化。
- teacher / base policy 可見 extrinsics latent。
- adaptation module 從最近一段 history 預測 extrinsics。

**對 RedRHex 的啟發**

這篇對我最直接的啟發有兩件事：

1. observation 不能只有單幀，必須有 **history**。  
2. 隨機化不能只有整台 robot 一起縮放，而要能表達 **單腿退化 / 故障** 這種不對稱情形。

**我這次已落地的部分**

- 新增 `policy_history_length = 5` 與 `history_observation_space = 224`。
- 加入腿級別 `main_strength_scale_per_leg`、`abad_strength_scale_per_leg`。
- 加入單腿故障 injection：
  - `dr_fault_probability = 0.12`
  - `dr_fault_strength_range = [0.15, 0.60]`
  - `dr_fault_max_legs = 1`
  - `dr_fault_apply_to_abad = True`
- fault 機率再隨 curriculum stage 放大：`stage_fault_probability_scale = [0.0, 0.0, 0.35, 0.70, 1.0]`

這些改動都直接把 RMA 的「adaptation 來源於歷史與擾動分布」翻譯進程式。

---

### 2.4 CMU — Extreme Parkour (Cheng et al., 2023)

**來源**

- arXiv: [https://arxiv.org/abs/2309.14341](https://arxiv.org/abs/2309.14341)
- Project page: [https://extreme-parkour.github.io/](https://extreme-parkour.github.io/)
- GitHub: [https://github.com/chengxuxin/extreme-parkour](https://github.com/chengxuxin/extreme-parkour)

**動機**

想證明便宜的小型腿足機器人，不靠大型 motion library，也可以學出接近 parkour 的高動態行為。

**理論背景**

Extreme Parkour 很重要的一點是：  
不是所有 locomotion 都需要超重的 controller stack。有些時候，**好地形分布 + 好 reward + privileged teacher** 就已經夠把策略帶到更高動態範圍。

**他們怎麼做**

- 單一 policy。
- privileged teacher + student distillation。
- 在隨機障礙物地形上大量訓練。

**對 RedRHex 的啟發**

RedRHex 目前不是跳躍平台，但這篇提醒我：  
**只在平地上訓練，再期待真機上 rough terrain 表現好，幾乎不可能。**

**我這次已落地的部分**

- terrain generator 改成 rough / wave / stairs / boxes 混合。
- curriculum 隨 stage 漸進拉高地形難度。

**這次沒有直接實作的部分**

- 我沒有加入專門的 clearance reward，因為目前 RedRHex 的主要目標是 robust wheg locomotion，不是高跳躍或跨大 gap。

---

### 2.5 KAIST — DreamWaQ (Nahrendra et al., 2023)

**來源**

- arXiv: [https://arxiv.org/abs/2301.10602](https://arxiv.org/abs/2301.10602)
- DreamWaQ++ follow-up: [https://arxiv.org/abs/2409.19709](https://arxiv.org/abs/2409.19709)

**動機**

不是每個機器人都該背深度相機與 mapping pipeline。  
在黑暗、煙霧、遮蔽環境中，vision pipeline 還可能直接失效。

**理論背景**

DreamWaQ 的核心是 **implicit terrain imagination**：  
不是直接觀測地形，而是從 proprio history 推斷 latent context。

**他們怎麼做**

- 用 history 建 context estimator。
- 用 asymmetric actor-critic。
- critic 看 privileged terrain / state。

**對 RedRHex 的啟發**

DreamWaQ 對我最重要的價值是證明：  
**純 proprio + history + privileged critic** 這條路本身就很強，而且非常適合現在這個 RedRHex 專案。

**我這次已落地的部分**

- actor 輸入改成 `policy + history`。
- critic 輸入改成 `policy + history + critic(privileged)`。
- teacher obs 進一步把 `policy + history + privileged` 全部串起來。

這裡要誠實說明：  
我這次沒有寫 DreamWaQ 那種獨立 latent estimator network，而是先把 DreamWaQ 所依賴的資料流與 observation interface 接好。這是刻意的工程取捨，因為對現有 RedRHex codebase 來說，先把 observation / critic / distillation 走通，收益最大。

---

### 2.6 CTS — Concurrent Teacher-Student (Wang et al., 2024)

**來源**

- arXiv: [https://arxiv.org/abs/2405.10830](https://arxiv.org/abs/2405.10830)

**動機**

傳統 teacher-student 兩階段流程慢，teacher 與 student 分離，訓練成本高。

**理論背景**

CTS 的重點是把 teacher 與 student 共享到同一個訓練框架裡，讓 privileged 與 deployable representation 同時演化。

**他們怎麼做**

- teacher encoder 看 privileged obs。
- student encoder 看 history obs。
- latent reconstruction / alignment loss。
- 同時用 RL 更新。

**對 RedRHex 的啟發**

我這次沒有直接改 rsl_rl 演算法本體去做真正 concurrent CTS，因為那會變成改 trainer / algorithm source，風險很高。  
但 CTS 對我這次仍然很重要，因為它告訴我：

- observation interface 一定要先切分成 student 與 teacher 可用的 group。
- history 和 teacher obs 不能混成一坨。

**我這次已落地的部分**

- observation group 明確切成 `policy`、`history`、`critic`、`teacher`。
- 新增 distillation runner config，先完成兩階段 teacher-student 版本。

**這裡是我的工程推論**

如果未來要做真正 CTS，只需要在目前 `policy/history/teacher` 這套 observation interface 之上，改 algorithm loss；環境端資料流已經不用重來。

---

### 2.7 ETH RSL — Symmetry-aware RL (Mittal et al., 2024)

**來源**

- arXiv: [https://arxiv.org/abs/2403.04359](https://arxiv.org/abs/2403.04359)

**動機**

左右對稱的機器人，如果不告訴 policy「左右交換後其實是等價狀態」，就會浪費大量資料效率，也容易學出不必要的偏側 gait。

**理論背景**

這篇強調的是 **task symmetry / morphological symmetry** 在 RL 中應被明確利用。  
最實用的方式之一就是資料增強，把左-右鏡像狀態與動作一併送進訓練。

**我這次已落地的部分**

- 新增 `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_symmetry.py`
- 對 `policy`、`history`、`critic`、`teacher`、`actions` 全部做左右鏡射轉換
- 在 PPO config 中啟用 `RslRlSymmetryCfg`

這不是 cosmetic 改動，而是會直接影響 sample efficiency 與 gait 對稱性。

---

### 2.8 UC Berkeley Hybrid Robotics — Morphological Symmetry (IROS 2024)

**來源**

- PDF: [https://hybrid-robotics.berkeley.edu/publications/IROS2024_Symmetry_RL_LeggedLoco.pdf](https://hybrid-robotics.berkeley.edu/publications/IROS2024_Symmetry_RL_LeggedLoco.pdf)

**動機與理論背景**

Berkeley 這篇更進一步指出，形態學上的對稱不只是能做 data augmentation，還能進一步做 representation 或 policy 結構上的等變性設計。

**對 RedRHex 的啟發**

這篇讓我確認：  
對六足這種明顯左右對稱平台，對稱性不是「可有可無的小技巧」，而是非常值得納入訓練系統的 inductive bias。

**我這次已落地的部分**

- 先採用實作風險最低、收益很高的 **mirror augmentation**。
- 沒有直接改成 group-equivariant network，因為目前主要瓶頸還在 sim-to-real robustness，不在 model architecture expressivity。

---

### 2.9 ETH RSL — Actuator Network / Sim-to-Real Actuation Awareness (Hwangbo et al., 2019)

**來源**

- arXiv / paper metadata: [https://arxiv.org/abs/1901.08652](https://arxiv.org/abs/1901.08652)

**動機**

很多 locomotion policy 在模擬中很強，但上真機立刻退化，原因常常不是 policy 不會走，而是 **actuator dynamics 被模擬得太理想**。

**理論背景**

sim-to-real gap 很大一部分來自致動器延遲、飽和、力矩上限與帶寬限制。  
就算不完全照抄 actuator network，也應該把 motor saturation、速度上限、effort 限制做得更像真機。

**我這次已落地的部分**

- `main_drive` 與 `abad` 從 `ImplicitActuatorCfg` 改為 `DCMotorCfg`
- 明確加入 `effort_limit`、`velocity_limit`、`saturation_effort`
- 保留 damper 的高剛性被動設定，但更新為新的 sim 參數欄位

這是把「policy 對理想馬達過擬合」的風險往下壓。

---

### 2.10 NVIDIA / Boston Dynamics — Spot locomotion in Isaac Lab (2024)

**來源**

- NVIDIA Technical Blog: [Closing the Sim-to-Real Gap: Training Spot Quadruped Locomotion with NVIDIA Isaac Lab](https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/)

**動機**

NVIDIA 官方 Spot 文章的價值，不在於某個數字，而在於它明確示範了：  
**Isaac Lab 本身就是被拿來做 sim-to-real locomotion pipeline 的。**

**對 RedRHex 的啟發**

這篇給我的工程方向很直接：

- terrain curriculum 要上。
- domain randomization 要上。
- actuation realism 要上。
- 驗證腳本要做，不要只看最終 reward 曲線。

**我這次已落地的部分**

- terrain generator + curriculum
- actuator realism
- validation scripts

---

### 2.11 MPC vs RL Benchmark (Akki & Chen, 2025)

**來源**

- IEEE Access DOI: [10.1109/ACCESS.2025.3582523](https://doi.org/10.1109/ACCESS.2025.3582523)
- arXiv preprint: [https://arxiv.org/abs/2501.16590](https://arxiv.org/abs/2501.16590)

**這篇對我的重要意義**

我重新查之後，這篇提供的是一個很重要的 framing，但不能被過度簡化。  
我查到的摘要重點是：

- RL 在某些設定下展現出很好的能效與對擾動的反應。
- MPC 在較大擾動恢復與未見地形泛化上仍然可能更強。

也就是說，真正的故事不是「RL 全面碾壓 MPC」，而是：

> RL 要靠 curriculum、history-based adaptation、teacher-student 與 sim-to-real 工程細節，才有機會把部署性能推上去。

這剛好就是我這次改革的總方向。

---

### 2.12 EPFL BioRob — CPG-RL (Bellegarda & Ijspeert, 2022)

**來源**

- arXiv: [https://arxiv.org/abs/2211.00458](https://arxiv.org/abs/2211.00458)

**動機**

完全自由的 end-to-end policy 很強，但在週期性 locomotion 上，結構化的 gait prior 往往更穩、更容易轉移。

**理論背景**

CPG-RL 的關鍵思想不是「一定要用神經生物學」，而是：

**週期性 locomotion 很適合把相位、節律、對稱性與步態先驗顯式寫進控制結構中。**

**對 RedRHex 的啟發**

RedRHex 本來就不是完全自由形變的腿足，而是明確有 wheg 相位結構。  
所以我這次沒有把原本的相位先驗拿掉，反而是：

- 保留 gait phase
- 保留 tripod 對應關係
- 保留 forward gait prior
- 再在上面加 history、privileged critic、symmetry、DR

這是我認為比硬改成 fully unconstrained policy 更合理的做法。

---

## 3. 我對原始 RedRHex codebase 的診斷

在我真正動手改之前，這份 codebase 的主要問題是：

1. **地形太單一**  
   預設仍以 plane 為主，對真機 rough terrain 的接觸不確定性準備不足。

2. **observation 缺少時序結構**  
   policy 沒有顯式 history，很難學到 RMA / DreamWaQ 式的隱式適應。

3. **critic / teacher 沒有獨立資料流**  
   沒有真正把 privileged info 與 deployable info 分流。

4. **domain randomization 太粗**  
   如果擾動只在 env 級別平均縮放，就無法模擬最關鍵的單腿退化與不對稱故障。

5. **對稱性沒有利用**  
   六足左右鏡像結構沒有進入訓練。

6. **致動器仍偏理想化**  
   對 sim-to-real 來說，motor saturation 不該缺席。

7. **缺少可重複的閉環驗證腳本**  
   沒有一個快速確認 observation / PPO / distillation stack 還活著的工具。

---

## 4. 這次我實際改了哪些程式

下面這一節是本次最重要的工程交付。

### 4.1 地形課程與 rough terrain generator

**檔案**

- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py:45-86`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py:503-516`

**我改了什麼**

- 新增 `REDRHEX_ROUGH_TERRAINS_CFG`
- 啟用 `terrain_type="generator"`
- 混合以下子地形：
  - `flat`
  - `random_rough`
  - `wave`
  - `stairs`
  - `boxes`

**關鍵參數**

- `size=(6.0, 6.0)`
- `num_rows=6`
- `num_cols=12`
- `curriculum=True`
- `difficulty_range=(0.0, 0.15)`
- `max_init_terrain_level=1`

**這代表什麼**

policy 不再只是在平地上學出一種很脆弱的 gait，而是從一開始就暴露在高度起伏、階梯與不連續接觸的分布之下。

---

### 4.2 致動器 realism：從理想隱式驅動改為 DCMotor

**檔案**

- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py:208-260`

**我改了什麼**

- `main_drive` 改成 `DCMotorCfg`
- `abad` 改成 `DCMotorCfg`
- `damper` 保留 implicit，但更新為新欄位 `effort_limit_sim` / `velocity_limit_sim`

**主驅動設定**

- `effort_limit = 100.0`
- `velocity_limit = 30.0`
- `stiffness = 0.0`
- `damping = 50.0`
- `saturation_effort = 100.0`

**ABAD 設定**

- `effort_limit = 8.0`
- `velocity_limit = 5.0`
- `stiffness = 40.0`
- `damping = 4.0`
- `saturation_effort = 8.0`

**這為什麼重要**

這讓 policy 在訓練時就碰到更像真機的力矩與速度限制，不會過度依賴模擬裡「其實真機做不到」的動作瞬變。

---

### 4.3 History actor + privileged critic + teacher observation

**檔案**

- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py:351-363`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py:522-577`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py:1108-1145`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py:2105-2115`

**我改了什麼**

- 保留原本 `observation_space = 56`
- 新增 `policy_history_length = 5`
- 新增 `history_observation_space = 224`
- 新增 `critic_privileged_observation_space = 47`
- 新增 `teacher_observation_space = 327`
- 新增 `_policy_obs_history` buffer

**最後 observation group 的意義**

- `policy`: 現在這一幀的 deployable proprio obs，56 維
- `history`: 前 4 幀 deployable obs 展平成 224 維
- `critic`: simulator privileged information，47 維
- `teacher`: `policy + history + critic = 327` 維

**critic 包含哪些 privileged data**

- `target_drive_vel`
- `target_abad_pos`
- `current_leg_in_stance`
- `main_strength_scale_per_leg`
- `abad_strength_scale_per_leg`
- `fault_mask`
- `mass_scale`
- `friction_scale`
- `terrain_level`
- `dr_stage_scale`
- `contact fraction`
- `action_warmup_scale`
- `body_tilt`
- `base height`
- `forward_stance_frac_ema`
- `forward_vel_ratio_proxy`
- `push_events_step`

**這些資料代表什麼**

- actor 只能看真機未來也能取得的東西
- critic / teacher 則可看 simulator 真值，幫助訓練更穩、更快

這就是 DreamWaQ / RMA / Miki / CTS 共通的資料流設計精神。

---

### 4.4 腿級別 domain randomization 與故障注入

**檔案**

- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py:1485-1523`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py:903-1004`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py:1918-1933`

**我改了什麼**

- 把 actuator strength randomization 從單一 scalar，改成 **每條腿各自採樣**
- 新增 fault injection，讓單腿主驅動與 ABAD 可以一起降級

**重要設定**

- `dr_main_actuator_strength_range = [0.85, 1.15]`
- `dr_abad_actuator_strength_range = [0.85, 1.15]`
- `dr_fault_enable = True`
- `dr_fault_probability = 0.12`
- `dr_fault_strength_range = [0.15, 0.60]`
- `dr_fault_max_legs = 1`
- `dr_fault_apply_to_abad = True`

**課程式故障機率**

- `stage_fault_probability_scale = [0.0, 0.0, 0.35, 0.70, 1.0]`

**執行層如何套用**

- `final_drive_vel = final_drive_vel * self._main_strength_scale_per_leg`
- `target_abad_pos = target_abad_pos * self._abad_strength_scale_per_leg`

這代表故障不是只是 log 上的標記，而是真的會改變 control target。

---

### 4.5 左右對稱資料增強

**檔案**

- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_symmetry.py:1-113`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_ppo_cfg.py:31-62`

**我改了什麼**

- 為 RedRHex 寫了專用 `compute_symmetric_states`
- 對以下內容做左右交換與符號鏡射：
  - body velocity
  - projected gravity
  - commands
  - main-drive sin/cos/vel
  - ABAD obs / actions
  - critic obs
  - teacher obs

**訓練時怎麼啟用**

- `symmetry_cfg=RslRlSymmetryCfg(...)`
- `use_data_augmentation=True`

**這帶來的好處**

- 讓 policy 不會因資料偶然偏差而學出左強右弱 gait
- 提升 sample efficiency

---

### 4.6 新的 PPO、teacher、distillation 訓練配置

**檔案**

- `source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_ppo_cfg.py:21-197`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_distillation_cfg.py:17-80`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/__init__.py:12-40`

**我改了什麼**

1. `PPORunnerCfg`
   - actor obs 使用 `policy + history`
   - critic obs 使用 `policy + history + critic`

2. `PPORunnerPrivilegedTeacherCfg`
   - actor / critic 都使用 `teacher`
   - 用來訓 privileged teacher

3. `RedrhexDistillationRunnerCfg`
   - student 看 `policy + history`
   - teacher 看 `teacher`
   - 用來把 privileged teacher 蒸餾成 deployable student

4. 在 gym registry 中新增：
   - `rsl_rl_teacher_cfg_entry_point`
   - `rsl_rl_distillation_cfg_entry_point`

**關鍵超參數**

`PPORunnerCfg`

- `actor_hidden_dims=[512, 256, 128]`
- `critic_hidden_dims=[512, 256, 128]`
- `learning_rate=3e-4`
- `entropy_coef=0.003`
- `clip_actions=1.0`
- `max_iterations=2500`

`RedrhexDistillationRunnerCfg`

- `student_hidden_dims=[512, 256, 128]`
- `teacher_hidden_dims=[512, 256, 128]`
- `learning_rate=1e-3`
- `gradient_length=12`
- `loss_type="huber"`

**這為什麼重要**

這讓 RedRHex 不再只有單一 PPO 訓練入口，而是具備：

- deployable PPO 訓練
- privileged teacher 訓練
- teacher-student distillation

三條清楚的訓練路徑。

---

### 4.7 驗證與閉環工具

**檔案**

- `scripts/rsl_rl/validate_reform_stack.py:23-240`
- `scripts/rsl_rl/validate_distillation_stack.py:22-102`

**我改了什麼**

`validate_reform_stack.py`

- 隨機 rollout 檢查 observation 維度與數值穩定性
- 可跑 PPO smoke
- 可跑 teacher PPO + distillation smoke
- 自動輸出 JSON 統計

`validate_distillation_stack.py`

- 獨立程序載入 teacher checkpoint
- 跑 1 iteration distillation smoke
- 產出 student checkpoint

這讓後續每次大改訓練架構時，都能在幾十秒內確認系統還活著。

---

### 4.8 小型修補：diagnostic cap

**檔案**

- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py:978-982`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py:1565-1567`

**我改了什麼**

我在驗證時發現 `diag_forward_vel_ratio_proxy` 在 stance speed 非常小時會爆成不合理大值，雖然只影響診斷，不影響控制，但會污染 TensorBoard 可讀性。  
因此新增：

- `forward_velocity_ratio_cap = 20.0`
- 對 `ratio_proxy` 做 `torch.clamp`

這是閉環修改流程中的一個例子：  
**不是只把大功能加進去，而是驗證後把診斷品質一起修乾淨。**

---

## 5. 現在 reward 與權重是怎麼設定的

這一節回答你很在意的三件事：

1. reward 現在到底在優化什麼  
2. 權重是多少  
3. 為什麼這樣設

### 5.1 reward 設計哲學

我沿用你前一版「節能 reward 不直接獎勵 spring proxy，而是看最終能效」的方向，這一點我認為是正確的。  
所以本次架構重構 **沒有推翻你前面做好的 energy-aware reward 思路**，而是把它放到更像 SOTA 的訓練框架上。

### 5.2 主要 reward 權重

**檔案**

- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py:1591-1661`

**目前主要 `v2_reward_scales`**

- `forward_progress = 5.0`
- `velocity_tracking = 4.0`
- `mode_specialization = 2.5`
- `axis_suppression = 1.5`
- `lateral_drive_soft_penalty = 1.5`
- `forward_prior_coherence = 1.2`
- `forward_prior_antiphase = 1.2`
- `forward_prior_duty = 0.9`
- `forward_prior_vel_ratio = 0.9`
- `forward_prior_overlap = 0.7`
- `height_maintain = 0.8`
- `leg_moving = 0.5`
- `stall_penalty = -2.0`
- `action_smooth = -0.01`
- `fall = -8.0`

### 5.3 能效 reward

**關鍵參數**

- `power_efficiency = 0.3`
- `power_efficiency_eps = 0.1`
- `power_efficiency_tanh_scale = 500.0`
- `torque_penalty = -0.0001`
- `torque_penalty_abad_weight = 0.5`

**數學形式**

對每一步，主動關節機械功率近似為：

\[
P_{\text{total}} = \sum_i |\tau_i \omega_i|
\]

有效推進速度為：

\[
v_{\text{progress}} = \max(0, \mathbf{v}_{xy}\cdot \hat{\mathbf{d}}_{\text{cmd}})
\]

所以 energy-aware reward 的核心概念仍是：

\[
\text{energy per progress} \approx \frac{P_{\text{total}}}{v_{\text{progress}} + \varepsilon}
\]

這個 reward 很重要，因為它不直接命令 policy「一定要用彈簧」，而是只要求 policy 用更低的單位位移能耗完成任務。  
如果 RedRHex 的彈簧與 wheg 結構真的有優勢，那種 gait 會自然浮現。

### 5.4 ForwardFast 設定

**檔案**

- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py:1773-1820`

在 `ForwardFast` 版本裡，我保留較保守的能耗權重：

- `power_efficiency = 0.15`
- `torque_penalty = -0.00005`

這是刻意的。因為 fast-convergence 版本應先確保能快速學出穩定 forward gait，再慢慢把節能權重往上拉。

---

## 6. 機器人到底用到了哪些資料，為什麼它拿得到

這一節直接回答你問的「你拿了哪些資料，機器人為什麼拿得到這些資料」。

### 6.1 部署期 actor 可用資料

這些都屬於 `policy` 與 `history`：

- `base_lin_vel`
- `base_ang_vel`
- `projected_gravity`
- `main_drive_pos_sin`
- `main_drive_pos_cos`
- `main_drive_vel`
- `abad_pos`
- `abad_vel`
- `commands`
- `gait_phase`
- `last_actions`
- 最近 4 幀歷史 observation

**為什麼真機拿得到**

- `base_lin_vel` / `base_ang_vel` / `projected_gravity`: 來自 IMU 與 state estimator
- `joint pos / vel`: 來自 encoder
- `commands`: 來自高層控制器或遙控輸入
- `last_actions`: 這是 controller 自己知道的輸出記錄
- `gait_phase`: 這是控制器內部狀態，不需要外部感測

也就是說，student / deployable actor 這邊沒有偷看 simulator-only 資料。

### 6.2 訓練期 privileged 資料

這些是 `critic` / `teacher` 額外可看，但真機 actor 不會看：

- `target_drive_vel`
- `target_abad_pos`
- `current_leg_in_stance`
- per-leg actuator scale
- fault mask
- `mass_scale`
- `friction_scale`
- `terrain_level`
- `dr_stage_scale`
- `contact_frac`
- `body_tilt`
- `base_height`
- `push_events_step`

**為什麼這些只該給訓練期**

這些資料很多是 simulator 真值，例如：

- ground friction scale
- mass randomization scale
- injected fault mask
- curriculum terrain level

真機部署時不應直接提供給 actor。  
但讓 critic / teacher 在訓練期看見它們，能加速 value estimation 與 teacher policy 學習，這正是 asymmetric actor-critic 的標準做法。

---

## 7. 驗證方法、命令與結果

這一節是完整閉環流程的核心證據。

### 7.1 驗證命令

#### A. 基本環境 smoke

```bash
TERM=xterm /home/jasonliao/isaaclab-env/bin/python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 4 \
  --steps 4 \
  --headless \
  --device cuda:0 \
  --json_out /tmp/redrhex_validate_stats.json
```

#### B. PPO smoke

```bash
TERM=xterm PYTHONUNBUFFERED=1 /home/jasonliao/isaaclab-env/bin/python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 2 \
  --steps 2 \
  --runner_smoke \
  --runner_steps 4 \
  --headless \
  --device cuda:0 \
  --json_out /tmp/redrhex_validate_runner_stats.json \
  --log_dir /tmp/redrhex_reform_runner
```

#### C. Teacher + Distillation smoke

```bash
TERM=xterm PYTHONUNBUFFERED=1 /home/jasonliao/isaaclab-env/bin/python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 2 \
  --steps 2 \
  --distill_smoke \
  --runner_steps 4 \
  --distill_steps 4 \
  --headless \
  --device cuda:0 \
  --json_out /tmp/redrhex_validate_distill_stats.json \
  --log_dir /tmp/redrhex_reform_distill
```

### 7.2 實際結果

#### A. 基本環境 smoke

`/tmp/redrhex_validate_stats.json`

```json
{
  "critic_dim": 47.0,
  "fault_env_ratio": 0.25,
  "history_dim": 224.0,
  "max_abs_reward": 16.000858306884766,
  "mean_abad_strength": 0.9838632345199585,
  "mean_fault_leg_count": 0.25,
  "mean_main_strength": 0.9678914546966553,
  "runner_smoke": 0.0,
  "teacher_dim": 327.0,
  "terrain_type": "generator"
}
```

**這證明了什麼**

- `terrain_type = generator`  
  表示 rough terrain generator 已經真的上線。

- `history_dim = 224`, `critic_dim = 47`, `teacher_dim = 327`  
  表示新的 observation interface 維度正確。

- `fault_env_ratio = 0.25`, `mean_fault_leg_count = 0.25`  
  表示 fault injection 確實發生，不是死設定。

#### B. PPO smoke

`/tmp/redrhex_validate_runner_stats.json`

```json
{
  "critic_dim": 47.0,
  "fault_env_ratio": 0.0,
  "history_dim": 224.0,
  "max_abs_reward": 2.8596978187561035,
  "mean_abad_strength": 1.002870798110962,
  "mean_fault_leg_count": 0.0,
  "mean_main_strength": 1.0113811492919922,
  "runner_smoke": 1.0,
  "teacher_dim": 327.0,
  "terrain_type": "generator"
}
```

**這證明了什麼**

- PPO runner 能真的跑 rollout + update
- observation group 與 symmetry config 沒有把 runner 弄壞
- 新 actor / critic 輸入堆疊是可訓練的

#### C. Teacher + Distillation smoke

`/tmp/redrhex_validate_distill_stats.json`

```json
{
  "critic_dim": 47.0,
  "distill_smoke": 1.0,
  "fault_env_ratio": 0.0,
  "history_dim": 224.0,
  "max_abs_reward": 2.8460097312927246,
  "mean_abad_strength": 1.002870798110962,
  "mean_fault_leg_count": 0.0,
  "mean_main_strength": 1.0113811492919922,
  "runner_smoke": 0.0,
  "teacher_dim": 327.0,
  "terrain_type": "generator"
}
```

另外，`/tmp/redrhex_reform_distill/distill/distill_stats.json` 顯示：

```json
{
  "distill_smoke": 1.0,
  "student_ckpt": "/tmp/redrhex_reform_distill/distill/distill_smoke.pt",
  "teacher_ckpt": "/tmp/redrhex_reform_distill/teacher/teacher_smoke.pt"
}
```

checkpoint 檔案也實際存在：

- teacher checkpoint: `/tmp/redrhex_reform_distill/teacher/teacher_smoke.pt`，約 7.7 MB
- student checkpoint: `/tmp/redrhex_reform_distill/distill/distill_smoke.pt`，約 2.5 MB

**這證明了什麼**

- privileged teacher config 可以正常訓練
- distillation runner 可以載入 teacher checkpoint
- student-teacher distillation stack 可以在獨立 helper script 中成功完成一次更新

這是本次最關鍵的閉環證據之一。

---

## 8. 這次改革「有沒有達到成效」

### 8.1 已經被驗證的成效

以下成效我已經可以很有把握地說成立：

1. **訓練堆疊能力提升**
   - 從原本單一路徑 PPO，升級成 `deployable PPO + privileged teacher + distillation` 三條可執行訓練路徑。

2. **robustness distribution 更接近真機**
   - terrain 不再是平地
   - actuator 不再過度理想化
   - fault injection 不再只是平均縮放

3. **資料流更符合 SOTA**
   - actor / critic / teacher / history 已經明確分流
   - 這讓 DreamWaQ / RMA / Miki / CTS 的下一步工作有真正可接的接口

4. **工程驗證從零碎手動變成可重複**
   - 現在有腳本可以快速檢查環境、PPO、teacher、distillation 是否正常

### 8.2 還不能過度宣稱的地方

我不能在這份期中報告裡直接說：

- 最終 tracking 一定提升幾%
- 最終 CoT 一定下降幾%
- 一定已經超過 MPC

因為這些都需要正式長訓練與對照實驗。

### 8.3 為什麼我仍然認為這次改革是有效的

因為 RedRHex 原本的主要瓶頸，不是單一 reward 權重少調 0.1，而是整體訓練堆疊缺少：

- rough terrain curriculum
- history-based adaptation path
- privileged teacher path
- per-leg fault randomization
- symmetry augmentation
- sim-to-real actuation realism

這些如果不先補齊，就算你一直調 reward，也很難把 policy 往真正可部署的方向推。

所以這次改革的價值，在於把 **基礎設施** 改對了。  
接下來長訓練才有意義。

---

## 9. 我建議的下一階段實驗

### 9.1 必做

如果你下一階段的近期目標只是：

- 先把機器人訓到「穩定往前直走」
- 不急著混 lateral / diagonal / yaw
- 但想保留 teacher / student 架構與節能 reward

那我建議不要直接走 `ForwardFast`，也不要一開始就跑完整五階段。  
最適合的起點是：

- `Template-Redrhex-Direct-v0 + env.stage=1`

這條路線的意義是：

- 用主 task 的正式 observation / reward / deployment stack
- 但把 curriculum 固定在 Stage1 forward-only
- 主 task 的節能 reward 權重仍然保留，例如 `power_efficiency = 0.3`、`torque_penalty = -0.0001`
- 相比之下，`ForwardFast` 比較偏向白天快速收斂與快速迭代，節能權重也刻意放得更保守

因此我建議的實驗順序是：

1. 先用 `Template-Redrhex-Direct-v0 + env.stage=1 + rsl_rl_cfg_entry_point` 訓一個正式 forward-only baseline。
2. 再用 `PPORunnerPrivilegedTeacherCfg` 在同樣的 `env.stage=1` 條件下訓一個 forward-only teacher。
3. 再用 `RedrhexDistillationRunnerCfg` 在同樣的 `env.stage=1` 條件下訓 deployable forward-only student。
4. 做以下對照：
   - 舊版 plane + no-history
   - 新版 rough terrain + history + critic
   - 新版 + symmetry
   - 新版 + fault DR
   - 新版 + teacher-student distillation

#### 9.1.1 可直接照抄的完整操作流程

如果目前實驗目標是：

- 只做穩定前進
- 保留主 task 的節能 reward、history actor、critic、teacher-student 架構
- 不急著混 lateral / diagonal / yaw

那可以直接照下面這套做。

**Step 1：進專案並啟動環境**

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

**Step 2：train forward-only baseline**

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name forward_stage1_student \
  env.stage=1
```

**Step 3：train forward-only teacher**

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name forward_stage1_teacher \
  env.stage=1
```

**Step 4：抓最新 teacher checkpoint**

```bash
TEACHER_RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg_teacher/* | head -1)")
TEACHER_CKPT=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg_teacher/$TEACHER_RUN/model_*.pt | tail -1)")
echo "TEACHER_RUN=$TEACHER_RUN"
echo "TEACHER_CKPT=$TEACHER_CKPT"
```

**Step 5：建立 distillation 讀取 teacher 的連結**

```bash
mkdir -p logs/rsl_rl/redrhex_wheg_distill
ln -s ../redrhex_wheg_teacher/$TEACHER_RUN logs/rsl_rl/redrhex_wheg_distill/$TEACHER_RUN
```

**Step 6：train forward-only distillation student**

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

這套流程的順序不能顛倒，因為：

1. baseline 是第一個可部署基線
2. teacher 是 distillation 的老師
3. distillation 一定要先有 teacher checkpoint 才能跑

### 9.2 要量的指標

- command tracking error
- episode return
- fall rate
- fault injection success rate
- energy per distance
- cost of transport
- 未見 rough terrain 成功率

### 9.3 如果你要對標 MPC

我建議不要只比平地穩定前進，而要比：

- 未知摩擦
- 單腿弱化
- 隨機推擠
- stairs / rough / mixed terrain
- unit-distance energy cost

因為這才是 RL 真正有機會展現優勢的地方。

---

## 10. 本次修改檔案總表

- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_symmetry.py`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_ppo_cfg.py`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_distillation_cfg.py`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/__init__.py`
- `scripts/rsl_rl/validate_reform_stack.py`
- `scripts/rsl_rl/validate_distillation_stack.py`

---

## 11. 參考來源

以下是本次實際查閱並作為設計依據的主要來源：

1. Miki et al., *Learning robust perceptive locomotion for quadrupedal robots in the wild*  
   [https://www.science.org/doi/10.1126/scirobotics.abk2822](https://www.science.org/doi/10.1126/scirobotics.abk2822)  
   [https://leggedrobotics.github.io/rl-perceptiveloco/](https://leggedrobotics.github.io/rl-perceptiveloco/)

2. Hoeller et al., *ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots*  
   [https://www.science.org/doi/10.1126/scirobotics.adi7566](https://www.science.org/doi/10.1126/scirobotics.adi7566)  
   [https://arxiv.org/abs/2306.14874](https://arxiv.org/abs/2306.14874)

3. Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots*  
   [https://arxiv.org/abs/2107.04034](https://arxiv.org/abs/2107.04034)  
   [https://ashish-kmr.github.io/rma-legged-robots/](https://ashish-kmr.github.io/rma-legged-robots/)

4. Cheng et al., *Extreme Parkour with Legged Robots*  
   [https://arxiv.org/abs/2309.14341](https://arxiv.org/abs/2309.14341)  
   [https://extreme-parkour.github.io/](https://extreme-parkour.github.io/)

5. Nahrendra et al., *DreamWaQ*  
   [https://arxiv.org/abs/2301.10602](https://arxiv.org/abs/2301.10602)  
   [https://arxiv.org/abs/2409.19709](https://arxiv.org/abs/2409.19709)

6. Wang et al., *Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion*  
   [https://arxiv.org/abs/2405.10830](https://arxiv.org/abs/2405.10830)

7. Mittal et al., *Symmetry Considerations for Learning Task Symmetric Robot Policies*  
   [https://arxiv.org/abs/2403.04359](https://arxiv.org/abs/2403.04359)

8. UC Berkeley Hybrid Robotics, *Leveraging Morphological Symmetry in Reinforcement Learning of Locomotion Gaits for Articulated Robots*  
   [https://hybrid-robotics.berkeley.edu/publications/IROS2024_Symmetry_RL_LeggedLoco.pdf](https://hybrid-robotics.berkeley.edu/publications/IROS2024_Symmetry_RL_LeggedLoco.pdf)

9. Hwangbo et al., *Learning agile and dynamic motor skills for legged robots*  
   [https://arxiv.org/abs/1901.08652](https://arxiv.org/abs/1901.08652)

10. NVIDIA Technical Blog, *Closing the Sim-to-Real Gap: Training Spot Quadruped Locomotion with NVIDIA Isaac Lab*  
    [https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/](https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/)

11. Akki & Chen, *Benchmarking Model Predictive Control and Reinforcement Learning-Based Control for Legged Robot Locomotion in MuJoCo Simulation*  
    [https://doi.org/10.1109/ACCESS.2025.3582523](https://doi.org/10.1109/ACCESS.2025.3582523)  
    [https://arxiv.org/abs/2501.16590](https://arxiv.org/abs/2501.16590)

12. Bellegarda & Ijspeert, *CPG-RL*  
    [https://arxiv.org/abs/2211.00458](https://arxiv.org/abs/2211.00458)

---

## 12. 總結

如果用一句話總結這次期中工作，那就是：

> 我已經把 RedRHex 從「可在平地上跑 PPO 的 locomotion task」，升級成「具備 rough terrain、history adaptation、privileged critic、teacher-student、symmetry 與腿級 fault randomization 的可部署訓練堆疊」。

這次最關鍵的不是某一個 reward 權重，而是整個訓練系統的方向已經被拉到正確的軌道上。  
接下來只要把正式長訓練與對照實驗補上，這套系統就有資格拿去做更嚴格的 sim-to-real 與 MPC 對標。
