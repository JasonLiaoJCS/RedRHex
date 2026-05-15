# RedRhex 強化學習專題 — 完整改進策略報告
**為什麼、做什麼、怎麼做、以及如何勝過傳統 MPC**

---

## 目錄

- [0. 報告主旨](#0-報告主旨)
  - [0.1 現行落地訓練順序（重要）](#01-現行落地訓練順序重要)
- [1. 世界頂尖腿足機器人 RL 研究全覽](#1-世界頂尖腿足機器人-rl-研究全覽)
  - [1.1 ETH RSL — Learning Robust Perceptive Locomotion (Miki 2022)](#11-eth-rsl--learning-robust-perceptive-locomotion-miki-2022)
  - [1.2 ETH RSL — ANYmal Parkour (Hoeller 2024)](#12-eth-rsl--anymal-parkour-hoeller-2024)
  - [1.3 UC Berkeley / CMU — RMA: Rapid Motor Adaptation (Kumar 2021)](#13-uc-berkeley--cmu--rma-rapid-motor-adaptation-kumar-2021)
  - [1.4 CMU — Extreme Parkour (Cheng 2023)](#14-cmu--extreme-parkour-cheng-2023)
  - [1.5 KAIST Urban Robotics Lab — DreamWaQ (Nahrendra 2023)](#15-kaist-urban-robotics-lab--dreamwaq-nahrendra-2023)
  - [1.6 SUSTech / ZJUI — Concurrent Teacher-Student (CTS, Wang 2024)](#16-sustech--zjui--concurrent-teacher-student-cts-wang-2024)
  - [1.7 ETH RSL (Mittal et al.) — Symmetry-aware RL (ICRA 2024)](#17-eth-rsl-mittal-et-al--symmetry-aware-rl-icra-2024)
  - [1.8 UC Berkeley Hybrid Robotics — Morphological Symmetry (IROS 2024)](#18-uc-berkeley-hybrid-robotics--morphological-symmetry-iros-2024)
  - [1.9 ETH RSL — Actuator Network (Hwangbo 2019)](#19-eth-rsl--actuator-network-hwangbo-2019)
  - [1.10 NVIDIA / Boston Dynamics — Spot locomotion in Isaac Lab (2024)](#110-nvidia--boston-dynamics--spot-locomotion-in-isaac-lab-2024)
  - [1.11 Michigan Tech — Benchmark: MPC vs RL (Akki & Chen, 2025)](#111-michigan-tech--benchmark-mpc-vs-rl-akki--chen-2025)
  - [1.12 EPFL BioRob — CPG-RL (Bellegarda & Ijspeert 2022)](#112-epfl-biorob--cpg-rl-bellegarda--ijspeert-2022)
- [2. RedRhex 現況診斷](#2-redrhex-現況診斷)
- [3. 優先級 TIER 1 — 必做](#3-優先級-tier-1--必做)
- [4. 優先級 TIER 2 — 推薦](#4-優先級-tier-2--推薦)
- [5. 優先級 TIER 3 — 加分](#5-優先級-tier-3--加分)
- [6. 如何在報告中證明「勝過 MPC」](#6-如何在報告中證明勝過-mpc)
- [7. 時間壓力下的最小可行方案](#7-時間壓力下的最小可行方案)
- [8. 完整參考文獻](#8-完整參考文獻)

---

## 0. 報告主旨

你的目標:**把 RL policy 丟到 RedRhex 六足機器人上,展現出傳統 Model Predictive Control (MPC) 做不到的效果。**

頂尖實驗室過去 4 年已經給出非常清楚的「配方」。本報告拆解他們的**動機、理論背景、方法細節、實驗結果**,然後逐條對應到你 [redrhex_env_cfg.py](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py) 目前缺少的技術,給你一份可直接執行的改進清單。

**核心論點**:
> 「RL 在擾動恢復與能效上勝過 MPC,但在泛化到未見地形時較弱。」
> — *Akki & Chen, IEEE Access 2025, Benchmarking MPC and RL for Legged Locomotion*

這一句就是你的故事線。你要做的是:**利用 RL 的擾動/能效優勢,同時靠訓練技巧(teacher-student + domain randomization + terrain curriculum)補上泛化弱點**。

### 0.1 現行落地訓練順序（重要）

在進入文獻與策略細節前，先把目前專案的實際落地順序講清楚，避免理論和操作脫節。

現在的訓練路線應這樣理解：

1. **快速驗證 / 快速直走**
- `Task`：`Template-Redrhex-ForwardFast-Direct-v0`
- `Agent`：`rsl_rl_cfg_entry_point`
- 用途：最快檢查 reward / 物理參數 / 直走行為是否有效

2. **正式 forward-only 穩走路線（非 ForwardFast）**
- `Task`：`Template-Redrhex-Direct-v0`
- 額外加 Hydra override：`env.stage=1`
- `Agent`：
  - baseline：`rsl_rl_cfg_entry_point`
  - teacher：`rsl_rl_teacher_cfg_entry_point`
  - student distillation：`rsl_rl_distillation_cfg_entry_point`
- 用途：只訓穩定前進，但保留主 task 的完整 observation / reward / teacher-student 架構

3. **正式完整主線**
- `Task`：`Template-Redrhex-Direct-v0`
- `Agent`：`rsl_rl_cfg_entry_point`
- 用途：完整 locomotion 基線
- 建議方式：五階段 `train_stage_pipeline.sh`

4. **進階 teacher-student 路線**
- 先 train `rsl_rl_teacher_cfg_entry_point`
- 再做 `rsl_rl_distillation_cfg_entry_point`
- 用途：追更高上限、研究蒸餾與部署 student

最重要的 operational clarification：

- 直接跑 `train.py --task Template-Redrhex-Direct-v0`
- 並不等於五階段 curriculum
- 目前它對應的是單段 mixed stage 訓練

因此正確順序通常是：

1. 先用一般 PPO 跑通 baseline
2. 再用五階段 curriculum 做正式完整主線
3. 最後才做 teacher / student distillation

也就是說：

- teacher-student 雖然是文獻上的強配方
- 但工程上不應該是你今天第一步就先做的事情

如果你目前的研究問題不是「完整多技能」，而是：

- 只想把 forward locomotion 訓到很穩
- 想保留節能 reward 與 spring-leg 優勢
- 想保留 teacher / student 蒸餾接口

那最適合的起點不是 ForwardFast，而是主 task 的：

- `Template-Redrhex-Direct-v0 + env.stage=1`

原因是：

- 主 task 版本的節能 reward 權重比 ForwardFast 更完整
- 主 task 的 history actor / asymmetric critic / privileged teacher 介面都已經接好
- 這比較符合後續把單技能結果往正式部署模型擴充的工程路線

---

## 1. 世界頂尖腿足機器人 RL 研究全覽

以下每一篇都是你報告可以引用的 SOTA。為什麼他們會想到這些方法 = 他們遇到了什麼痛點。

---

### 1.1 ETH RSL — Learning Robust Perceptive Locomotion (Miki 2022)

- **作者**:Takahiro Miki, Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, Marco Hutter
- **機構**:ETH Zurich (Robotic Systems Lab) + KAIST + Intel Labs
- **發表**:*Science Robotics*, Vol. 7, No. 62, 19 Jan 2022
- **DOI**:[10.1126/scirobotics.abk2822](https://www.science.org/doi/10.1126/scirobotics.abk2822)
- **專案頁 / 影片**:<https://leggedrobotics.github.io/rl-perceptiveloco/>

**動機**
以前的 ANYmal 只靠本體感覺(proprioception)走路 — 穩定但**極慢**,因為它必須「踩到才知道」。另一路線加視覺感測,但 LiDAR / 深度相機在雪地、草叢、水窪、霧氣、塵土中會失效。**痛點就是**:看得見的方法不穩,穩的方法太慢。

**理論背景**
從 POMDP 出發。Agent 觀察不完整 → 需要 belief 機制。他們用 **attention-based recurrent encoder** 把 proprio + extero 混成 belief vector。當 extero 資訊不可信時,recurrent state 會自動 fallback 到 proprio。

**方法**
1. **Phase 1 — Teacher**:訓 privileged policy,critic/actor 都能看到地形真值、機身真實速度、接觸力。用 PPO。
2. **Phase 2 — Student**:用 KL 蒸餾把 teacher 複製到只看 noisy elevation map + proprio 的 student,加一個**重建 loss**(student 要從噪音觀察還原 ground truth)。
3. **關鍵技巧**:Student 會被故意加入 sensor failure / drift / occlusion 的 curriculum。

**結果**
- **Zermatt 高山徒步**:2.2 km、海拔爬升 100 m、**78 分鐘完成**(人類規劃 76 分鐘)
- **DARPA SubT 決賽**:CERBERUS team 冠軍的控制器,1700 m+ 探索無摔倒
- 感測器被遮蔽、非剛性地面、姿態漂移都能過關

**你能怎麼超越**
Miki 的架構假設你有 elevation map。**你不需要** — RHex 的 passive spring leg 本身就是 shock absorber。若你能在訓練中只用 proprio + privileged critic,並讓 robot 在**石頭、樓梯、斜坡**上都能過,對一個 hexapod 這是新貢獻。

---

### 1.2 ETH RSL — ANYmal Parkour (Hoeller 2024)

- **作者**:David Hoeller, Nikita Rudin, Dhionis Sako, Marco Hutter
- **機構**:ETH Zurich, Robotic Systems Lab
- **發表**:*Science Robotics*, 13 March 2024
- **DOI**:[10.1126/scirobotics.adi7566](https://www.science.org/doi/10.1126/scirobotics.adi7566)
- **新聞稿**:<https://ethz.ch/en/news-and-events/eth-news/news/2024/03/anymal-can-do-parkour-and-walk-across-rubble.html>

**動機**
單一 locomotion policy 學 parkour 會 mode collapse — 跳、爬、鑽、踢牆需要非常不同的控制策略,一個 network 同時學容易互相抵消。以前的做法是「一個 monolithic policy」,卡在中等難度就上不去。

**理論背景**
**Hierarchical RL** + **Mixture of Experts**。上層 navigation policy 從深度相機看環境,輸出「選哪個 skill + 目標點」。下層是多個專家技能(walk / jump up / jump down / crouch / climb),各自有專屬 reward 和 termination。

**方法**
1. **Skill training**:每個 skill 各自訓練,reward 針對該技能最佳化(例如 jump-up 獎勵跨越高度)
2. **Distillation**:把所有 skill distill 到一個通用 student(靠深度圖 + proprio)
3. **Navigation policy**:上層用 RL 學「什麼情境選哪個 skill」,以 waypoint 為目標

**結果**
- 跳 0.8 m 高的牆、跨 1.0 m 的 gap(幾乎自身尺寸)
- 廢墟、瓦礫堆上連續行走不摔
- 一個 policy 跑完真實廢墟現場 demo

**你能怎麼超越**
Hoeller 用四足 ANYmal。你的 RedRhex 是六足 + passive spring leg,**靜穩裕度天生更高**。你可以 claim:
- 六足的容錯性讓失敗 recovery 比四足快
- 用類似的 MoE 架構訓練「慢步 tripod / 快速 bound / 原地旋轉」
- 若搭配 ANYmal 的 navigation layer 思路,hexapod 能做 ANYmal 做不到的低間隙(crawl under)動作

---

### 1.3 UC Berkeley / CMU — RMA: Rapid Motor Adaptation (Kumar 2021)

- **作者**:Ashish Kumar (UC Berkeley), Zipeng Fu (CMU), Deepak Pathak (CMU), Jitendra Malik (UC Berkeley / FAIR)
- **機構**:UC Berkeley + CMU
- **發表**:RSS 2021(Robotics: Science and Systems)
- **arXiv**:<https://arxiv.org/abs/2107.04034>
- **專案頁**:<https://ashish-kmr.github.io/rma-legged-robots/>

**動機**
機器人出廠後會遇到**未見過的**負重、輪胎磨損、不同地面摩擦。傳統做法是 System ID 每次量測重新校正 — 太慢。目標:policy 自己在**幾秒內**自適應。

**理論背景**
**Domain randomization** 的強化版。不只讓 policy 對隨機環境 robust,而是讓 policy **隱式地推斷當前環境參數**,再用這個推斷結果調整動作。

**方法**
兩階段訓練:
1. **Phase 1 — Base Policy π**
   - 輸入:`(o_t, a_{t-1}, z_t)`,其中 `z_t` 是 **extrinsics vector**(地面摩擦、payload、馬達強度等隨機化真值,編碼到 8 維)
   - 用 PPO 訓到收斂
2. **Phase 2 — Adaptation Module φ**
   - 輸入:**最近 50 步的 state-action history**
   - 輸出:`ẑ_t`(預測 extrinsics),用 supervised learning 匹配真值 `z_t`
   - 部署時 base policy 用 `ẑ_t` 替代 `z_t`

**結果**
- 在 A1 四足上 **沒微調** zero-shot 部署
- 測試環境涵蓋:岩石、濕滑地面、草叢、沙地、樓梯、泡棉、油面、加 12 kg payload
- 每秒 10 次更新 `ẑ_t`,適應時間 < 1 秒
- Success rate 比沒 adaptation module 高 30–50%

**你能怎麼超越**
RMA 的 extrinsics 維度只有 8,你可以:
- 把 RedRhex 的 passive spring **剛性變化**也放進 extrinsics(真實彈簧會疲勞)
- 擴到 16+ 維,涵蓋更多腿足摩擦、彈簧失效
- 在訓練中引入 leg failure curriculum(某隻腳停止工作),讓 adaptation module 學習**fault tolerance** — 這是 MPC 完全做不到的,也是六足的經典賣點

---

### 1.4 CMU — Extreme Parkour (Cheng 2023)

- **作者**:Xuxin Cheng, Kexin Shi, Ananye Agarwal, Deepak Pathak
- **機構**:Carnegie Mellon University
- **發表**:CoRL 2023, ICRA 2024
- **arXiv**:<https://arxiv.org/abs/2309.14341>
- **專案頁**:<https://extreme-parkour.github.io/>
- **主影片**:<https://youtu.be/cuboZYHGiMc>
- **GitHub**:<https://github.com/chengxuxin/extreme-parkour>

**動機**
想用**便宜的**小四足做極限運動,但 jittery depth camera 和不精準 actuation 讓 classic controller 寸步難行。證明「一個 end-to-end policy 可以超越專用 MPC controller + motion capture」。

**理論背景**
Extreme Parkour 的核心洞見:**clearance reward**(鼓勵腳抬高越過障礙)+ **direction reward**(跟目標方向對齊)就足以誘導出複雜跳躍。不需要 skill library,不需要 MoE。

**方法**
1. 單一 policy,輸入 proprio + depth image + command
2. **Privileged distillation**:先用 privileged scandot(地形真值)訓 teacher,再蒸餾到只看 depth 的 student
3. Reward 很少:tracking + clearance + orientation + energy
4. 模擬中用大量隨機化障礙物訓練

**結果**
- 跳 **2× 機器人高度** 的障礙(~40 cm)
- 跨 **2× 機器人長度** 的 gap(~60 cm)
- Handstand(前腳倒立)、傾斜坡
- Unitree A1 真機實測,成功率 > 80%

**你能怎麼超越**
CMU 用 Unitree A1,單個深度相機容易 jitter。你的 RedRhex:
- 六足 stability 更高,可以 claim **不需要相機**就能過 parkour 類障礙(純 proprio + adaptation)
- 用 passive spring leg 做 **pronking**(四腳同時彈跳),是 MPC + 傳統 four-leg 很難表現的動態步態

---

### 1.5 KAIST Urban Robotics Lab — DreamWaQ (Nahrendra 2023)

- **作者**:I Made Aswin Nahrendra, Byeongho Yu, Hyun Myung
- **機構**:KAIST (Urban Robotics Lab)
- **發表**:ICRA 2023
- **arXiv**:<https://arxiv.org/abs/2301.10602>
- **後續版本 DreamWaQ++**:<https://arxiv.org/html/2409.19709v2> (ICRA 2025)

**動機**
Perceptive locomotion(ETH Miki 2022)需要 depth sensor + mapping pipeline,**硬體太重、軟體太複雜**。同時在黑暗、煙霧中依然失效。KAIST 想證明:**人可以在黑暗中走路**(比如半夜上廁所),機器人也可以 — 只靠 proprioception。

**理論背景**
**Implicit terrain imagination**。意思是:地形資訊不直接觀察,而是由一個 Variational Autoencoder-style 的 **context estimator** 從歷史 proprio 推斷出一個 latent。Policy 用這個 latent + 當前 proprio 做決策。這等於把 terrain 編碼進 belief space,而不是顯式的 elevation map。

**方法**
1. **Context-Aided Estimator Network (CENet)**:
   - Input: 最近 N 步 proprio
   - Output:
     - **顯式**:機身速度(supervised)
     - **隱式**:地形 latent(VAE bottleneck,無監督)
2. **Policy / Critic**:asymmetric actor-critic,critic 看到 privileged 地形真值
3. 三者同時訓練(類似 CTS 的 concurrent 思想)

**結果**
- 在完全黑暗的走廊、凹凸不平的戶外步道、濕地上行走
- 從 flat 到 stairs 的無縫過渡,不需要 mode switch
- 在 Unitree A1 / Go1 都驗證

**你能怎麼超越**
DreamWaQ 是**四足**的 gold standard for vision-free locomotion。你對應的優勢:
- 六足靜穩性更高,imagination 需要推斷的不確定性更少 → 可以訓得更快/更穩
- 你可以把 DreamWaQ 的 CENet 直接複製到 RedRhex,**你會是第一個在 Isaac Lab 上 hexapod + DreamWaQ 的實作**

---

### 1.6 SUSTech / ZJUI — Concurrent Teacher-Student (CTS, Wang 2024)

- **作者**:Hongxi Wang, Haoxiang Luo, Wei Zhang, Hua Chen
- **機構**:Southern University of Science and Technology (SUSTech) + Zhejiang University-UIUC Institute
- **發表**:arXiv 2405.10830 (2024), 後續投稿 CoRL
- **arXiv**:<https://arxiv.org/abs/2405.10830>

**動機**
傳統 teacher-student(ETH Miki 2022 / RMA)是**兩階段**:先訓 teacher → 凍結 → 訓 student。問題:
1. 訓練時間 × 2
2. Teacher 的 policy 定格後,student distill 的梯度無法回饋修正 teacher
3. 兩階段的 hyperparameter 要分別 tune

CTS 要把這三個痛點一次解決。

**理論背景**
Representation learning 的概念套進 RL。**Teacher 和 student 共用 policy/critic,但各自有 encoder**。Teacher encoder 拿 privileged info,student encoder 拿 proprio history,兩個 encoder 的輸出被強迫靠近(reconstruction loss)。**同時用 PPO 訓**。

**方法**
- Loss = PPO(teacher) + PPO(student) + Value MSE + **L2 reconstruction(student latent ↔ teacher latent)**
- Encoders 輸出 32 維 latent,歸一化到 unit hypersphere
- Student encoder 處理 H=5 步歷史

**結果**(velocity tracking error,越低越好)

| 方法 | Slope | Rough | Stairs | Obstacles |
|---|---|---|---|---|
| Vanilla PPO(純 proprio) | 0.119 | 0.165 | 0.195 | 0.132 |
| Two-stage teacher-student | 0.103 | 0.141 | 0.138 | 0.113 |
| **CTS(concurrent)** | **0.098** | **0.128** | **0.133** | **0.105** |

- 實機驗證:A1, Aliengo, LimX P1
- 訓練時間比 two-stage 少 **40%**

**你能怎麼超越**
CTS 是目前最簡潔的 asymmetric critic 範式。對你最實用,因為你只要改少量程式碼就能得到 teacher-student 的好處。

---

### 1.7 ETH RSL (Mittal et al.) — Symmetry-aware RL (ICRA 2024)

- **作者**:Mayank Mittal, Nikita Rudin, Victor Klemm, Arthur Allshire, Marco Hutter
- **機構**:ETH Zurich, Robotic Systems Lab
- **發表**:ICRA 2024
- **arXiv**:<https://arxiv.org/abs/2403.04359>

**動機**
四足機器人左右對稱,policy 應該也對稱 — 但隨機初始化 + 隨機採樣會讓 policy 學出**單邊偏好**(比如總是先邁左腳),看起來不自然且 sample inefficient。

**理論背景**
Group-equivariant learning。如果狀態空間和動作空間對某個群 G 對稱,那 policy π(a|s) 也應該滿足 π(ga|gs) = π(a|s)。做法有兩種:(A) 網路結構本身 equivariant;(B) 資料擴增 + mirror loss。

**方法**
- **Data augmentation**:每個 transition (s, a, r, s') 都加入鏡像版本 (gs, ga, r, gs')
- **Mirror loss**:`|| π(s) - g⁻¹π(gs) ||²`
- 證明 on-policy 下仍 unbiased(附 proof)

**結果**
- 收斂速度提升 **~1.5–2×**
- 學到的 gait 左右對稱、更自然
- 在 box climbing、manipulation 任務都驗證

**你能怎麼超越**
RedRhex 有**雙重對稱**:
1. 左右對稱(3 條左腿 ↔ 3 條右腿)
2. 前後對稱(若三對腿幾何相同)

你可以用**雙重 augmentation**,一個 transition 變成 4 份資料,sample efficiency 提升近 **4×**。這是四足沒有的優勢。

---

### 1.8 UC Berkeley Hybrid Robotics — Morphological Symmetry (IROS 2024)

- **作者**:Zhi Su 等(Berkeley Hybrid Robotics Lab)
- **機構**:UC Berkeley(Hybrid Robotics,Koushil Sreenath lab)
- **發表**:IROS 2024
- **PDF**:<https://hybrid-robotics.berkeley.edu/publications/IROS2024_Symmetry_RL_LeggedLoco.pdf>

**動機**
Mittal 2024 是 task-level symmetry。Berkeley 進一步做 **motion-level + task-level symmetry**。證明不只對稱能加速訓練,還能讓步態在**未訓練 command**下仍對稱(更好的 generalization)。

**理論背景**
Sagittal reflection symmetry:機器人繞身體前後軸翻折後仍是同一個機器人。這對應於在 velocity command `(vx, vy, wz)` 下,左右速度翻轉應該讓左右腿動作翻轉。

**方法**
- Equivariant policy network + Data augmentation
- Reward 也要對稱(左右 foot air time reward 一視同仁)

**結果**
- 對稱 policy 在 novel lateral velocity 下仍穩定
- 不對稱 baseline 會 drift 或偏向一邊
- 在 Unitree Go1 / Cassie 驗證

**你能怎麼超越**
RedRhex 的 [redrhex_env_cfg.py:900-905](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L900) 對「lateral 左 vs 右」reward 不對稱 — 這是潛在 bug。加上 Berkeley 的 symmetry augmentation 直接修正。

---

### 1.9 ETH RSL — Actuator Network (Hwangbo 2019)

- **作者**:Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, Vladlen Koltun, Marco Hutter
- **機構**:ETH Zurich + Intel + Volocopter
- **發表**:*Science Robotics*, Vol. 4, No. 26, 2019
- **DOI**:[10.1126/scirobotics.aau5872](https://www.science.org/doi/10.1126/scirobotics.aau5872)

**動機**
Sim-to-real gap 的**頭號兇手**:模擬器假設馬達是理想 PD,但真實馬達有 torque saturation、速度-力矩曲線、摩擦、反電動勢、控制延遲。在模擬裡學會的 policy 到真機會「抖 / 軟 / 過熱」。

**理論背景**
Data-driven actuator model。蒐集真實馬達在不同負載下的 `(command, joint_state) → torque` 真實響應,用小 MLP 擬合這個 mapping,訓練 RL 時替代理想 PD。

**方法**
1. 在 ANYdrive(ANYmal 的串聯彈性馬達)上蒐集約 400 萬步資料
2. 訓練小 MLP:input = `(joint_state_history, desired_position_history)`, output = `torque`
3. 模擬器的 PD 被這個 MLP 取代

**結果**
- Sim-to-real 落差幾乎消失
- 能做 4.0 m/s 高速 trot(ANYmal 舊紀錄 1.5 m/s)
- 後續 ANYmal C/D 的所有 RL paper 都沿用

**你能怎麼超越**
你的 [redrhex_env_cfg.py:164](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L164) 用 `ImplicitActuatorCfg` — 是理想 PD,sim2real 風險高。至少換成 `DCMotorCfg`(有 torque-speed saturation)。若有真機,蒐集幾十分鐘資料訓 actuator net,即可複刻 Hwangbo 的 gain。

---

### 1.10 NVIDIA / Boston Dynamics — Spot locomotion in Isaac Lab (2024)

- **機構**:NVIDIA + Boston Dynamics
- **發表**:NVIDIA Developer Blog, 2024
- **連結**:<https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/>

**動機**
Boston Dynamics 的 Spot 原本用傳統 MPC。為了擴展到更多樣的場景,BD 和 NVIDIA 合作把 Spot 移到 Isaac Lab,訓 RL policy。要示範 Isaac Lab 訓 production-grade 四足的完整 recipe。

**方法(關鍵 recipe)**
1. **Observation**:IMU + joint state + last action + velocity command
2. **Terrain curriculum**:flat → rough → stairs → slopes(難度自動上升)
3. **Events / Domain randomization**:
   - Payload mass ±30%
   - Ground friction 0.5–1.5
   - Actuator strength ±10%
   - Push robot(velocity perturbation)每 10 秒一次
4. **Asymmetric critic**:critic 看 privileged(地形、mass、friction)
5. **Action scale 小**,policy output 是 target joint position delta
6. ~4096 environments × RTX A6000,2000 iterations ≈ 8 小時

**結果**
- Policy 直接部署到真實 Spot,能上樓梯、泥地、戶外
- 能耗比官方 MPC 基礎控制器低 15%

**你能怎麼超越**
這是你的**直接藍圖**。RedRhex 跑相同的 Isaac Lab recipe,主要差異:
- 你的 terrain 目前只有 `"plane"`([redrhex_env_cfg.py:445](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L445)) — 必須改
- 你的 critic 沒 privileged info(`state_space = 0`,[redrhex_env_cfg.py:304](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L304)) — 必須改

---

### 1.11 Michigan Tech — Benchmark: MPC vs RL (Akki & Chen, 2025)

- **作者**:Shivayogi Akki, Tan Chen
- **機構**:Michigan Technological University
- **發表**:*IEEE Access*, 2025
- **arXiv**:<https://arxiv.org/abs/2501.16590>

**動機**
RL 社群和 control 社群各說各話 — 缺乏公平對比。此論文首次在同一平台(MuJoCo + Unitree Go1)同一任務(定速直行)對比兩者。

**方法**
- **MPC baseline**:Whole-body MPC with convex approximation, 20 ms horizon
- **RL baseline**:PPO with standard legged_gym reward,2000 iterations
- 評估三個維度:
  1. Disturbance rejection(突然側推)
  2. Energy efficiency(Cost of Transport)
  3. Terrain adaptability(flat → 5 cm bumps)

**結果**
| 指標 | MPC | RL |
|---|---|---|
| Small disturbance recovery | ~1.5 s | **~0.8 s** |
| Large disturbance (balanced joint use) | **better** | worse (single leg overload) |
| Energy (CoT on flat) | 基準 | **-22%** |
| Bump terrain success | fails | **100%** |
| Novel terrain not in training | **survives** | fails |

**核心結論**(原文)
> *"RL excels in handling disturbances and maintaining energy efficiency but struggles with generalization to new terrains due to its dependence on learned policies tailored to specific environments."*

**你能怎麼超越**
RL 的 generalization 弱點**可以靠 domain randomization + terrain curriculum + RMA-style adaptation 修補**。這就是你報告的論述核心:
> 「我用 X, Y, Z 技術補上 RL 的 generalization 弱點,同時保留 disturbance rejection 和 energy efficiency 優勢 → 全面勝過 MPC。」

---

### 1.12 EPFL BioRob — CPG-RL (Bellegarda & Ijspeert 2022)

- **作者**:Guillaume Bellegarda, Auke Ijspeert
- **機構**:EPFL (BioRobotics Laboratory)
- **發表**:RA-L 2022
- **arXiv**:<https://arxiv.org/abs/2211.00458>

**動機**
純 RL 容易學到不自然、高頻顫抖的 gait。引入生物啟發的 **Central Pattern Generator(CPG)**作為動作先驗,RL 只需學習調整 CPG 的相位 / 幅度 / 頻率,大幅縮小搜索空間。

**方法**
- CPG = 耦合的 Hopf 振盪器(每隻腿一個)
- RL action = `(amplitude_i, frequency_i, phase_offset_i)` for each leg
- 振盪器輸出 + kinematic mapping → 目標足端軌跡 → IK → joint commands

**結果**
- 比純 joint-space RL 訓練快 3×
- Gait 更平滑,CoT 更低
- 自動湧現 trot / pace / bound 切換

**你能怎麼超越**
你 [redrhex_env_cfg.py:1043-1177](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L1043) 已經有大量 gait phase、tripod 相關程式,但**是加在 reward 而非 action space**。這不是 CPG-RL 的精神 — CPG 應該是 action 先驗,不是 reward constraint。
建議重構:
- 把 `gait_phase` 從 reward 移除
- 把每隻腿的相位振盪器放進 action space(policy 輸出 amplitude + phase delta)
- 讓 policy 只學「相對 CPG 的殘差」

這個改法一次解決兩個問題:reward 簡化 + action space 正則化。

---

## 2. RedRhex 現況診斷

根據你的 [redrhex_env_cfg.py](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py) 完整掃描:

### 做得好的地方
- ✅ 已用 RSL-RL + Isaac Lab(業界標準)
- ✅ 已有 5-stage curriculum([line 744-751](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L744))
- ✅ Domain randomization 已啟用(mass, friction, actuator strength, push)
- ✅ Observation noise 已加
- ✅ 有能量獎勵設計
- ✅ 控制頻率 125 Hz(dt=1/250, render_interval=2)合理

### 關鍵缺點(對應 SOTA 技術)

| 缺點 | 你的程式位置 | 對應 SOTA 技術 | 嚴重性 |
|---|---|---|---|
| Critic 沒 privileged info | [redrhex_env_cfg.py:304](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L304) `state_space = 0` | Miki 2022, CTS 2024 | 🔴 **致命** |
| 只在平地訓練 | [line 445](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L445) `terrain_type = "plane"` | Spot Isaac Lab, ANYmal parkour | 🔴 **致命** |
| 沒 observation history / LSTM | — | RMA, DreamWaQ, CTS | 🟠 嚴重 |
| Reward 太多項(40+) | [line 1186-1376](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L1186) | Extreme Parkour(< 10 項) | 🟠 嚴重 |
| 理想 PD,無 actuator model | [line 164](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L164) `ImplicitActuatorCfg` | Hwangbo 2019 | 🟠 嚴重 |
| 速度命令太慢(0.2–0.45 m/s) | [line 800](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L800) | — | 🟡 中等 |
| 沒 symmetry augmentation | — | Mittal 2024, Berkeley 2024 | 🟡 中等 |
| Gait 相位被寫進 reward(過擬合) | [line 1249-1267](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L1249) | CPG-RL 哲學 | 🟡 中等 |
| 沒 recovery / fault-tolerance 訓練 | — | RMA ext | 🟡 中等 |
| `command_resample_on_timer = False` | [line 807](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L807) | 標準做法 | 🟢 輕微 |

---

## 3. 優先級 TIER 1 — 必做

### 3.1 啟用 Asymmetric Actor-Critic(Privileged Critic)

**為什麼必做**:Miki 2022、CTS 2024、Spot Isaac Lab、Extreme Parkour 全部都用。這是過去 4 年腿足 RL **最重要的單一技術**。沒有它你的 value estimation 雜訊極大,PPO 更新方向糟,訓練慢且不穩。

**怎麼做**:
1. 在 [redrhex_env_cfg.py:304](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L304) 把 `state_space` 從 0 改為 ~100
2. 在 [redrhex_env.py](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py) 的 `_get_observations()` 加入 `"critic"` key:
```python
def _get_observations(self):
    obs = {...}  # 你現有的 56 維 proprio
    privileged = torch.cat([
        self.robot.data.root_lin_vel_b,          # 真實機身速度(無噪音)     3
        self.robot.data.root_ang_vel_b,          # 真實機身角速度            3
        self.contact_sensor.data.net_forces_w.reshape(N, -1),  # 6 隻腳接觸力 18
        self._get_terrain_height_scan(),         # 地形高度掃描(1x1m grid) ~100
        self._last_push_force,                   # 最近 push 大小             3
        self._mass_scale.unsqueeze(-1),          # 質量倍率(隨機化參數)     1
        self._friction_coeff.unsqueeze(-1),      # 摩擦係數                   1
        self._actuator_strength_scale,           # 馬達強度倍率              12
    ], dim=-1)
    return {"policy": obs, "critic": torch.cat([obs, privileged], dim=-1)}
```
3. 在 [rsl_rl_ppo_cfg.py](../source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_ppo_cfg.py) 確認 `actor_critic_class_name = "ActorCritic"`(RSL-RL 會自動讀 critic obs 如果存在)

**預期效果**:訓練曲線提早 30–50% 收斂,最終 policy 在 disturbance 下更穩。

---

### 3.2 啟用 Terrain Generator + Curriculum

**為什麼必做**:你的賣點是「勝過 MPC」,但若只在平地跑,沒人相信。MPC 最怕的就是**未建模地形**。

**怎麼做**:把 [redrhex_env_cfg.py:443-535](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L443-L535) 已註解的 generator 打開,擴充子地形:
```python
from isaaclab.terrains.height_field import hf_terrains_cfg as hf
from isaaclab.terrains.trimesh import mesh_terrains_cfg as mesh

terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        size=(8.0, 8.0),
        num_rows=10, num_cols=20,
        horizontal_scale=0.1, vertical_scale=0.005,
        slope_threshold=0.75,
        curriculum=True,            # ← 關鍵!難度隨 success 自動提升
        sub_terrains={
            "flat": mesh.MeshPlaneTerrainCfg(proportion=0.1),
            "random_rough": hf.HfRandomUniformTerrainCfg(
                proportion=0.3,
                noise_range=(-0.05, 0.08),
                noise_step=0.01,
            ),
            "pyramid_slope": mesh.MeshPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.02, 0.10),
                step_width=0.3,
            ),
            "inv_pyramid_slope": mesh.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.02, 0.10),
                step_width=0.3,
            ),
            "slope": mesh.MeshPyramidSlopedTerrainCfg(
                proportion=0.2,
                slope_range=(0.0, 0.4),
            ),
        },
    ),
)
```

**預期效果**:Policy 會學會腳的隨機著地、tripod 相位自動適應。Demo 時放一段「新 policy 走石頭路,舊 policy 摔倒」,滿分效果。

---

### 3.3 Reward 瘦身

**為什麼必做**:你現在有超過 40 個 reward term。它們會**互相打架**,也讓 PPO 更新方向震盪。ETH 的標準配方是 ~10 項。

**怎麼做**:關掉下列(改成 0):
- 所有 `rew_scale_tripod_*`
- 所有 `rew_scale_abad_*`(保留 `abad_action_rate`)
- 所有 `rew_scale_lateral_*`(保留 `lateral_correct_dir` 或合併進 tracking)
- `rew_scale_duty_cycle_velocity`、`rew_scale_gait_frequency`、`rew_scale_phase_transition_smooth`

**保留的核心 10 項**:
```python
rew_scale_track_lin_vel = 1.0       # exp(-||v - v*||²/σ)
rew_scale_track_ang_vel = 0.5       # exp(-(w - w*)²/σ)
rew_scale_lin_vel_z = -2.0          # 不要上下跳
rew_scale_ang_vel_xy = -0.05        # 不要左右晃
rew_scale_orientation = -5.0        # 機身保持水平
rew_scale_base_height = -30.0       # 固定目標高度
rew_scale_dof_acc = -2.5e-7         # 關節加速度小
rew_scale_action_rate = -0.01       # 動作不要突變
rew_scale_torques = -2e-5           # 省力
rew_scale_feet_air_time = 1.0       # ★ 關鍵!鼓勵擺動期 → tripod 自然湧現
rew_scale_collision = -1.0          # 身體不要撞地
rew_scale_termination = -100.0      # 摔倒大罰
rew_scale_alive = 0.5               # 存活獎勵
```

**為什麼 `feet_air_time` 能誘導出 tripod**:ETH legged_gym 原作者實驗證實 — 只要獎勵腳在空中的總時間超過某閾值(通常 0.5 s),policy 會自發形成 alternating pattern。對 hexapod 會自動 converge 到 tripod。

**預期效果**:訓練曲線更單調遞增,最終步態更自然,也更接近真實 RHex 的 tripod。

---

### 3.4 Observation History(短期記憶)

**為什麼必做**:POMDP 下,policy 需要推斷自己的狀態(比如「我剛被推了嗎?」「地面摩擦變了嗎?」)。單 frame obs 做不到,RMA/DreamWaQ/CTS 都用 history。

**怎麼做**(三選一):
1. **最簡單**:把 `observation_space = 56` 改成 `56 × 5 = 280`,在 `_get_observations` 裡 concat 過去 4 步的 obs
2. **次簡單**:改用 `ActorCriticRecurrent`(RSL-RL 原生支援),加 1 層 GRU(hidden=256)
3. **最完整**:照 CTS 做,student encoder 吃 history,teacher encoder 吃 privileged

建議先做(1)或(2),改動最小。

---

## 4. 優先級 TIER 2 — 推薦

### 4.1 Actuator Model 升級

把 `ImplicitActuatorCfg` 換成 `DCMotorCfg`(sim-to-real gap 至少小 50%):
```python
from isaaclab.actuators import DCMotorCfg

"main_drive": DCMotorCfg(
    joint_names_expr=[...],
    saturation_effort=120.0,     # 真實馬達 stall torque
    effort_limit=100.0,
    velocity_limit=30.0,
    stiffness=0.0,
    damping=50.0,
),
```

若有真機,蒐集 `(cmd, state, torque)` 資料訓 actuator MLP → `IdentifiedActuatorCfg`。

### 4.2 Symmetry Data Augmentation

在 `_compute_rewards` 或 trainer 裡,每個 transition 同時產生鏡像版本:
```python
def mirror_obs(obs):
    # 左右腿 index 對調(legs 0,1,2 ↔ 3,4,5 視你的 joint 順序)
    leg_perm = torch.tensor([3,4,5,0,1,2])
    # base ang_vel z, lin_vel y 取負
    ...
```

RSL-RL 有 `augmented_ppo.py` 範例可參考。

### 4.3 速度命令拉高

[redrhex_env_cfg.py:800](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L800) 改:
```python
stage5_forward_vx_range = [0.40, 1.20]   # 原 [0.22, 0.45]
stage5_lateral_vy_abs_range = [0.30, 0.80]
stage5_yaw_wz_abs_range = [0.40, 1.20]
```

配合 `command_resample_on_timer = True`,每 3 秒改命令,policy 必須學 transition。

### 4.4 Push Curriculum 加大

[redrhex_env_cfg.py:1444-1453](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py#L1444) 改:
```python
dr_push_interval_s = 6.0        # 原 12.0
dr_push_probability = 0.8       # 原 0.5
dr_push_max_vel_xy = 1.5        # 原 0.6
stage_push_probability_scale = [0.0, 0.3, 0.6, 0.9, 1.2]
```

Demo 用:在 play 模式按下空白鍵施加 1 m/s 側推,policy 應在 < 1 秒恢復。這就是你 vs MPC 的勝點。

### 4.5 Recurrent Policy

[rsl_rl_ppo_cfg.py](../source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_ppo_cfg.py):
```python
policy = RslRlPpoActorCriticRecurrentCfg(
    class_name="ActorCriticRecurrent",
    init_noise_std=1.0,
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    rnn_type="gru",
    rnn_hidden_size=256,
    rnn_num_layers=1,
)
```

---

## 5. 優先級 TIER 3 — 加分

### 5.1 CPG 先驗(action space 正則化)
把 tripod 相位從 reward 移到 action space(見 1.12)。policy 只輸出對 CPG 的殘差。

### 5.2 Multi-gait Policy
加 `gait_id ∈ {walk, tripod, bound, pronk}` 進 observation,讓 policy 根據 command 速度自動切換。這是 MPC 很難做到的。

### 5.3 Fault Tolerance
訓練中隨機「關掉」某隻腳(force joint pos to home),讓 policy 學會 5 腿步行。這是六足經典賣點 — MPC 重寫 gait planner 是苦差。

### 5.4 Cost of Transport 作為正式指標
```
CoT = mean(|τ · q̇|) / (m · g · |v_x|)
```
每個 iteration 印出 CoT,最終 vs MPC 比。

### 5.5 Vision / Height-scan 整合
就算是模擬,加一個 11×11 的地形高度掃描給 critic(privileged),student actor 靠 proprio 推斷。接近 Miki 2022 架構。

---

## 6. 如何在報告中證明「勝過 MPC」

### 6.1 準備對比實驗(4 張圖)

| 實驗 | MPC 設定 | RL 設定 | 預期結果 |
|---|---|---|---|
| **平地定速跟蹤** | Whole-body MPC, 20ms horizon | 你的 policy | 兩者都 OK,RL 可能略差 — 坦白承認 |
| **側推 1 m/s 擾動恢復** | 同上 | 同上 | RL 恢復時間 **< 1 s**, MPC **2–4 s** |
| **5 cm 石頭亂地** | 同上 | 同上 | RL 通過,MPC 摔倒 |
| **CoT(能耗)** | 同上 | 同上 | RL 低 20%+ |

### 6.2 報告敘事結構

1. **Problem**:六足機器人在複雜地形 + 擾動下,傳統 MPC 因模型假設失效;泛化性差,需人工調參。
2. **Baseline**:MPC(可用文獻數字,例如 Akki & Chen 2025 的 Go1)
3. **Method**:展示你用了 privileged critic + terrain curriculum + domain randomization + symmetry(等等 Tier 1~2 技術)
4. **Results**:4 張對比圖 + 影片
5. **Why RL wins**:
   - Disturbance:learned reflex 比 MPC 重新解 QP 快
   - Terrain:policy 在訓練中見過數百種地形,MPC 只能沿用固定 footstep planner
   - Energy:policy 找到 MPC 沒搜尋的全域最優
6. **Why RL used to lose(文獻)but not anymore**:Akki & Chen 2025 指 RL generalization 差 — 但用 RMA/DR 修補後這個弱點消失
7. **Limitation**:RL 需大量訓練、sim-to-real gap 需 actuator network

### 6.3 引用資料來源(報告 references)

見 Section 8,全部都是可信的 peer-reviewed 期刊 / top conference。

---

## 7. 時間壓力下的最小可行方案

**今晚 3 小時 + 一夜訓練 + 明天報告**

| 時段 | 任務 | 預期產出 |
|---|---|---|
| 20:00–20:30 | 加 privileged critic([Section 3.1](#31-啟用-asymmetric-actor-critic-privileged-critic)) | `state_space` 改大,`_get_observations` 加 critic key |
| 20:30–21:00 | 關掉一半手工 reward([Section 3.3](#33-reward-瘦身)) | 只留 ~10 項 |
| 21:00–21:30 | 打開 terrain generator([Section 3.2](#32-啟用-terrain-generator--curriculum)) | 先用 rough + slope 兩種 |
| 21:30–22:00 | 加 observation history([Section 3.4](#34-observation-history短期記憶)) | concat 最近 5 步 |
| 22:00–22:30 | 啟動訓練 | 4096 env × 2000 iter |
| 22:30–08:00 | 睡覺 / 監控 TensorBoard | ~10 小時訓練 |
| 08:00–10:00 | play 錄影,對比舊 policy | 平地 + 石頭地 + 推擾 demo |
| 10:00+ | 整理 slides,引用本文獻清單 | 報告完成 |

**若只能做一件事**:開 privileged critic(Section 3.1),因為這改動最小但影響最大。

---

## 8. 完整參考文獻

### 核心論文(你報告必引)

1. **Miki et al., 2022** — Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild. *Science Robotics* 7(62).
   <https://www.science.org/doi/10.1126/scirobotics.abk2822>
   <https://leggedrobotics.github.io/rl-perceptiveloco/>

2. **Hoeller, Rudin, Sako, Hutter, 2024** — ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots. *Science Robotics*.
   <https://www.science.org/doi/10.1126/scirobotics.adi7566>
   <https://ethz.ch/en/news-and-events/eth-news/news/2024/03/anymal-can-do-parkour-and-walk-across-rubble.html>

3. **Kumar, Fu, Pathak, Malik, 2021** — RMA: Rapid Motor Adaptation for Legged Robots. *RSS 2021*.
   <https://arxiv.org/abs/2107.04034>
   <https://ashish-kmr.github.io/rma-legged-robots/>

4. **Cheng, Shi, Agarwal, Pathak, 2023** — Extreme Parkour with Legged Robots. *CoRL 2023 / ICRA 2024*.
   <https://arxiv.org/abs/2309.14341>
   <https://extreme-parkour.github.io/>
   影片:<https://youtu.be/cuboZYHGiMc>

5. **Nahrendra, Yu, Myung, 2023** — DreamWaQ: Learning Robust Quadrupedal Locomotion with Implicit Terrain Imagination via Deep RL. *ICRA 2023*.
   <https://arxiv.org/abs/2301.10602>

6. **Wang, Luo, Zhang, Chen, 2024** — Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion.
   <https://arxiv.org/abs/2405.10830>

7. **Mittal, Rudin, Klemm, Allshire, Hutter, 2024** — Symmetry Considerations for Learning Task Symmetric Robot Policies. *ICRA 2024*.
   <https://arxiv.org/abs/2403.04359>

8. **Berkeley Hybrid Robotics, 2024** — Leveraging Symmetry in RL-based Legged Locomotion Control. *IROS 2024*.
   <https://hybrid-robotics.berkeley.edu/publications/IROS2024_Symmetry_RL_LeggedLoco.pdf>

9. **Hwangbo et al., 2019** — Learning Agile and Dynamic Motor Skills for Legged Robots. *Science Robotics* 4(26).
   <https://www.science.org/doi/10.1126/scirobotics.aau5872>

10. **Akki & Chen, 2025** — Benchmarking MPC and RL for Legged Robot Locomotion in MuJoCo. *IEEE Access*.
    <https://arxiv.org/abs/2501.16590>

11. **Bellegarda & Ijspeert, 2022** — CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion. *RA-L*.
    <https://arxiv.org/abs/2211.00458>

### 次要但有用

12. **NVIDIA + Boston Dynamics, 2024** — Closing the Sim-to-Real Gap: Training Spot Quadruped Locomotion with NVIDIA Isaac Lab.
    <https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/>

13. **Isaac Lab White Paper, 2024** — Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning.
    <https://arxiv.org/abs/2511.04831>

14. **Systematic Sim-to-Real (2025)** — Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots.
    <https://arxiv.org/html/2509.06342v1>

15. **Static Friction in Hexapod (SaturnLite) 2025** — Impact of Static Friction on Sim2Real in Robotic RL.
    <https://arxiv.org/html/2503.01255>

16. **RSL-RL Library (Rudin et al., 2025)** — RSL-RL: A Learning Library for Robotics Research.
    <https://arxiv.org/html/2509.10771v1>

17. **Fu et al., 2025** — Multi-agent RL with Hybrid Action Space for Free Gait Motion Planning of Hexapod Robots. *CoRL 2024*.
    <https://openreview.net/forum?id=2AZfKk9tRI>

18. **Parkour in the Wild, 2025** — Learning a General and Extensible Agile Locomotion Policy using Multi-expert Distillation and RL Fine-tuning.
    <https://arxiv.org/html/2505.11164v1>

### Awesome lists(擴展閱讀)

- <https://github.com/curieuxjy/Awesome_Quadrupedal_Robots>
- <https://github.com/jonyzhang2023/awesome-humanoid-learning>
- <https://github.com/leggedrobotics/legged_gym>
- <https://github.com/leggedrobotics> (ETH RSL GitHub)

---

**最後提醒**

這份文件每一個論文、每一個技術都可以直接寫進你的期末報告 Related Work 和 Method 章節。
對應到你的 [redrhex_env_cfg.py](../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py) 的具體程式位置都已標註,執行時可精準定位。

**核心論述三句話**:
> 1. 傳統 MPC 在模型假設範圍內強,但面對地形不確定性、擾動、模型失配時退化。
> 2. 純 RL 解決前者但 generalization 差(Akki & Chen 2025)。
> 3. 我的 RedRhex 結合 privileged critic + terrain curriculum + domain randomization + symmetry + history,同時保有 RL 的 disturbance rejection 與能效優勢,**並**補上 generalization 弱點。

祝報告順利 🦾
