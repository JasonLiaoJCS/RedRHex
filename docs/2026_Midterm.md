# RedRHex 2026 期中報告

**日期**: 2026-04-16  
**主題**: RedRHex 彈簧腿機器人的節能獎勵重設計  
**範圍**: 將節能 reward 簡化為「馬達能耗 / 有效平移距離」，並將彈簧腿的儲能與釋能機制定位為理論背景與診斷指標，而非直接的 reward 目標。

---

## 1. 摘要

本次期中更新的核心，是把 RedRHex 的節能 reward，從「機構導向、機制導向」的複雜設計，改成「結果導向」的簡化設計。

先前的方向曾嘗試把彈簧相關機制直接寫進 reinforcement learning reward，例如：

- spring recovery
- spring utilization
- spring-specific gating
- 依照彈簧能量交換設計的額外 shaping

這些設計在物理直覺上是合理的，但也帶來一個重要問題：  
如果我們直接在 reward 裡明確鼓勵 policy「使用彈簧」，那麼最後就算 policy 真的學出了一種看起來很會利用彈簧的步態，我們也很難強而有力地宣稱，這是機器人自身結構優勢自然湧現的結果。因為也有可能只是 reward 明確要求它這麼做。

因此，本次 redesign 採取更簡單也更有說服力的原則：

1. 機器人首先必須完成 locomotion 任務。
2. 在能完成 locomotion 的前提下，它應該偏好「每移動一段有效距離所消耗的馬達能量更少」的策略。
3. 彈簧腿的優勢仍然存在於物理系統中，但不直接寫進 reward，而是讓 policy 自己去學出是否值得利用這個優勢。

也就是說，新的 reward 不再直接要求 policy 去最大化 spring release，而是只要求它最小化：

\[
\text{馬達能量消耗} \div \text{有效平移距離}
\]

如果 RedRHex 的彈簧腿機構真的有節能優勢，那麼 policy 在優化這個最終目標時，就應該自然學出較有效率的 gait。這樣的結果，會比直接把 spring term 寫進 reward 更具有研究與論證價值。

---

## 2. 本次修改背後的研究想法

### 2.1 原本的物理直覺仍然成立

本研究原本的理論出發點依然正確：

- RedRHex 具有帶有彈簧性質的腿部關節行為。
- 在 stance 與 release 的過程中，腿部形變所儲存的能量，有可能在之後回饋成推進身體前進的動能。
- 如果這個機制真的有效，它應該能降低機器人為了移動相同距離所需的主動馬達能量。

這些內容，依然是本專案的理論基礎。

### 2.2 為什麼原本的 reward 方向會太複雜

先前的 reward 設計想把這件事直接寫進學習目標，所以加入了像是：

- spring recovery ratio
- spring utilization
- spring gating
- spring-based power normalization

這樣的設計有三個主要問題。

第一，reward engineering 成本變高。  
reward 項越多，權重越多，解釋與調參就越複雜。

第二，容易產生 reward hacking。  
policy 可能學會提高某個 spring proxy，但不一定真的改善 task-level 的 locomotion efficiency。

第三，會削弱研究結論。  
如果 policy 的確展現出 spring-like gait，但這種 gait 本來就是 reward 明確要求的，那就很難說明這是機器人本體結構優勢自然浮現的結果。

### 2.3 本次 redesign 採取的核心原則

本次設計遵循一個很重要的研究原則：

> 如果我們相信某個機構可以提升性能，那麼最有說服力的驗證方式，不是直接獎勵那個機構，而是只優化最終可量測的性能指標，然後觀察該機構是否自然被利用。

在本研究裡，最終可量測的性能指標不是「spring release」，而是：

- 機器人實際消耗了多少馬達能量
- 它因此推進了多少有效距離

這個量不只物理意義明確，也更容易對應到模擬與真機共同可量測的資料。

---

## 3. 最終 reward 設計哲學

### 3.1 哪些內容保留為理論背景

本研究依然保留以下理論論述作為背景：

> RX / RedRHex 的彈簧腿結構，有可能在 gait cycle 中透過儲能與釋能機制，降低主動馬達為了推進身體所需付出的能量成本。

這句話說明了我們為什麼相信機器人應該能學出更節能的 gait。

### 3.2 哪些內容成為真正的 reward 目標

真正的 reward 目標簡化為：

> 每單位有效平移距離所消耗的馬達能量越低越好。

也就是說，我們不再直接告訴 policy：

- 要儲存多少彈簧能量
- 要釋放多少彈簧能量
- 要在什麼時刻利用彈簧
- 什麼樣的 damper deflection 才是好的

我們只告訴 policy：

- 要能夠移動
- 而且要用更少的能量完成有效移動

這樣的設計，才更符合「讓機構優勢自己浮現」的研究目標。

---

## 4. 數學形式化

### 4.1 馬達能量 proxy

在每個 step，我們用主動關節的機械功率近似馬達能量消耗：

\[
P_{\text{motor}} = \sum_i |\tau_i \omega_i|
\]

其中：

- \(\tau_i\) 是主動關節 torque
- \(\omega_i\) 是主動關節 angular velocity
- 取絕對值，是因為我們關心的是 actuation effort，而不是 signed work 的抵消

實作上，這個總功率包含：

- main drive joints
- ABAD joints

因此總機械功率為：

\[
P_{\text{total}} = P_{\text{main}} + P_{\text{abad}}
\]

### 4.2 有效平移進度

新的 reward 不再用 spring proxy 當分母，也不再直接用 spring reward，而是使用「沿著命令方向的有效平移進度」：

\[
v_{\text{progress}} = \max(0, \mathbf{v}_{xy} \cdot \hat{\mathbf{d}}_{\text{cmd}})
\]

其中：

- \(\mathbf{v}_{xy}\) 是實際平面速度
- \(\hat{\mathbf{d}}_{\text{cmd}}\) 是命令平移方向的單位向量
- 若進度是反方向，則 clamp 成 0

這正好對應本次設計想法：

- 如果機器人有耗能但沒有真正往目標方向前進，分母就很小，ratio 就會變大
- 如果機器人能用相同的馬達能耗推進更多距離，ratio 就會變小

### 4.3 為什麼 step-wise reward 仍然可視為 energy per distance

若每個控制步長為 \(\Delta t\)，則：

\[
E_{\text{step}} = P_{\text{total}} \Delta t
\]

而對應的位移為：

\[
d_{\text{step}} = v_{\text{progress}} \Delta t
\]

所以：

\[
\frac{E_{\text{step}}}{d_{\text{step}}}
=
\frac{P_{\text{total}} \Delta t}{v_{\text{progress}} \Delta t}
=
\frac{P_{\text{total}}}{v_{\text{progress}}}
\]

因此，在 step-based RL 中，直接使用：

\[
P_{\text{total}} / v_{\text{progress}}
\]

就等價於最小化每單位距離所需的能量。

### 4.4 最終 reward 項

本次實作採用的節能 reward 為：

\[
r_{\text{energy}} =
- w \cdot \tanh\left(
\frac{P_{\text{total}}}{v_{\text{progress}} + \varepsilon}
\cdot \frac{1}{s}
\right)
\]

其中：

- \(w = \texttt{power\_efficiency}\)
- \(\varepsilon = \texttt{power\_efficiency\_eps}\)
- \(s = \texttt{power\_efficiency\_tanh\_scale}\)

此外，這個 translational energy reward 只在命令確實要求平移時才啟用：

\[
\|\mathbf{v}_{\text{cmd}}\| > \texttt{energy\_min\_command\_motion}
\]

這樣可以避免 pure yaw 等情境被硬套進 distance-based 的節能目標。

---

## 5. 為什麼 spring 不再直接進 reward

### 5.1 關鍵論點

如果 spring 行為被直接獎勵，那 policy 的行為就有一部分是設計者明確指定的。

如果 spring 行為沒有被直接獎勵，但 policy 為了降低 energy per distance 而自然學出更會利用彈簧的 gait，那麼這個結果就更有價值：

- 它表示 spring 機構真的有用
- 它表示 morphology 本身參與了性能提升
- 它支持 RedRHex 結構優勢是真實存在的，而不是 reward 硬寫出來的

### 5.2 Spring 相關量的新角色

spring 相關量並不是沒用了，而是角色改變了。

它們現在的用途是：

- diagnostics
- interpretation
- post-training evidence

它們現在不再是：

- direct reward target
- learning objective
- 明確教 policy 如何使用 compliance 的 shaping term

### 5.3 訓練後的解讀方式

在訓練結束後，我們可以問：

- energy per distance 有沒有下降？
- spring release 或 spring recovery ratio 有沒有上升？
- policy 是不是在沒有被直接要求的情況下，自然學出了更具彈性回饋特性的 gait？

如果答案是肯定的，那麼研究結論會更有說服力。

---

## 6. 實際程式修改內容

本節逐一記錄這次已完成的實作修改。

### 6.1 檔案：`source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py`

#### 修改 A：簡化主節能 reward

原本 energy-aware 區塊中包含：

- mechanical power efficiency term
- spring recovery reward
- spring utilization reward
- spring reward gating

現在 reward 的邏輯改為：

1. 計算 active motor mechanical power。
2. 依照命令平移方向，計算 useful translational progress speed。
3. 計算 `energy_per_distance = total_mech_power / (useful_progress_speed + eps)`。
4. 用 `tanh` 把數值壓到穩定範圍。
5. 僅在命令確實要求平移時才啟用這個節能項。

#### 修改 B：Spring 項改成 diagnostics only

下列量仍然會被計算：

- `spring_energy`
- `spring_power`
- `spring_release`
- `spring_store`
- `spring_recovery_ratio`
- `spring_deflection_std`
- `damper_dissipation`

但是：

- `rew_spring_recovery` 現在固定為 0
- `rew_spring_utilization` 現在固定為 0
- `total_reward` 不再加入 spring-specific reward

#### 修改 C：新增更直接的紀錄欄位

這次新增了以下 episode logging：

- `rew_energy_per_distance`
- `diag_useful_progress_speed`
- `diag_energy_per_distance`

同時，為了不立刻破壞既有 TensorBoard 或下游分析腳本，舊欄位：

- `rew_power_efficiency`

仍然保留，但其語意現在等同於新的 `energy_per_distance` reward。

### 6.2 檔案：`source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`

#### 修改 A：簡化 reward scale 設定

節能 reward 區塊現在聚焦在：

- `power_efficiency`
- `power_efficiency_eps`
- `power_efficiency_tanh_scale`
- `torque_penalty`
- `torque_penalty_abad_weight`

已從 direct reward config 移除：

- `spring_recovery`
- `spring_recovery_eps`
- `spring_recovery_abad_weight`
- `spring_utilization`
- `spring_util_max_deflection`

#### 修改 B：重新說明 threshold 的意義

`energy_min_command_motion` 現在被明確定義為：

- 當命令平移速度太低時，不啟用平移型節能 reward
- 特別是 pure yaw 等模式，不應套用這個 distance-based objective

這樣 config 的語意就與新的 reward 設計一致。

#### 修改 C：同步更新 `ForwardFast` 設定

`RedrhexForwardFastEnvCfg` 也一起移除了舊的 spring reward scales，避免 fast training config 還保留過時設定。

### 6.3 檔案：`scripts/rsl_rl/eval_command_sweep.py`

#### 修改 A：評估腳本改成對齊新的 reward 目標

`collect_energy_metrics()` 現在新增計算：

- `progress_speed`
- `energy_per_distance`

而且是依照當前 translational command direction 來計算。  
也就是說，evaluation script 現在不只會輸出一般性的 motion-equivalent energy 指標，也會直接輸出訓練真正優化的那個核心量。

#### 修改 B：新增 command-level 與 skill-level 的 energy-per-distance 報表

Command sweep 現在會輸出：

- `energy_progress_speed_mean`
- `energy_per_distance`

同時仍保留：

- `energy_power_per_motion`

作為相容性欄位，但它現在應該被理解為新的 energy-per-distance 指標。

#### 修改 C：仍然保留 spring diagnostics

雖然 spring 已經不再是 reward 一部分，但 evaluation 仍保留：

- spring energy
- spring release power
- spring store power
- spring recovery ratio

這樣後續分析仍然可以回答一個很重要的研究問題：

> 當 energy per distance 下降時，是否同時伴隨更明顯的被動彈性能量回饋？

---

## 7. 為什麼這個設計更適合期中報告

### 7.1 它更簡單

新的 reward 可以很直接地被說明成：

- 機器人要能移動
- 而且要用更少的馬達能量完成有效位移

這比起一大串 spring-specific proxy reward，更適合在期中報告中清楚論述。

### 7.2 它更容易量測

新的目標比較容易對應到真機可量測量：

- 距離或速度，可由 locomotion sensor / kinematics / localization 取得
- 馬達能耗，可由 current、voltage、torque estimate 或 simulator torque-power proxy 估計

這讓整個設計更接近未來 sim-to-real 的驗證方式。

### 7.3 它保留了最強的研究論述

最重要的一點是：

- 如果 robot 在這個簡單 reward 下學出了更節能的 gait
- 而且 spring diagnostics 同時顯示出更明顯的被動儲能與釋能特徵

那麼我們就可以比較有力地主張：

- RedRHex 的彈簧腿結構本身確實提供了性能優勢

這遠比「因為 reward 直接叫它用 spring，所以它才用 spring」更有價值。

---

## 8. 預期行為效果

在新的 reward 下，policy 應該會偏好：

- 保持沿命令方向的有效平移進度
- 減少不必要的 torque 與 actuation 浪費
- 避免高能耗卻沒有實際推進的動作
- 只有在 compliance 真能降低能耗時，才自然學出更有效率的彈性 gait

因此，新 reward 也應該自然抑制：

- 高功率亂抖
- 原地耗能
- 表面上在刷 spring 指標，但實際沒有改善 locomotion economy 的行為

---

## 9. 訓練時建議觀察的 diagnostics

雖然 spring 不再直接進 reward，但以下指標依然非常重要。

### 9.1 與訓練目標直接對齊的指標

- `rew_energy_per_distance`
- `diag_energy_per_distance`
- `diag_useful_progress_speed`
- `diag_mech_power_total`
- `diag_cost_of_transport`

這些指標可以直接告訴我們，機器人是否真的變得更省能。

### 9.2 用來做研究解讀的指標

- `diag_spring_energy_total`
- `diag_spring_power_release`
- `diag_spring_power_store`
- `diag_spring_recovery_ratio`
- `diag_spring_deflection_std`
- `diag_damper_dissipation`

這些指標用來回答：

> 當 energy per distance 下降時，是否也出現了更明顯的 spring-assisted locomotion 特徵？

---

## 10. 建議的實驗論述方式

若要在期中報告中把故事講得更完整，建議後續比較：

1. 不加 simplified energy term 的 baseline。
2. 加入 simplified `energy_per_distance` reward 的版本。
3. 比較兩者在以下指標上的差異：
   - task success / tracking quality
   - mechanical power
   - energy per distance
   - CoT proxy
   - spring diagnostics

最有說服力的結果會是：

- tracking 維持或接近 baseline
- energy per distance 下降
- spring recovery diagnostics 上升
- 而且這些 spring 指標不是被 reward 直接指定出來的

這樣就能支持一個很強的研究結論：

> RedRHex 的彈簧腿 morphology 確實有助於 locomotion economy，而且這個優勢可以在不直接獎勵 spring 的情況下自然浮現。

---

## 11. 期中結論

本次更新的本質，是把 RedRHex 的節能設計，從「機制導向的複雜 reward」轉成「結果導向的簡潔 reward」。

最核心的一句話是：

> 我們不需要直接獎勵機器人「使用彈簧」。  
> 我們只需要獎勵機器人「以更低的馬達能耗完成有效位移」。

如果 RedRHex 的彈簧腿結構真的有優勢，那 policy 應該會在這個簡單 objective 下，自然學出更有效率的 gait。

這就是這次 redesign 對 2026 期中的真正意義：

- 它降低了 reward engineering 的複雜度
- 它強化了研究論證的可信度
- 它提升了結果的可解釋性
- 它讓未來的 sim-to-real 驗證更有一致性

---

## 12. 本次已修改檔案

- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`
- `scripts/rsl_rl/eval_command_sweep.py`
- `docs/2026_Midterm.md`
