# RedRhex 彈簧腿節能 Reward 整合報告

**日期**: 2026-03-31  
**作者**: Claude (AI Assistant)  
**專案**: RedRHex — 六足 Wheg+ABAD 機器人 RL 訓練  
**分支**: `Fast-forward`

---

## 目錄

1. [背景與目標](#1-背景與目標)
2. [物理模型推導](#2-物理模型推導)
3. [Sim-to-Real 感測對照](#3-sim-to-real-感測對照)
4. [節能 Reward 設計](#4-節能-reward-設計)
5. [程式碼修改清單](#5-程式碼修改清單)
6. [權重調校建議](#6-權重調校建議)
7. [消融實驗與驗證流程](#7-消融實驗與驗證流程)
8. [風險與注意事項](#8-風險與注意事項)
9. [附錄：完整 Diff](#9-附錄完整-diff)

---

## 1. 背景與目標

### 1.1 研究問題

RedRhex 是一台六足 wheg（wheel-leg）機器人，每條腿由三個關節組成：

| 關節群組 | 數量 | 控制模式 | 功能 |
|---------|------|---------|------|
| **main_drive** | 6 | 速度控制（連續旋轉） | 產生前進推力 |
| **ABAD** | 6 | 位置控制 | 外展/內收，調整步態 |
| **damper** | 6 | 被動（彈簧-阻尼器） | 緩衝、儲能、釋能 |

其中 **damper 關節**（避震關節）是物理彈簧腿的核心——它們不是由馬達驅動，而是由扭轉彈簧（k=200 N·m/rad）和阻尼器（d=20 N·m·s/rad）提供被動力。

### 1.2 目標

在現有 RL 訓練管線中整合**節能獎勵**（Energy-aware rewards），使得：

1. 機器人學會利用彈簧腿的**儲能-釋能循環**，而非純靠馬達蠻力推進
2. 降低有源關節的機械功率消耗
3. 提高 Cost of Transport (CoT) 效率
4. **不犧牲**原有的命令追蹤能力和穩定性

### 1.3 設計原則

```
優先級順序：生存 > 命令追蹤 > 防作弊 > 節能 > 彈簧活用
權重比例：追蹤(~9.0) >> 節能(~1.0)
```

---

## 2. 物理模型推導

### 2.1 基礎量

**關節角速度** ω：直接從 `joint_vel` 讀取（Isaac Lab 的 ArticulationData 提供）

**關節力矩** τ：
- 優先使用 `robot.data.applied_torque`（Isaac Lab 物理引擎回報）
- Fallback: 根據 ImplicitActuator PD 模型重建
  - main_drive: `τ = k_d × (ω_target - ω)`，其中 k_d ≈ 50.0
  - ABAD: `τ = k_p × (θ_target - θ) + k_d × (0 - ω)`

**機械功率** P：

$$P = \sum_{i} |\tau_i \cdot \omega_i|$$

使用絕對值是因為我們關心的是總能量消耗，無論方向。

**能量消耗** E：

$$E = \int_0^T |P(t)| \, dt \approx \sum_{t} |P(t)| \cdot \Delta t$$

### 2.2 彈簧-阻尼器模型

damper 關節建模為**扭轉彈簧-阻尼器**：

$$\tau_{spring} = -k \cdot \Delta\theta - d \cdot \omega$$

其中：
- `k = 200.0 N·m/rad`（扭轉彈簧剛度）
- `d = 20.0 N·m·s/rad`（阻尼係數）
- `Δθ = θ_current - θ_rest`（相對於靜止位的偏轉角）

**彈簧位能**：

$$E_{spring} = \frac{1}{2} k \cdot \Delta\theta^2$$

**彈簧功率**（位能變化率）：

$$\dot{E}_{spring} = \frac{d}{dt}\left(\frac{1}{2}k\Delta\theta^2\right) = k \cdot \Delta\theta \cdot \dot{\theta}$$

- **$\dot{E}_{spring} > 0$**：彈簧在蓄能（stance phase，地面反力壓縮彈簧）
- **$\dot{E}_{spring} < 0$**：彈簧在釋能（swing phase，位能轉化為動能輔助推進）

**阻尼耗散功率**（不可逆）：

$$P_{dissipation} = d \cdot \omega^2$$

### 2.3 效率指標

**Mechanical Cost of Transport (mCoT)**:

$$\text{mCoT} = \frac{\sum |P|}{m \cdot g \cdot v}$$

其中 m=14.0 kg, g=9.81 m/s²。

**Power-per-Speed Proxy**（用於 reward 的連續化替代指標）：

$$\text{PPS} = \frac{\sum |P_{active}|}{v + \epsilon}$$

使用 ε=0.1 防止靜止時除零。

**Spring Recovery Ratio**（彈簧能量回收率）：

$$\eta_{spring} = \frac{P_{release}}{P_{main} + \epsilon}$$

其中 $P_{release} = \sum \max(-\dot{E}_{spring,i}, 0)$

---

## 3. Sim-to-Real 感測對照

| 物理量 | Isaac Lab (Sim) | 實機 (Real) | 備註 |
|--------|----------------|------------|------|
| 關節角度 θ | `joint_pos` | 編碼器 | 直接可用 |
| 關節角速度 ω | `joint_vel` | 編碼器差分 / 反電動勢 | 可能需濾波 |
| 關節力矩 τ | `applied_torque` | 電流感測 × 力矩常數 | 需校準 |
| 機身線速度 | `root_lin_vel_b` | IMU 積分 / 視覺里程計 | 積分漂移問題 |
| 機身角速度 | `root_ang_vel_b` | IMU 陀螺儀 | 直接可用 |
| 機身姿態 | `root_quat_w` | IMU 融合 | 可用 |
| 彈簧偏轉 Δθ | `joint_pos[damper] - rest` | damper 編碼器 | 需知 rest angle |
| 彈簧角速度 | `joint_vel[damper]` | damper 編碼器差分 | 可能需濾波 |
| 接地力 | ContactSensor (未啟用) | 力感測器 / 電流激增 | sim 中目前用運動學推斷 |

### Sim-to-Real 風險

1. **力矩估計**：sim 中 `applied_torque` 是精確的；real 中需從電流推算，存在摩擦和效率損失
2. **彈簧參數**：sim 中 k=200 是精確值；real 中彈簧可能有非線性、疲勞、溫度效應
3. **速度估計**：sim 中 `root_lin_vel_b` 是精確的；real 中需要 state estimator

### 建議感測方案

| 方案 | 精度 | 可行性 | 適用場景 |
|------|------|--------|---------|
| **理想方案**: 全感測 | 高 | 成本高 | 研究原型 |
| **實用方案**: 電流 + 編碼器 + IMU | 中 | 成本適中 | 多數場景 |
| **降級方案**: 僅編碼器 + IMU | 低 | 最便宜 | 初步驗證 |

---

## 4. 節能 Reward 設計

### 4.1 Reward 清單

共新增 4 個 reward term + 9 個 diagnostic metric：

#### E1: Power Efficiency（有源機械功率效率）

```
rew = -tanh( sum(|τ*ω|) / (v + ε) / 500 ) × w × healthy_gate
```

- **意義**：每單位速度的功率消耗越低越好
- **歸一化**：`tanh(x/500)` 將典型值（~250）映射到 [0, 1]
- **權重**：w = 0.3（default）/ 0.15（ForwardFast）
- **healthy_gate**：只有活著才計算，防止「死了省電」的 reward hacking

#### E2: Spring Recovery（彈簧能量回收）

```
rew = clamp( spring_release / (P_main + ε), 0, 1 ) × w × healthy_gate
```

- **意義**：彈簧釋放的能量佔主驅動功率的比例越高越好
- **clamp [0,1]**：防止回收率超過 100% 的不合理值
- **權重**：w = 0.4（default）/ 0.2（ForwardFast）

#### E3: Spring Utilization（彈簧活用度）

```
rew = clamp( std(Δθ) / max_deflection, 0, 1 ) × w × healthy_gate
```

- **意義**：6 個彈簧偏轉的標準差越大，表示彈簧被動態使用（非全部靜止）
- **max_deflection**：0.3 rad（~17°），用於歸一化
- **權重**：w = 0.2（default）/ 0.1（ForwardFast）

#### E4: Torque Penalty（力矩平方懲罰）

```
rew = ( sum(τ²_main) + β × sum(τ²_abad) ) × w
```

- **意義**：抑制極端力矩輸出
- **β = 0.5**：ABAD 力矩懲罰較輕（因其本身力矩較小）
- **權重**：w = -0.0001（很輕微，只防極端）

### 4.2 Diagnostic Metrics（僅記錄，不影響 reward）

| Metric | 公式 | 用途 |
|--------|------|------|
| `diag_mech_power_main` | Σ\|τ_main × ω_main\| | 主驅動功率 |
| `diag_mech_power_abad` | Σ\|τ_abad × ω_abad\| | ABAD 功率 |
| `diag_spring_energy_total` | Σ(½kΔθ²) | 總彈簧位能 |
| `diag_spring_power_release` | Σmax(-ĖÇspring, 0) | 彈簧釋能功率 |
| `diag_spring_power_store` | Σmax(Ėspring, 0) | 彈簧蓄能功率 |
| `diag_damper_dissipation` | d × Σω² | 阻尼耗散 |
| `diag_spring_deflection_std` | std(Δθ across 6 dampers) | 彈簧偏轉離散度 |
| `diag_cost_of_transport` | P/(m·g·v) | CoT |
| `diag_torque_rms_main` | √mean(τ²_main) | 主驅動力矩 RMS |

### 4.3 Reward Hacking 分析與防護

| 作弊策略 | 防護機制 |
|---------|---------|
| 停下來不動以省電 | `healthy_gate` + 速度追蹤 reward 遠大於節能 |
| 故意抖動彈簧刷 recovery | `action_smooth` 懲罰 + clamp 上限 |
| 走極慢以提高效率 | 速度追蹤權重(~9.0) >> 節能(~1.0) |
| 極端力矩瞬間衝刺 | `torque_penalty` 壓制 |
| 軀體傾倒觸發彈簧 | `fall_penalty` + `healthy_gate` |

---

## 5. 程式碼修改清單

共修改 **3 個檔案**，新增 **213 行**程式碼。

### 5.1 `redrhex_env_cfg.py` (+40 行)

**檔案路徑**: `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`

#### 修改 1: `RedrhexEnvCfg.v2_reward_scales` 新增節能 reward 鍵值

**位置**: `yaw_tracking_sigma` 之後（約 line 1574）

新增 8 個 reward scale 參數：

```python
# 節能 reward（Energy-aware rewards）
"power_efficiency": 0.3,
"power_efficiency_eps": 0.1,
"spring_recovery": 0.4,
"spring_recovery_eps": 0.01,
"spring_utilization": 0.2,
"spring_util_max_deflection": 0.3,
"torque_penalty": -0.0001,
"torque_penalty_abad_weight": 0.5,
```

#### 修改 2: 新增物理常數

**位置**: `v2_reward_scales` dict 結束後

```python
damper_stiffness = 200.0    # N·m/rad
damper_damping = 20.0       # N·m·s/rad
robot_mass_kg = 14.0        # kg
```

#### 修改 3: `RedrhexForwardFastEnvCfg.v2_reward_scales` 新增節能 reward（保守權重）

**位置**: 檔案末尾

ForwardFast 變體使用 **50% 權重**，避免在純前進模式中過度優化節能而影響速度：

```python
"power_efficiency": 0.15,     # 50% of default
"spring_recovery": 0.2,       # 50% of default
"spring_utilization": 0.1,    # 50% of default
"torque_penalty": -0.00005,   # 50% of default
```

### 5.2 `redrhex_env.py` (+117 行)

**檔案路徑**: `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py`

#### 修改 1: `_setup_buffers()` 新增彈簧物理常數

**位置**: `_damper_initial_pos` 初始化之後（line ~275）

```python
self._spring_k = float(getattr(self.cfg, 'damper_stiffness', 200.0))
self._spring_d = float(getattr(self.cfg, 'damper_damping', 20.0))
self._robot_mass = float(getattr(self.cfg, 'robot_mass_kg', 14.0))
```

使用 `getattr` 帶 fallback 確保向後相容。

#### 修改 2: `episode_sums` 新增 13 個追蹤鍵

**位置**: `diag_stance_count` 之後（line ~450）

新增 4 個 reward sum + 9 個 diagnostic sum。

#### 修改 3: `_compute_simplified_rewards()` 插入節能 reward 計算

**位置**: R8 (action_smooth) 與 R9 (fall_penalty) 之間（line ~2323）

新增約 60 行核心計算邏輯，包含：

1. **讀取 damper 關節狀態** → damper_pos, damper_vel, damper_deflection
2. **讀取有源關節力矩** → main_torques, abad_torques（含 fallback）
3. **E1 計算** → mech_power → power_per_speed → tanh 歸一化 → 乘權重
4. **E2 計算** → spring_energy → spring_power → spring_release → recovery_ratio
5. **E3 計算** → spring_deflection_std → 歸一化
6. **E4 計算** → torque_sq → 乘權重
7. **Diagnostic 計算** → 9 個指標

#### 修改 4: Episode sum 累加

**位置**: reward 累加區域（line ~2457）

新增 4 + 9 = 13 個 episode_sums 累加語句。

### 5.3 `eval_command_sweep.py` (+59 行)

**檔案路徑**: `scripts/rsl_rl/eval_command_sweep.py`

#### 修改 1: 新增 7 個 KPI 累加器

**位置**: `energy_count` 之後（line ~460）

```python
spring_energy_sum = 0.0
spring_release_sum = 0.0
spring_store_sum = 0.0
mech_power_main_sum = 0.0
cot_proxy_sum = 0.0
damper_dissipation_sum = 0.0
energy_kpi_count = 0
```

#### 修改 2: Per-step 彈簧/能量數據收集

**位置**: `energy_count += 1` 之後（line ~685）

從 unwrapped environment 讀取 damper 關節數據，計算：
- spring_energy, spring_release, spring_store
- damper_dissipation
- mech_power_main, cot_proxy

#### 修改 3: 終端輸出新增 7 行能量 KPI

```
energy.spring_energy_mean(J)
energy.spring_release_power_mean(W)
energy.spring_store_power_mean(W)
energy.damper_dissipation_mean(W)
energy.mech_power_main_mean(W)
energy.cost_of_transport_proxy
energy.spring_recovery_ratio
```

#### 修改 4: CSV summary 新增 7 個 metric rows

在 CSV 輸出中加入上述 7 個能量指標。

---

## 6. 權重調校建議

### 6.1 初始值

| 參數 | Default | ForwardFast | 說明 |
|------|---------|-------------|------|
| power_efficiency | 0.3 | 0.15 | 功率效率 reward 權重 |
| spring_recovery | 0.4 | 0.2 | 彈簧回收 reward 權重 |
| spring_utilization | 0.2 | 0.1 | 彈簧活用度 reward 權重 |
| torque_penalty | -0.0001 | -0.00005 | 力矩懲罰（負值）|

### 6.2 調校順序

1. **先跑 baseline** — 將所有節能 reward 權重設為 0，確認追蹤性能不退化
2. **啟用 torque_penalty** — 最安全的 reward，不太會壞事
3. **啟用 power_efficiency** — 觀察 TensorBoard 中 `diag_mech_power_main` 是否下降，且速度追蹤不掉
4. **啟用 spring_recovery** — 觀察 `diag_spring_power_release` 是否上升
5. **最後啟用 spring_utilization** — 這個 reward 最容易導致奇怪步態

### 6.3 判斷標準

- **成功**：mCoT 下降 > 10%，且命令追蹤 pass_ratio 不掉超過 5%
- **需調整**：mCoT 下降但追蹤也下降 → 降低節能權重
- **失敗**：機器人學到「站著不動」或「原地抖動」→ 檢查 healthy_gate 和追蹤權重

---

## 7. 消融實驗與驗證流程

### 7.1 四版本消融設計

| 版本 | power_eff | spring_rec | spring_util | torque_pen | 目的 |
|------|-----------|-----------|-------------|------------|------|
| **A: Baseline** | 0 | 0 | 0 | 0 | 對照組 |
| **B: +Torque** | 0 | 0 | 0 | -0.0001 | 測試力矩懲罰單獨效果 |
| **C: +Power** | 0.3 | 0 | 0 | -0.0001 | 測試功率效率 |
| **D: Full** | 0.3 | 0.4 | 0.2 | -0.0001 | 完整節能系統 |

### 7.2 評估指標

每個版本需觀察的 TensorBoard 曲線：

**必看 reward 曲線**:
- `rew_power_efficiency`
- `rew_spring_recovery`
- `rew_spring_utilization`
- `rew_torque_penalty`

**必看 diagnostic 曲線**:
- `diag_mech_power_main` — 應逐步下降
- `diag_spring_power_release` — 應逐步上升
- `diag_cost_of_transport` — 核心效率指標
- `diag_spring_deflection_std` — 彈簧使用程度

**追蹤性能曲線（不可退化）**:
- `rew_vx_tracking`, `rew_vy_tracking`, `rew_yaw_tracking`
- eval_command_sweep 的 `command_pass_ratio`

### 7.3 eval_command_sweep 評估

訓練後用 eval_command_sweep 跑 multi-command 測試，比較：

```bash
python scripts/rsl_rl/eval_command_sweep.py --task Template-Redrhex-Direct-v0 \
    --load_run <version_A_or_D> --num_envs 64
```

重點比較：
- `energy.cost_of_transport_proxy` — D < A 表示真正更高效
- `energy.spring_recovery_ratio` — D > A 表示彈簧在被利用
- `acceptance.command_pass_ratio` — D ≈ A 表示追蹤能力未受損

### 7.4 「真正更高效」vs「只是更慢」的判斷

**真正更高效的標準**:
1. `mCoT` 下降（同速度下功率更低）
2. `command_pass_ratio` 持平或更好
3. `average_speed` 未顯著下降

**只是更慢**的信號:
1. 功率下降但速度也下降，mCoT 不變
2. `command_pass_ratio` 下降
3. 機器人傾向走低速指令

---

## 8. 風險與注意事項

### 8.1 假設與限制

| 假設 | 風險 | 緩解 |
|------|------|------|
| 彈簧為線性扭轉彈簧 | Real 彈簧可能非線性 | 可加 domain randomization |
| k=200, d=20 與 actuator cfg 一致 | 若 cfg 改了需同步 | 已用 `getattr` + fallback |
| `applied_torque` 可用 | Isaac Lab 版本差異 | 已實作 fallback model |
| 14kg 整機質量 | 加裝感測器後會變 | 可在 cfg 調整 |

### 8.2 潛在問題

1. **Reward hacking 風險**：雖已設防護，但 RL agent 可能找到意想不到的漏洞。建議訓練初期多看 TensorBoard 影片確認行為合理
2. **數值穩定性**：所有除法都有 ε 保護，tanh/clamp 確保有界
3. **Sim-to-Real gap**：彈簧 reward 依賴精確的物理模擬參數；transfer 時可能需要重新標定
4. **ForwardFast 權重**：初始設為 50%，可能需要根據實際訓練結果調整

### 8.3 向後相容性

- 所有新增參數都有 `getattr(..., default)` fallback
- 若 `v2_reward_scales` 中沒有節能 key，對應 reward 權重為 0 → 行為等同修改前
- episode_sums 新增 key 不影響既有 logging 邏輯
- eval_command_sweep 的新增 KPI 以 `hasattr` 保護，不影響無節能版本的 eval

---

## 9. 附錄：完整 Diff

### 9.1 修改統計

```
 scripts/rsl_rl/eval_command_sweep.py               |  59 ++++++++++++
 .../RedRhex/tasks/direct/redrhex/redrhex_env.py    | 117 +++++++++++++++++++-
 .../tasks/direct/redrhex/redrhex_env_cfg.py        |  40 +++++++
 3 files changed, 213 insertions(+), 3 deletions(-)
```

### 9.2 語法驗證

所有三個修改後的檔案均通過 Python `ast.parse()` 語法驗證，無語法錯誤。

### 9.3 關鍵程式碼位置

| 修改 | 檔案 | 起始行 |
|------|------|--------|
| 節能 reward scales | redrhex_env_cfg.py | ~1574 |
| 物理常數 | redrhex_env_cfg.py | ~1610 |
| ForwardFast 節能權重 | redrhex_env_cfg.py | ~1743 |
| 彈簧常數初始化 | redrhex_env.py | ~275 |
| episode_sums 新 key | redrhex_env.py | ~450 |
| 節能 reward 計算 | redrhex_env.py | ~2323 |
| episode_sums 累加 | redrhex_env.py | ~2457 |
| 能量 KPI 累加器 | eval_command_sweep.py | ~460 |
| per-step 彈簧計算 | eval_command_sweep.py | ~685 |
| KPI 輸出 | eval_command_sweep.py | ~900 |
| CSV 匯出 | eval_command_sweep.py | ~973 |

---

## 總結

本次整合為 RedRhex 訓練管線新增了完整的**彈簧腿節能 reward 系統**，包含：

- **4 個 reward terms**：功率效率、彈簧回收、彈簧活用度、力矩懲罰
- **9 個 diagnostic metrics**：可在 TensorBoard 觀察的能量相關指標
- **7 個 eval KPIs**：在 command sweep 評估時輸出的能量效率指標
- **完整的防 reward hacking 機制**：healthy_gate、clamp、tanh 歸一化
- **向後相容設計**：所有新增功能都有 fallback，不破壞現有行為

下一步建議：執行四版本消融實驗（A/B/C/D），根據 TensorBoard 曲線和 eval 結果調整權重。
