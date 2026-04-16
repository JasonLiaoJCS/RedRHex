# RedRhex 彈簧腿節能 RL Reward 整合報告

**日期**: 2026-03-31  
**作者**: Codex (AI Research Assistant)  
**專案**: RedRHex — 六足 Wheg + ABAD + 被動彈簧/阻尼關節機器人  
**環境**: `Template-Redrhex-Direct-v0` / `Template-Redrhex-ForwardFast-Direct-v0`  
**文件狀態**: 已對齊目前 repo 中實際完成的程式實作  

> 更新註記（2026-04-16）：目前 reward 設計方向已改為以結果為導向的
> `馬達能耗 / 有效距離` 目標。最新、與期中報告對齊的說明與程式修改摘要，
> 請參考 `docs/2026_Midterm.md`。

---

## 目錄

1. 研究問題重述
2. 物理與數學推導
3. 模擬 / 真機感測映射
4. Reward 設計
5. `redrhex_env_cfg.py` 修改版
6. `redrhex_env.py` 修改版
7. `eval_command_sweep.py` 修改版
8. 權重調參建議
9. Ablation 與驗證流程
10. 風險與注意事項

---

## 1. 研究問題重述

### 1.1 RedRhex 的關節架構與物理角色

RedRhex 是六足 wheg 機器人，每條腿有三種物理角色不同的關節：

| 關節群組 | 數量 | 控制型式 | Isaac Lab actuator 參數 | 物理角色 |
|---|---:|---|---|---|
| `main_drive` | 6 | 速度控制 | `stiffness=0`, `damping=50`, `effort_limit=100` | 連續旋轉 wheg，產生主推進 |
| `ABAD` | 6 | 位置控制 | `stiffness=40`, `damping=4`, `effort_limit=8` | 外展/內收，調整側向落點、姿態與轉向 |
| `damper` | 6 | 被動/高剛性 | `stiffness=200`, `damping=20`, `effort_limit=50` | 吸震、儲能、釋能 |

對應關節群組在程式中的索引：
- `self._main_drive_indices`
- `self._abad_indices`
- `self._damper_indices`

### 1.2 核心研究問題

本研究不是只讓 policy 學會「能走」，而是讓 policy 在以下四件事之間取得平衡：

1. 追蹤命令速度與方向。
2. 保持穩定，不摔倒，不靠亂抖、亂滑、亂轉偷分。
3. 降低主動關節的有源能耗。
4. 真正利用 damper/spring 關節在 stance-swing 轉換中的儲能與釋能效果。

核心目標可表述為：

> 讓 RedRhex 學出「穩定、可追蹤、低有源功率、且有實際彈簧回能利用」的 locomotion policy。

### 1.3 為什麼不能只靠一般 locomotion reward

若 reward 只看 tracking，policy 很容易學成：
- 主驅動暴力輸出
- ABAD 大幅擺動補償
- 不在乎 damper/spring 是否真的幫忙

若 reward 只看 torque 或 power，policy 又容易學成：
- 幾乎不動
- 速度嚴重不足
- 靠降低任務完成度換取能耗下降

因此本專案採用的不是單一節能項，而是：

```text
生存 / 穩定  >  命令追蹤  >  反作弊  >  能耗效率  >  彈簧活用
```

這是整份設計的最高原則。

---

## 2. 物理與數學推導

### 2.1 關節角度與角速度

對每一條腿定義：
- `q_m`: main drive 關節角
- `q_a`: ABAD 關節角
- `q_s`: spring/damper 關節角

角速度定義為：

$$
\omega(t) = \dot q(t) = \frac{dq}{dt}
$$

對應到三類關節：
- `omega_main = dq_m/dt`
- `omega_abad = dq_a/dt`
- `omega_spring = dq_s/dt`

**物理意義**：
- `main_drive` 的角速度直接對應 wheg 旋轉速度，是主推進來源。
- `ABAD` 的角速度反映腿在 body lateral direction 的調整速率。
- `damper` 的角速度反映彈簧壓縮/回彈速度，與 spring power 直接相關。

### 2.2 關節力矩

#### 2.2.1 Main drive：速度控制器

目前 RedRhex 的 `main_drive` 是 implicit actuator，控制律可近似為：

$$
\tau_{m,i} \approx b_m(\omega_{m,i}^{cmd} - \omega_{m,i})
$$

其中：
- $b_m = 50$
- effort limit = 100 N·m

因此程式中的 fallback 估計為：

$$
\tau_{m,i}^{est} = \text{clip}\left(50(\omega_{m,i}^{cmd} - \omega_{m,i}), -100, 100\right)
$$

#### 2.2.2 ABAD：位置控制器

`ABAD` 近似為 PD 位置控制：

$$
\tau_{a,i} \approx k_a(q_{a,i}^{cmd} - q_{a,i}) - b_a\omega_{a,i}
$$

其中：
- $k_a = 40$
- $b_a = 4$
- effort limit = 8 N·m

程式中的 fallback 估計為：

$$
\tau_{a,i}^{est}
=
\text{clip}\left(40(q_{a,i}^{cmd} - q_{a,i}) - 4\omega_{a,i}, -8, 8\right)
$$

#### 2.2.3 Spring/damper：被動關節

對 damper/spring 關節，主要關注的是內部儲能，而不是把它當作有源 actuator reward 來源。

彈簧力矩：

$$
\tau_{s,i}^{spring} = -k_s(q_{s,i} - q_{s,i}^{rest})
$$

阻尼力矩：

$$
\tau_{s,i}^{damp} = -d_s\omega_{s,i}
$$

總被動反力矩：

$$
\tau_{s,i}^{passive} = -k_s(q_{s,i} - q_{s,i}^{rest}) - d_s\omega_{s,i}
$$

目前 cfg 中使用：
- $k_s = 200$ N·m/rad
- $d_s = 20$ N·m·s/rad

### 2.3 瞬時機械功率

任一關節的瞬時機械功率：

$$
P_i = \tau_i\omega_i
$$

為了估計「消耗」而不是淨功，reward 與 KPI 主要使用絕對值版本：

$$
|P_i| = |\tau_i\omega_i|
$$

並分開計算：

$$
P_{main} = \sum_{i\in main} |\tau_i\omega_i|
$$

$$
P_{abad} = \sum_{j\in abad} |\tau_j\omega_j|
$$

$$
P_{act} = P_{main} + P_{abad}
$$

### 2.4 單步與單回合機械能

每一 simulation step 的能量 proxy：

$$
E_{step} = P_{act}\Delta t
$$

單回合累積：

$$
E_{episode} = \sum_t P_{act}(t)\Delta t
$$

由於 RL reward 是 step-wise，真正放進 reward 的不是 $E_{episode}$，而是與速度正規化後的 step-wise power proxy。

### 2.5 電流、電功率與電能估計

若真機可取得馬達常數，可由 joint torque 估電流：

$$
\tau_{joint} = N\eta_gK_t I
\Rightarrow
I = \frac{\tau_{joint}}{N\eta_gK_t}
$$

電功率可分為兩個主要部分：

1. 銅損：

$$
P_{copper} = I^2R
$$

2. 機械輸出功率與效率折算：

$$
P_{mech,out} = \tau\omega
$$

$$
P_{elec} \approx I^2R + \frac{\tau\omega}{\eta_m\eta_d}
$$

因此即使沒有完整電機常數，仍可用兩個很重要的 proxy：
- $|\tau\omega|$：機械輸出功率 proxy
- $\tau^2$：銅損 proxy

這就是本專案同時保留 `power_efficiency` 與 `torque_penalty` 的原因。

### 2.6 彈簧位能、彈簧功率與阻尼耗散

令 damper 偏轉為：

$$
\Delta q_{s,i} = q_{s,i} - q_{s,i}^{rest}
$$

則彈簧位能：

$$
E_{s,i} = \frac{1}{2}k_s(\Delta q_{s,i})^2
$$

彈簧位能變化率：

$$
\dot E_{s,i} = k_s\Delta q_{s,i}\dot q_{s,i}
$$

若：
- $\dot E_{s,i} > 0$：代表儲能
- $\dot E_{s,i} < 0$：代表釋能

定義：

$$
P_{store} = \sum_i \max(\dot E_{s,i}, 0)
$$

$$
P_{release} = \sum_i \max(-\dot E_{s,i}, 0)
$$

阻尼耗散功率：

$$
P_{diss} = d_s\sum_i \omega_{s,i}^2
$$

阻尼是不可逆損耗，因此：
- `P_release` 是有可能幫助 locomotion 的正面量
- `P_diss` 是只會耗掉能量的量

### 2.7 CoT、回收效率與單位運動能耗

#### 2.7.1 等效運動速度

本次實作引入 yaw-aware 的等效速度：

$$
v_{eq} = \sqrt{v_x^2 + v_y^2 + (r_{yaw}\omega_z)^2}
$$

其中：
- $r_{yaw} = 0.18$ m

這是為了讓 pure yaw 命令不會被誤判成「零速高耗能」。

#### 2.7.2 Cost of Transport proxy

$$
CoT^* = \frac{P_{act}}{mg(v_{eq} + \epsilon)}
$$

這不是精確真機 CoT，而是 sim/eval 中可穩定比較的 proxy。

#### 2.7.3 Spring recovery ratio

本專案中實際拿來做 reward 的 spring 回收比率為：

$$
\eta_{rec}
=
\text{clamp}\left(
\frac{P_{release}}{P_{main} + \alpha P_{abad} + \epsilon},
0, 1
\right)
$$

其中 $\alpha = 0.35$。

#### 2.7.4 Spring utilization

$$
U_s = \text{std}(\Delta q_{s,1}, \ldots, \Delta q_{s,6})
$$

這是步態結構 proxy，不是直接能量收益。

### 2.8 哪些適合 reward、哪些只適合 diagnostics

| 指標 | 適合 reward | 適合 diagnostics | 理由 |
|---|---|---|---|
| `P_act / v_eq` | 是 | 是 | 直接反映單位運動輸出的有源功率 |
| `eta_rec` | 是 | 是 | 反映彈簧釋能對有源功率的相對貢獻 |
| `spring utilization` | 是 | 是 | 是結構性 proxy，但需低權重使用 |
| `tau^2` | 是 | 是 | 對應銅損 proxy，且能抑制暴力輸出 |
| `E_episode` | 否 | 是 | 對 episode 長度敏感，不適合 step reward |
| `P_store` / `P_release` | 否 | 是 | 單看其中一個都可能被 hacking |
| `P_diss` | 否 | 是 | 偏向分析，不建議直接回饋 |
| 真實 `P_elec` | 視感測而定 | 是 | 真機若無完整 motor constants，不宜直接進 reward |

---

## 3. 模擬 / 真機感測映射

### 3.1 Isaac Lab / Isaac Sim 中可取得的量

| 物理量 | Ideal | Practical | Fallback / Proxy |
|---|---|---|---|
| 關節角度 `q` | `self.joint_pos[:, idx]` | 同 ideal | 無 |
| 關節角速度 `omega` | `self.joint_vel[:, idx]` | 同 ideal | 差分估計 |
| 主動關節力矩 `tau` | `robot.data.applied_torque` | `computed_torque` / `joint_torque` 若版本不同 | controller-based fallback |
| base 位置/速度 | `root_pos_w`, `base_lin_vel`, `base_ang_vel` | 同 ideal | 無 |
| damper rest pose | `cfg.robot_cfg.init_state.joint_pos` | 同 ideal | 手動標定 |
| 接觸資訊 | 真正 contact sensor | phase-based `_current_leg_in_stance` | `_contact_count` |

目前 repo 中為了版本相容，實作的安全讀取順序是：

1. `applied_torque`
2. `computed_torque`
3. `joint_torque`
4. 若都沒有，再用控制器模型 fallback

### 3.2 真機中的對應方式

| 物理量 | Ideal sensing | Practical sensing | Fallback proxy |
|---|---|---|---|
| `q` | 高解析 encoder | 一般 encoder | 低頻視覺/marker |
| `omega` | encoder + estimator | encoder 差分 + 低通/Kalman | IMU/視覺輔助估 |
| `tau` | torque transducer | motor current + $K_t$ | 控制器模型 + 誤差校正 |
| `v_x, v_y, omega_z` | motion capture / state estimator | IMU + odometry fusion | 單 IMU 近似 |
| `Delta q_s` | damper encoder | linkage 幾何回推 | chassis-leg relative displacement proxy |
| GRF / contact | force plate / load cell | foot switch / current spike / contact pad | gait phase proxy |

### 3.3 力矩估計的三層方案

#### Ideal

若真機直接有 torque sensor，則：

$$
\tau = \tau_{measured}
$$

#### Practical

若有 motor current：

$$
\tau_{joint} = N\eta_gK_t I
$$

#### Fallback

若沒有 torque sensor 也沒有 current，則只能退回控制器模型：

- main drive:
  $$
  \tau_m^{est} \approx b_m(\omega_m^{cmd} - \omega_m)
  $$
- ABAD:
  $$
  \tau_a^{est} \approx k_a(q_a^{cmd} - q_a) - b_a\omega_a
  $$

### 3.4 接觸與彈簧壓縮量的 proxy 風險

本 repo 目前沒有把真正 foot contact sensor 接進節能 reward，因此：
- `spring_release` 與 `spring_store` 的 contact gating 仍是 **phase-based contact proxy**
- 這很適合 sim 內快速疊代 reward
- 但不能把它當成真實 GRF 驗證

因此教授報告時要明講：

> 本研究目前是「energy-aware reward engineering with physically informed proxies」，不是完整 force-sensing locomotion framework。

---

## 4. Reward 設計

### 4.1 設計哲學：平衡型 reward

本設計不是單純最小化 torque，也不是單純最大化 tracking，而是：

```text
tracking / stability 必須先成立
energy term 只能在此基礎上做二階優化
spring term 只能在真的有 locomotion 輸出時給正面鼓勵
```

### 4.2 目前實作的四個節能相關 reward 項

#### E1. Power Efficiency

$$
r_{E1}
=
-\tanh\left(
\frac{P_{act}}{(v_{eq}+\epsilon_v)s_{tanh}}
\right)
\cdot w_{power}
\cdot \mathbb{1}_{healthy}
$$

目前參數：
- `power_efficiency = 0.3`
- `power_efficiency_eps = 0.1`
- `power_efficiency_tanh_scale = 500.0`

#### E2. Spring Recovery

$$
r_{E2}
=
\eta_{rec}\cdot w_{rec}
\cdot \mathbb{1}_{healthy}
\cdot \mathbb{1}_{cmd}
\cdot g_{track}
$$

其中：

$$
\eta_{rec}
=
\text{clamp}\left(
\frac{P_{release}}{P_{main} + \alpha P_{abad} + \epsilon_{rec}},
0,1
\right)
$$

目前參數：
- `spring_recovery = 0.4`
- `spring_recovery_eps = 0.01`
- `spring_recovery_abad_weight = 0.35`
- `energy_min_command_motion = 0.05`

#### E3. Spring Utilization

$$
r_{E3}
=
\text{clamp}\left(
\frac{\text{std}(\Delta q_s)}{\Delta q_{max}},
0,1
\right)
\cdot w_{util}
\cdot \mathbb{1}_{healthy}
\cdot \mathbb{1}_{cmd}
\cdot g_{track}
$$

目前參數：
- `spring_utilization = 0.2`
- `spring_util_max_deflection = 0.3`

#### E4. Torque Penalty

$$
r_{E4}
=
\left(
\sum \tau_m^2 + \beta \sum \tau_a^2
\right)w_{\tau}
$$

目前參數：
- `torque_penalty = -0.0001`
- `torque_penalty_abad_weight = 0.5`

### 4.3 正向 spring reward 的 gating 機制

這次實作真正補強的重點不是只有公式，而是 gating：

```text
spring_reward_gate
= healthy_gate
 * cmd_motion_gate
 * motion_tracking_gate
```

其含義：
- `healthy_gate`: 沒摔倒、姿態合理
- `cmd_motion_gate`: 真的有 locomotion 命令
- `motion_tracking_gate`: policy 至少做出一部分對應的實際運動

因此 policy 不能靠：
- 原地抖彈簧
- 空中甩腿
- 不追蹤命令卻刷 spring 指標

### 4.4 diagnostics 與 TensorBoard 指標

目前 env 會記錄：
- `rew_power_efficiency`
- `rew_spring_recovery`
- `rew_spring_utilization`
- `rew_torque_penalty`
- `diag_mech_power_main`
- `diag_mech_power_abad`
- `diag_mech_power_total`
- `diag_spring_energy_total`
- `diag_spring_power_release`
- `diag_spring_power_store`
- `diag_spring_recovery_ratio`
- `diag_damper_dissipation`
- `diag_spring_deflection_std`
- `diag_cost_of_transport`
- `diag_motion_speed_equiv`
- `diag_cmd_motion_speed_equiv`
- `diag_torque_rms_main`

### 4.5 Reward hacking 風險與緩解

| 風險 | 可能作弊方式 | 緩解機制 |
|---|---|---|
| 不動省電 | 停在原地減少 power / torque | tracking reward、stall penalty、leg_moving |
| 只慢慢走 | 壓低速度換低 CoT | `diag_motion_speed_equiv` 與 tracking quality 一起比較 |
| 抖 damper 刷 spring 分 | 高頻 oscillation | `spring_reward_gate`、action smoothness、stance mask |
| yaw 被誤罰 | pure yaw 時 $v \to 0$ | 改用 `v_eq` |
| ABAD 功率被低估 | fallback 沒有真 target | 新增 `_target_abad_pos` cache |

---

## 5. `redrhex_env_cfg.py` 修改版

**檔案**: `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`

### 5.1 `v2_reward_scales` 新增的節能權重

目前 default config 中與節能相關的正式欄位為：

```python
"power_efficiency": 0.3,
"power_efficiency_eps": 0.1,
"power_efficiency_tanh_scale": 500.0,
"spring_recovery": 0.4,
"spring_recovery_eps": 0.01,
"spring_recovery_abad_weight": 0.35,
"spring_utilization": 0.2,
"spring_util_max_deflection": 0.3,
"torque_penalty": -0.0001,
"torque_penalty_abad_weight": 0.5,
```

### 5.2 新增的物理常數與估測參數

```python
damper_stiffness = 200.0
damper_damping = 20.0
robot_mass_kg = 14.0
energy_velocity_yaw_radius = 0.18
energy_min_command_motion = 0.05
main_drive_torque_estimate_damping = 50.0
main_drive_torque_estimate_limit = 100.0
abad_torque_estimate_stiffness = 40.0
abad_torque_estimate_damping = 4.0
abad_torque_estimate_limit = 8.0
```

### 5.3 ForwardFast 的保守版本

`RedrhexForwardFastEnvCfg` 中使用較保守的權重：

```python
"power_efficiency": 0.15,
"power_efficiency_eps": 0.1,
"power_efficiency_tanh_scale": 500.0,
"spring_recovery": 0.2,
"spring_recovery_eps": 0.01,
"spring_recovery_abad_weight": 0.35,
"spring_utilization": 0.1,
"spring_util_max_deflection": 0.3,
"torque_penalty": -0.00005,
"torque_penalty_abad_weight": 0.5,
```

原因：
- ForwardFast 的主要任務是先把 forward locomotion 穩定下來。
- 節能項先用 50% 左右強度比較保守。

---

## 6. `redrhex_env.py` 修改版

**檔案**: `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py`

### 6.1 `_setup_buffers()` 中新增的內部狀態

除了舊版已存在的：
- `self._spring_k`
- `self._spring_d`
- `self._robot_mass`

目前額外新增：

```python
self._energy_yaw_radius
self._energy_min_cmd_motion
self._main_drive_torque_estimate_damping
self._main_drive_torque_estimate_limit
self._abad_torque_estimate_stiffness
self._abad_torque_estimate_damping
self._abad_torque_estimate_limit
self._target_abad_pos
```

### 6.2 新 helper：`_compute_energy_equivalent_speed()`

功能：
- 將 `vx, vy, wz` 轉換成等效速度 `v_eq`
- 供 reward 與 diagnostics 使用

核心邏輯：

```python
yaw_equiv = self._energy_yaw_radius * yaw_rate
v_eq = sqrt(vx^2 + vy^2 + yaw_equiv^2)
```

### 6.3 新 helper：`_get_active_joint_torques()`

功能：
- 優先讀 simulator 的真 torque tensor
- 若欄位不存在，自動退回控制器估測

讀取順序：
1. `applied_torque`
2. `computed_torque`
3. `joint_torque`
4. fallback estimate

這是為了處理不同 Isaac Lab 版本 API 差異。

### 6.4 `_apply_action()` 中新增 `self._target_abad_pos`

這個改動雖然小，但對 ABAD torque fallback 很重要：

```python
self._target_abad_pos = target_abad_pos.clone()
```

沒有這個 cache，就不能正確估：

$$
\tau_a^{est} = k_a(q_a^{cmd} - q_a) - b_a\omega_a
$$

### 6.5 `_compute_simplified_rewards()` 的正式節能流程

目前 reward 核心流程為：

1. 取 `damper_pos`, `damper_vel`, `damper_deflection`
2. 取 `abad_vel`
3. 用 `_get_active_joint_torques()` 得到 `main_torques`, `abad_torques`
4. 算 `mech_power_main`, `mech_power_abad`, `total_mech_power`
5. 算 `actual_motion_speed` 與 `cmd_motion_speed`
6. 計算 `rew_power_efficiency`
7. 算 `spring_energy`, `spring_power`
8. 用 `_current_leg_in_stance` 當 contact mask，得到 `spring_release`, `spring_store`
9. 算 `spring_reward_gate`
10. 計算 `rew_spring_recovery`
11. 計算 `rew_spring_utilization`
12. 計算 `rew_torque_penalty`
13. 計算 `spring_recovery_ratio`, `cot_proxy`, `torque_rms_main`
14. 寫入 `episode_sums`

### 6.6 實際新增的 diagnostics

除了舊版草稿列到的指標，目前實作還新增：

```python
diag_mech_power_total
diag_spring_recovery_ratio
diag_motion_speed_equiv
diag_cmd_motion_speed_equiv
```

這些是教授報告裡很重要的量，因為它們可以回答：
- 變省能是不是只是速度變慢？
- spring 指標上升是不是有真實 locomotion 輸出支撐？

### 6.7 `_reset_idx()` 的補強

現在 reset 時也會清掉：

```python
self._target_drive_vel[env_ids] = 0.0
self._target_abad_pos[env_ids] = 0.0
self._base_velocity[env_ids] = 0.0
```

避免上一回合殘留 target 污染下一回合前幾步的 torque fallback。

---

## 7. `eval_command_sweep.py` 修改版

**檔案**: `scripts/rsl_rl/eval_command_sweep.py`

### 7.1 新增 `collect_energy_metrics()`

目前 `eval_command_sweep.py` 不再只看 tracking / success / fall，而是新增 helper：

```python
collect_energy_metrics(
    unwrapped_env,
    actual_vx,
    actual_vy,
    actual_wz,
)
```

輸出：
- `motion_speed`
- `mech_power_main`
- `mech_power_abad`
- `mech_power_total`
- `cot_proxy`
- `spring_energy`
- `spring_release`
- `spring_store`
- `spring_recovery_ratio`
- `damper_dissipation`

### 7.2 每個 command 的 energy KPI

每個 command 現在都會記錄：

```python
energy_mech_power_main_mean
energy_mech_power_total_mean
energy_cost_of_transport_proxy
energy_spring_energy_mean
energy_spring_release_power_mean
energy_spring_store_power_mean
energy_spring_recovery_ratio
energy_motion_speed_mean
energy_power_per_motion
```

### 7.3 每個 skill 的彙總

除了 per-command，現在還會按 skill 匯總：
- `forward`
- `lateral`
- `diagonal`
- `yaw`

輸出在 terminal 的：
- `=== Energy By Command ===`
- `=== Skill-level Energy Summary ===`

以及 summary CSV 的：
- `energy.skill.forward.*`
- `energy.skill.lateral.*`
- `energy.skill.diagonal.*`
- `energy.skill.yaw.*`

### 7.4 為什麼這個 evaluation 很關鍵

因為現在可以區分：

1. `P_total` 降了，但 `motion_speed_mean` 也大降  
   - 代表只是變慢，不是真正更有效率。

2. `CoT proxy` 降了，`tracking_quality` 持平，`success_ratio` 持平  
   - 代表這是有意義的能效提升。

3. `spring_recovery_ratio` 升了，但 `fall_rate` 升了  
   - 代表 policy 可能在刷 spring 指標，而不是學到更穩定步態。

---

## 8. 權重調參建議

### 8.1 初始權重建議

| 項目 | Default | ForwardFast | 調參原則 |
|---|---:|---:|---|
| `power_efficiency` | 0.30 | 0.15 | tracking 穩後再拉高 |
| `spring_recovery` | 0.40 | 0.20 | 比 `spring_utilization` 更重要 |
| `spring_utilization` | 0.20 | 0.10 | 小權重，避免刷偏轉 |
| `torque_penalty` | -1e-4 | -5e-5 | 只抑制極端輸出 |
| `torque_penalty_abad_weight` | 0.50 | 0.50 | ABAD 力矩量級較小 |

### 8.2 推薦調整順序

1. 先只開 baseline tracking/stability，確認能走。
2. 加 `torque_penalty`，抑制暴力輸出。
3. 加 `power_efficiency`，看 `diag_mech_power_total` 與 `diag_cost_of_transport` 是否下降。
4. 最後再加 `spring_recovery` 與 `spring_utilization`。

### 8.3 常見症狀與對應調法

| 症狀 | 可能原因 | 建議調法 |
|---|---|---|
| 速度掉很多但 CoT 沒明顯變好 | `power_efficiency` 太重 | 先降 `power_efficiency` |
| spring 指標沒變化 | spring reward 太弱或 tracking 太差 | 先改善 tracking，再加 `spring_recovery` |
| ABAD 很暴力 | ABAD 被低估成本 | 拉高 `torque_penalty_abad_weight` |
| 原地抖動 | spring reward 被刷 | 降 `spring_utilization`，檢查 action_smooth |

---

## 9. Ablation 與驗證流程

### 9.1 四個版本

建議比較四個版本：

- A: baseline
- B: baseline + torque penalty
- C: baseline + power efficiency
- D: baseline + power efficiency + spring recovery/utilization

### 9.2 TensorBoard 必看曲線

節能相關：
- `Episode_Reward/rew_power_efficiency`
- `Episode_Reward/rew_spring_recovery`
- `Episode_Reward/rew_spring_utilization`
- `Episode_Reward/rew_torque_penalty`
- `Episode_Reward/diag_mech_power_total`
- `Episode_Reward/diag_cost_of_transport`
- `Episode_Reward/diag_spring_recovery_ratio`
- `Episode_Reward/diag_motion_speed_equiv`
- `Episode_Reward/diag_cmd_motion_speed_equiv`

任務相關：
- `Episode_Reward/rew_tracking`
- `Episode_Reward/rew_mode`
- `Episode_Reward/rew_fall`
- `Episode_Reward/diag_vel_error`
- `Episode_Termination/terminated`

### 9.3 `eval_command_sweep.py` 的正式比較方式

對 A/B/C/D 四個版本都跑相同 profile，至少比較：

1. `tracking_quality`
2. `success_ratio`
3. `fall_rate`
4. `energy_mech_power_total_mean`
5. `energy_cost_of_transport_proxy`
6. `energy_spring_recovery_ratio`
7. `energy_power_per_motion`

### 9.4 多模式分析

應分 skill 檢查：
- `forward`
- `lateral`
- `diagonal`
- `yaw`

理由：
- 很多節能 reward 在 forward 看起來有效，但會誤傷 lateral 或 yaw。
- 只有 mode-wise 檢查，才能知道 reward 是否 truly balanced。

### 9.5 如何判斷「真的更省能」而不是「只是變慢」

至少要同時滿足：

1. `tracking_quality` 不明顯下降
2. `success_ratio` 不明顯下降
3. `fall_rate` 不惡化
4. `energy_cost_of_transport_proxy` 下降
5. `energy_power_per_motion` 下降

若只有第 4、5 點改善，但第 1、2 點明顯變差，則不能聲稱真的更高效。

---

## 10. 風險與注意事項

### 10.1 物理假設

目前模型假設：
- damper 是線性扭轉彈簧
- 阻尼是線性黏滯阻尼
- `k=200`, `d=20` 與 actuator config 一致
- yaw 等效半徑可用固定 `0.18 m` 近似

這些都屬於「物理合理但仍為近似」。

### 10.2 感測與 sim2real 限制

- sim 中 torque 可精確取得，真機通常不能。
- 真機若只有 encoder + IMU，則 power/CoT 很多量只能是 proxy。
- 目前 contact gating 仍是 phase-based，不是 force-sensor based。

### 10.3 Reward engineering 限制

- `spring_utilization` 本質上是結構 proxy，不是直接能量收益。
- `CoT proxy` 是比較指標，不是真實電池放電功率。
- `tau^2` 是銅損 proxy，但未顯式考慮摩擦、gear hysteresis 與 driver switching loss。

### 10.4 目前實作驗證狀態

本次已完成：
- `redrhex_env_cfg.py` 實作
- `redrhex_env.py` 實作
- `eval_command_sweep.py` 實作
- `python -m py_compile` 靜態語法檢查通過

本次尚未在這個 shell 內直接完成：
- Isaac Lab runtime 實際 rollout
- 真機量測驗證

### 10.5 建議下一步

1. 用 A/B/C/D 四版跑 `eval_command_sweep.py`
2. 先確認 `forward` 模式下 `P_total` 與 `CoT` 的方向正確
3. 再檢查 `lateral`、`diagonal`、`yaw` 是否被新 reward 誤傷
4. 若一切穩定，再考慮把真機 current-based power estimation 接進 evaluation pipeline

---

## 附錄 A：本次實際完成的程式修改

已修改檔案：

1. `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`
2. `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py`
3. `scripts/rsl_rl/eval_command_sweep.py`
4. `docs/energy_aware_reward_integration_report.md`

其中節能核心程式修改包括：
- 新 config 權重與物理常數
- torque fallback helper
- yaw-aware equivalent speed
- spring reward gating
- ABAD target cache
- energy diagnostics
- per-command / per-skill evaluation KPI

## 附錄 B：關鍵程式碼位置索引

| 項目 | 檔案 | 位置 |
|---|---|---|
| `power_efficiency_tanh_scale`, `spring_recovery_abad_weight` | `redrhex_env_cfg.py` | 約 1584 |
| `energy_velocity_yaw_radius`, torque fallback cfg | `redrhex_env_cfg.py` | 約 1609 |
| `_compute_energy_equivalent_speed()` | `redrhex_env.py` | 約 1386 |
| `_get_active_joint_torques()` | `redrhex_env.py` | 約 1395 |
| `_target_abad_pos` cache | `redrhex_env.py` | 約 1943 |
| reward 中 `power_per_motion` / `spring_reward_gate` | `redrhex_env.py` | 約 2404 |
| `diag_mech_power_total`, `diag_motion_speed_equiv` | `redrhex_env.py` | 約 476, 2549 |
| `collect_energy_metrics()` | `eval_command_sweep.py` | 約 363 |
| per-command energy KPI | `eval_command_sweep.py` | 約 835 |
| `Energy By Command` / `Skill-level Energy Summary` | `eval_command_sweep.py` | 約 988, 1010 |

## 附錄 C：教授報告時可直接強調的三個重點

1. **這不是只加一個 torque penalty。**  
   本研究把有源功率、彈簧釋能、彈簧活用度、以及 mode-aware evaluation 全部整合進 RL framework。

2. **這不是只靠概念性 proxy。**  
   目前 reward 與 evaluation 已經實際寫進 Isaac Lab 環境，並考慮了 torque 欄位版本差異、ABAD target cache、以及 pure yaw 的能耗正規化問題。

3. **這不是只追求低功率。**  
   全部 energy KPI 都和 tracking / success / fall-rate 一起檢驗，用來區分「真正更高效」與「只是變慢」。

---

## 附錄 D：Overleaf 可直接使用的 LaTeX 原稿

優先建議直接使用 repo 內已整理好的獨立檔案：`docs/energy_aware_reward_integration_report.tex`。  
請在 Overleaf 設定中把 Compiler 改成 **XeLaTeX**；不要直接拿這份 `.md` 當成 LaTeX 編譯。  
如果你是從下面的 code block 手動複製，請確認不要把最外層的 ````tex` 與結尾 ``` 一起貼進 `main.tex`。

```tex
% !TeX program = XeLaTeX
\documentclass[12pt,a4paper,UTF8]{ctexart}

\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs,longtable,array,multirow}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{setspace}
\usepackage{float}
\usepackage{fontspec}

\IfFontExistsTF{Noto Serif CJK TC}{
    \setCJKmainfont{Noto Serif CJK TC}
    \setCJKsansfont{Noto Sans CJK TC}
}{
    \IfFontExistsTF{Noto Serif CJK SC}{
        \setCJKmainfont{Noto Serif CJK SC}
        \setCJKsansfont{Noto Sans CJK SC}
    }{
        \setCJKmainfont{FandolSong-Regular}
        \setCJKsansfont{FandolHei-Regular}
    }
}

\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{Latin Modern Mono}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!60!black,
    urlcolor=blue!60!black,
    citecolor=blue!60!black
}

\lstdefinestyle{pythonstyle}{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue!70!black},
    commentstyle=\color{green!40!black},
    stringstyle=\color{red!60!black},
    showstringspaces=false,
    breaklines=true,
    frame=single,
    columns=fullflexible
}

\title{RedRhex 彈簧腿節能導向強化學習 Reward 整合報告}
\author{Codex Research Assistant}
\date{2026-03-31}

\begin{document}
\maketitle
\tableofcontents
\newpage

\section{研究問題重述}

\subsection{系統背景}

本研究對象為 RedRhex 六足 wheg 機器人。每條腿包含三種功能不同的關節：

\begin{enumerate}[leftmargin=2em]
    \item \textbf{main drive 關節}：速度控制，負責 wheg 旋轉與主要推進。
    \item \textbf{ABAD 關節}：位置控制，負責外展/內收與側向落點調整。
    \item \textbf{damper/spring-like 關節}：被動關節，負責吸震、儲能與釋能。
\end{enumerate}

目前使用 Isaac Lab 環境：
\begin{itemize}[leftmargin=2em]
    \item \texttt{Template-Redrhex-Direct-v0}
    \item \texttt{Template-Redrhex-ForwardFast-Direct-v0}
\end{itemize}

主要目標不是只讓策略學會「能走」，而是讓策略學會：
\begin{enumerate}[leftmargin=2em]
    \item 追蹤命令速度；
    \item 保持穩定與不摔倒；
    \item 避免暴力輸出與 reward hacking；
    \item 降低有源能耗；
    \item 實際利用彈簧腿儲能與釋能優勢。
\end{enumerate}

\subsection{核心研究問題}

本研究要回答的核心問題為：

\begin{quote}
如何把 RedRhex 的被動彈簧腿機制，正式整合進 Isaac Lab 強化學習 reward，使策略不僅能 locomote，還能在保持穩定與追蹤性能的前提下，學會更低能耗且更具彈簧回能特性的步態？
\end{quote}

\section{物理與數學推導}

\subsection{關節角度、角速度與力矩}

對每條腿定義：
\begin{align}
q_m &: \text{main drive 關節角}, \\
q_a &: \text{ABAD 關節角}, \\
q_s &: \text{spring/damper 關節角}.
\end{align}

角速度定義為
\begin{equation}
\omega = \dot{q} = \frac{dq}{dt}.
\end{equation}

\subsection{Main drive 力矩模型}

對 main drive 關節，Isaac Lab implicit actuator 可近似為速度控制器：
\begin{equation}
\tau_{m,i} \approx b_m \left( \omega^{cmd}_{m,i} - \omega_{m,i} \right),
\end{equation}
其中本專案採用
\begin{equation}
b_m = 50, \qquad |\tau_{m,i}| \le 100 \text{ N$\cdot$m}.
\end{equation}

因此 fallback 力矩估計為
\begin{equation}
\tau^{est}_{m,i} =
\mathrm{clip}\left(
50(\omega^{cmd}_{m,i} - \omega_{m,i}), -100, 100
\right).
\end{equation}

\subsection{ABAD 力矩模型}

對 ABAD 關節，採位置控制近似：
\begin{equation}
\tau_{a,i} \approx k_a(q^{cmd}_{a,i} - q_{a,i}) - b_a \omega_{a,i},
\end{equation}
其中
\begin{equation}
k_a = 40, \qquad b_a = 4, \qquad |\tau_{a,i}| \le 8 \text{ N$\cdot$m}.
\end{equation}

因此 fallback 估計為
\begin{equation}
\tau^{est}_{a,i} =
\mathrm{clip}\left(
40(q^{cmd}_{a,i} - q_{a,i}) - 4\omega_{a,i}, -8, 8
\right).
\end{equation}

\subsection{被動彈簧與阻尼模型}

令 damper 偏轉為
\begin{equation}
\Delta q_{s,i} = q_{s,i} - q^{rest}_{s,i}.
\end{equation}

則彈簧力矩與阻尼力矩可寫為
\begin{align}
\tau^{spring}_{s,i} &= -k_s \Delta q_{s,i}, \\
\tau^{damp}_{s,i} &= -d_s \omega_{s,i},
\end{align}
其中目前設計使用
\begin{equation}
k_s = 200 \text{ N$\cdot$m/rad}, \qquad d_s = 20 \text{ N$\cdot$m$\cdot$s/rad}.
\end{equation}

\subsection{機械功率與機械能}

單一關節的瞬時機械功率為
\begin{equation}
P_i = \tau_i \omega_i.
\end{equation}

由於本研究希望估計有源輸出成本，而非淨做功，因此使用絕對值：
\begin{align}
P_{main} &= \sum_{i \in main} |\tau_i \omega_i|, \\
P_{abad} &= \sum_{j \in abad} |\tau_j \omega_j|, \\
P_{act} &= P_{main} + P_{abad}.
\end{align}

每一步機械能 proxy 為
\begin{equation}
E_{step} = P_{act}\Delta t,
\end{equation}
單回合則為
\begin{equation}
E_{episode} = \sum_t P_{act}(t)\Delta t.
\end{equation}

\subsection{電流與電功率估計}

若真機可量到 motor current，則可由
\begin{equation}
\tau_{joint} = N \eta_g K_t I
\end{equation}
得
\begin{equation}
I = \frac{\tau_{joint}}{N \eta_g K_t}.
\end{equation}

電功率可近似為
\begin{equation}
P_{elec} \approx I^2R + \frac{\tau \omega}{\eta_m \eta_d}.
\end{equation}

因此即使不知道完整電機常數，仍可用兩個重要 proxy：
\begin{enumerate}[leftmargin=2em]
    \item $|\tau \omega|$：機械功率 proxy；
    \item $\tau^2$：銅損 proxy。
\end{enumerate}

\subsection{彈簧位能、釋能與耗散}

彈簧位能為
\begin{equation}
E_{s,i} = \frac{1}{2}k_s(\Delta q_{s,i})^2.
\end{equation}

位能變化率為
\begin{equation}
\dot{E}_{s,i} = k_s \Delta q_{s,i} \dot{q}_{s,i}.
\end{equation}

定義儲能與釋能功率為
\begin{align}
P_{store} &= \sum_i \max(\dot{E}_{s,i}, 0), \\
P_{release} &= \sum_i \max(-\dot{E}_{s,i}, 0).
\end{align}

阻尼耗散功率為
\begin{equation}
P_{diss} = d_s \sum_i \omega_{s,i}^2.
\end{equation}

\subsection{等效運動速度與 CoT proxy}

本研究為了讓 yaw 模式也能合理做能耗比較，引入等效運動速度
\begin{equation}
v_{eq} = \sqrt{v_x^2 + v_y^2 + (r_{yaw}\omega_z)^2},
\end{equation}
其中
\begin{equation}
r_{yaw} = 0.18 \text{ m}.
\end{equation}

定義 CoT proxy 為
\begin{equation}
CoT^* = \frac{P_{act}}{mg(v_{eq} + \epsilon)}.
\end{equation}

\subsection{彈簧回收率與活用度}

本研究實際使用的 spring recovery ratio 為
\begin{equation}
\eta_{rec}
=
\mathrm{clamp}\left(
\frac{P_{release}}{P_{main} + \alpha P_{abad} + \epsilon_{rec}},
0,1
\right),
\end{equation}
其中
\begin{equation}
\alpha = 0.35.
\end{equation}

彈簧活用度則採用六條腿偏轉的標準差：
\begin{equation}
U_s = \mathrm{std}(\Delta q_{s,1}, \ldots, \Delta q_{s,6}).
\end{equation}

\section{模擬 / 真機感測映射}

\subsection{模擬中可直接取得的量}

\begin{longtable}{p{3cm}p{4cm}p{6cm}}
\toprule
物理量 & 模擬中理想讀取方式 & 備註 \\
\midrule
關節角度 $q$ & \texttt{self.joint\_pos[:, idx]} & 直接由 articulation data 提供 \\
關節角速度 $\omega$ & \texttt{self.joint\_vel[:, idx]} & 直接可用 \\
力矩 $\tau$ & \texttt{applied\_torque} / \texttt{computed\_torque} / \texttt{joint\_torque} & 視 Isaac Lab 版本而定 \\
base 線速度 & \texttt{self.base\_lin\_vel} & 可直接用於 tracking 與 CoT \\
base 角速度 & \texttt{self.base\_ang\_vel} & yaw 模式很重要 \\
damper rest pose & \texttt{cfg.robot\_cfg.init\_state.joint\_pos} & 用於計算 $\Delta q_s$ \\
接觸資訊 & \texttt{\_current\_leg\_in\_stance} & 目前為 phase-based proxy \\
\bottomrule
\end{longtable}

\subsection{真機感測的三層方案}

\subsubsection*{Ideal sensing}
\begin{itemize}[leftmargin=2em]
    \item encoder 量測 $q$；
    \item torque transducer 量測 $\tau$；
    \item motion capture 或 state estimator 量測 $(v_x,v_y,\omega_z)$；
    \item foot force plate 或 load cell 量測 GRF/contact。
\end{itemize}

\subsubsection*{Practical sensing}
\begin{itemize}[leftmargin=2em]
    \item encoder 差分估 $\omega$；
    \item motor current 轉 torque；
    \item IMU + odometry fusion 估 base velocity；
    \item 利用 linkage 幾何回推 damper 偏轉。
\end{itemize}

\subsubsection*{Fallback proxy}
\begin{itemize}[leftmargin=2em]
    \item 若無 torque sensor 與 current，退回 controller-based torque estimate；
    \item 若無真實 contact sensor，退回 gait phase/contact proxy；
    \item 若無彈簧位移量測，退回相對幾何壓縮 proxy。
\end{itemize}

\section{Reward 設計}

\subsection{設計哲學}

本設計遵循以下優先順序：
\begin{equation}
\text{stability} > \text{tracking} > \text{anti-cheat} > \text{energy} > \text{spring use}.
\end{equation}

也就是說：
\begin{itemize}[leftmargin=2em]
    \item 先保命；
    \item 再追蹤；
    \item 最後才對能效與彈簧利用做二階優化。
\end{itemize}

\subsection{E1: Power Efficiency}

\begin{equation}
r_{E1}
=
-\tanh\left(
\frac{P_{act}}{(v_{eq}+\epsilon_v)s_{tanh}}
\right)
\cdot w_{power}
\cdot \mathbb{1}_{healthy}
\end{equation}

本專案目前設定：
\begin{align}
w_{power} &= 0.3, \\
\epsilon_v &= 0.1, \\
s_{tanh} &= 500.0.
\end{align}

\subsection{E2: Spring Recovery}

\begin{equation}
r_{E2}
=
\eta_{rec}\cdot w_{rec}
\cdot \mathbb{1}_{healthy}
\cdot \mathbb{1}_{cmd}
\cdot g_{track}
\end{equation}

其中：
\begin{itemize}[leftmargin=2em]
    \item $\mathbb{1}_{cmd}$：命令等效速度超過門檻；
    \item $g_{track}$：實際等效速度相對命令等效速度的達成比例；
    \item 並且僅在 stance contact proxy 成立的腿上計入 $P_{release}$ 與 $P_{store}$。
\end{itemize}

\subsection{E3: Spring Utilization}

\begin{equation}
r_{E3}
=
\mathrm{clamp}\left(
\frac{U_s}{\Delta q_{max}}, 0, 1
\right)
\cdot w_{util}
\cdot \mathbb{1}_{healthy}
\cdot \mathbb{1}_{cmd}
\cdot g_{track}
\end{equation}

其中
\begin{equation}
\Delta q_{max} = 0.3 \text{ rad}.
\end{equation}

\subsection{E4: Torque Penalty}

\begin{equation}
r_{E4}
=
\left(
\sum \tau_m^2 + \beta \sum \tau_a^2
\right) w_{\tau}
\end{equation}
其中
\begin{equation}
\beta = 0.5, \qquad w_{\tau} = -10^{-4}.
\end{equation}

\subsection{為何需要 spring reward gating}

若只獎勵 spring release 或 spring utilization，策略可能學會：
\begin{itemize}[leftmargin=2em]
    \item 原地抖動 damper；
    \item 空中甩腿刷彈簧偏轉；
    \item 不追蹤命令，卻拿到正面 spring reward。
\end{itemize}

因此目前正式實作採用
\begin{equation}
\texttt{spring\_reward\_gate}
=
\texttt{healthy\_gate}
\cdot
\texttt{cmd\_motion\_gate}
\cdot
\texttt{motion\_tracking\_gate}.
\end{equation}

\section{模擬實作與程式修改}

\subsection{\texttt{redrhex\_env\_cfg.py} 中新增的參數}

\begin{lstlisting}[style=pythonstyle,caption={節能 reward 權重與物理常數}]
"power_efficiency": 0.3,
"power_efficiency_eps": 0.1,
"power_efficiency_tanh_scale": 500.0,
"spring_recovery": 0.4,
"spring_recovery_eps": 0.01,
"spring_recovery_abad_weight": 0.35,
"spring_utilization": 0.2,
"spring_util_max_deflection": 0.3,
"torque_penalty": -0.0001,
"torque_penalty_abad_weight": 0.5,

damper_stiffness = 200.0
damper_damping = 20.0
robot_mass_kg = 14.0
energy_velocity_yaw_radius = 0.18
energy_min_command_motion = 0.05
main_drive_torque_estimate_damping = 50.0
main_drive_torque_estimate_limit = 100.0
abad_torque_estimate_stiffness = 40.0
abad_torque_estimate_damping = 4.0
abad_torque_estimate_limit = 8.0
\end{lstlisting}

\subsection{\texttt{redrhex\_env.py} 中新增的 helper}

\begin{lstlisting}[style=pythonstyle,caption={等效運動速度 helper}]
def _compute_energy_equivalent_speed(self, lin_xy, yaw_rate):
    yaw_equiv = self._energy_yaw_radius * yaw_rate
    return torch.sqrt(torch.sum(torch.square(lin_xy), dim=1) + torch.square(yaw_equiv))
\end{lstlisting}

\begin{lstlisting}[style=pythonstyle,caption={主動關節力矩讀取與 fallback}]
def _get_active_joint_torques(self, main_drive_vel, abad_vel):
    for attr_name in ("applied_torque", "computed_torque", "joint_torque"):
        candidate = getattr(self.robot.data, attr_name, None)
        if isinstance(candidate, torch.Tensor) and candidate.ndim == 2:
            return candidate[:, self._main_drive_indices], candidate[:, self._abad_indices]

    main_torques = self._main_drive_torque_estimate_damping * (self._target_drive_vel - main_drive_vel)
    main_torques = torch.clamp(main_torques, -self._main_drive_torque_estimate_limit, self._main_drive_torque_estimate_limit)

    abad_pos_err = self._target_abad_pos - self.joint_pos[:, self._abad_indices]
    abad_torques = self._abad_torque_estimate_stiffness * abad_pos_err - self._abad_torque_estimate_damping * abad_vel
    abad_torques = torch.clamp(abad_torques, -self._abad_torque_estimate_limit, self._abad_torque_estimate_limit)
    return main_torques, abad_torques
\end{lstlisting}

\subsection{ABAD 目標位置 cache}

\begin{lstlisting}[style=pythonstyle,caption={ABAD target cache}]
target_abad_pos = torch.clamp(target_abad_pos, min=-abad_limit, max=abad_limit)
self._target_abad_pos = target_abad_pos.clone()
self.robot.set_joint_position_target(target_abad_pos, joint_ids=self._abad_indices)
\end{lstlisting}

\subsection{reward 核心節能片段}

\begin{lstlisting}[style=pythonstyle,caption={節能 reward 核心計算}]
main_torques, abad_torques = self._get_active_joint_torques(main_drive_vel, abad_vel)

mech_power_main = torch.sum(torch.abs(main_torques * main_drive_vel), dim=1)
mech_power_abad = torch.sum(torch.abs(abad_torques * abad_vel), dim=1)
total_mech_power = mech_power_main + mech_power_abad

actual_motion_speed = self._compute_energy_equivalent_speed(actual_lin, actual_wz)
cmd_motion_speed = self._compute_energy_equivalent_speed(cmd_lin, cmd_wz)

rew_power_efficiency = -torch.tanh(
    (total_mech_power / (actual_motion_speed + power_eff_eps)) / power_eff_tanh_scale
) * scales.get("power_efficiency", 0.3)

if hasattr(self, "_current_leg_in_stance") and self._current_leg_in_stance.shape == damper_pos.shape:
    spring_contact_mask = self._current_leg_in_stance.float()
else:
    spring_contact_mask = torch.ones_like(damper_pos)

spring_release = torch.sum(torch.clamp(-spring_power, min=0.0) * spring_contact_mask, dim=1)
spring_store = torch.sum(torch.clamp(spring_power, min=0.0) * spring_contact_mask, dim=1)

cmd_motion_gate = (cmd_motion_speed > self._energy_min_cmd_motion).float()
motion_tracking_gate = torch.clamp(
    actual_motion_speed / torch.clamp(cmd_motion_speed, min=power_eff_eps),
    min=0.0,
    max=1.0,
)
spring_reward_gate = healthy_gate * cmd_motion_gate * motion_tracking_gate
\end{lstlisting}

\subsection{\texttt{eval\_command\_sweep.py} 中新增的 energy evaluation}

\begin{lstlisting}[style=pythonstyle,caption={energy KPI 收集 helper}]
def collect_energy_metrics(unwrapped_env, actual_vx, actual_vy, actual_wz):
    lin_xy = torch.stack((actual_vx, actual_vy), dim=1)
    yaw_radius = float(getattr(unwrapped_env, "_energy_yaw_radius", 0.18))
    motion_speed = torch.sqrt(torch.sum(torch.square(lin_xy), dim=1) + torch.square(yaw_radius * actual_wz))
    ...
    return {
        "motion_speed": motion_speed,
        "mech_power_main": main_power,
        "mech_power_abad": abad_power,
        "mech_power_total": total_power,
        "cot_proxy": cot_proxy,
        "spring_energy": spring_energy,
        "spring_release": spring_release,
        "spring_store": spring_store,
        "spring_recovery_ratio": spring_recovery_ratio,
        "damper_dissipation": damper_dissipation,
    }
\end{lstlisting}

每個 command 目前都會記錄：
\begin{itemize}[leftmargin=2em]
    \item \texttt{energy\_mech\_power\_main\_mean}
    \item \texttt{energy\_mech\_power\_total\_mean}
    \item \texttt{energy\_cost\_of\_transport\_proxy}
    \item \texttt{energy\_spring\_energy\_mean}
    \item \texttt{energy\_spring\_release\_power\_mean}
    \item \texttt{energy\_spring\_store\_power\_mean}
    \item \texttt{energy\_spring\_recovery\_ratio}
    \item \texttt{energy\_motion\_speed\_mean}
    \item \texttt{energy\_power\_per\_motion}
\end{itemize}

\section{權重設計與調參建議}

\begin{longtable}{p{4cm}p{2cm}p{2cm}p{6cm}}
\toprule
項目 & Default & ForwardFast & 調參原則 \\
\midrule
\texttt{power\_efficiency} & 0.30 & 0.15 & 先保守，tracking 穩後再加強 \\
\texttt{spring\_recovery} & 0.40 & 0.20 & 是最核心的 spring reward \\
\texttt{spring\_utilization} & 0.20 & 0.10 & 小權重，避免刷偏轉 \\
\texttt{torque\_penalty} & -1e-4 & -5e-5 & 只抑制極端有源輸出 \\
\texttt{torque\_penalty\_abad\_weight} & 0.50 & 0.50 & 避免 ABAD 被過度壓制 \\
\bottomrule
\end{longtable}

推薦調整順序：
\begin{enumerate}[leftmargin=2em]
    \item baseline tracking/stability；
    \item 加 torque penalty；
    \item 加 power efficiency；
    \item 最後加 spring recovery / spring utilization。
\end{enumerate}

\section{Ablation 與驗證流程}

\subsection{四版本比較}

\begin{enumerate}[leftmargin=2em]
    \item A：baseline；
    \item B：baseline + torque penalty；
    \item C：baseline + power efficiency；
    \item D：baseline + power efficiency + spring recovery/utilization。
\end{enumerate}

\subsection{TensorBoard 必看曲線}

\begin{itemize}[leftmargin=2em]
    \item \texttt{rew\_power\_efficiency}
    \item \texttt{rew\_spring\_recovery}
    \item \texttt{rew\_spring\_utilization}
    \item \texttt{rew\_torque\_penalty}
    \item \texttt{diag\_mech\_power\_total}
    \item \texttt{diag\_cost\_of\_transport}
    \item \texttt{diag\_spring\_recovery\_ratio}
    \item \texttt{diag\_motion\_speed\_equiv}
    \item \texttt{diag\_cmd\_motion\_speed\_equiv}
\end{itemize}

\subsection{如何判斷「真的變省能」}

以下條件至少應同時成立：
\begin{enumerate}[leftmargin=2em]
    \item tracking quality 不顯著下降；
    \item success ratio 不顯著下降；
    \item fall rate 不惡化；
    \item $CoT^*$ 下降；
    \item power-per-motion 下降。
\end{enumerate}

若只看到功率下降，但 motion speed 也大幅下降，則不能聲稱真正更高效。

\section{風險與注意事項}

\begin{itemize}[leftmargin=2em]
    \item 目前 damper contact gating 仍是 phase-based proxy，不是真實 GRF。
    \item 真機若缺 torque/current sensing，很多電功率結果只能作 proxy。
    \item \texttt{spring\_utilization} 是結構 proxy，不是直接機械能收益。
    \item yaw 等效半徑 $r_{yaw}$ 是工程近似，未來可再做系統辨識。
    \item 本次已完成程式實作與靜態語法檢查，但未在本 shell 內直接做 Isaac Lab runtime rollout。
\end{itemize}

\section{結論}

本研究已將 RedRhex 的節能議題從概念層推進到可執行的 Isaac Lab reward engineering 實作。其核心貢獻包括：
\begin{enumerate}[leftmargin=2em]
    \item 以物理一致的方式建構 main drive、ABAD、damper 的能量模型；
    \item 以 $v_{eq}$ 解決 pure yaw 任務的能耗正規化問題；
    \item 將 spring recovery 與 spring utilization 納入 reward，並加入 anti-hacking gating；
    \item 在 evaluation pipeline 中加入 per-command 與 per-skill energy KPI；
    \item 將所有設計落實到 \texttt{redrhex\_env\_cfg.py}、\texttt{redrhex\_env.py} 與 \texttt{eval\_command\_sweep.py}。
\end{enumerate}

因此，這套方法不僅能回答「機器人有沒有走起來」，也能回答更重要的研究問題：
\begin{quote}
機器人是否在保持 locomotion 任務品質的前提下，真的學會用更低的有源功率，並更有效地利用被動彈簧腿機制？
\end{quote}

\end{document}
```
