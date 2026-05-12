# RedRhex ROS2 ONNX Deployment Stack

這個 workspace 是把 IsaacLab / RSL-RL 匯出的 `policy.onnx` 部署到 Jetson Orin Nano 的 MVP。設計精神參考 DeepRobotics Lite3_rl_deploy：把 robot interface、state machine、policy runner、action scaling、hardware backend 分層；但 RedRhex 的 observation/action/gating 完全以本 repo 的 IsaacLab task 為準。

## 架構

```text
Sensor Input
  -> ObservationBuilder
  -> ONNX PolicyONNXRunner
  -> ActionDecoder
  -> SafetyFilter
  -> /redrhex/motor_commands
  -> LowLevelBridge
  -> Low-level Board
  -> Motor Drivers
  -> Motors
```

MVP 先用自訂 `RedRhexMotorCommand` bridge。`ros2_control` 的優點是長期可接 `hardware_interface`、controller_manager、標準 controller；缺點是現在低階板 protocol 未定，直接上 ros2_control 會先把未知硬體介面寫死。建議先跑通本 MVP，等 MCU/CAN/micro-ROS protocol 穩定後再包成 ros2_control hardware plugin。

## 明天一步一步操作手冊

這一段是給「我現在只有 `policy.onnx`，不知道怎麼讓真機動起來」的完整流程。不要跳步。RedRhex 是 18 個關節的真機，policy 只輸出 12 維 action；真正安全讓它動起來靠的是 observation、action decoder、state machine、low-level bridge 和 E-stop。

### 0. 明天出門前先準備

請先準備這些東西：

```text
Jetson Orin Nano
RedRhex 本體
低階控制板，例如 ESP32 / ESP-Rail / MCU
IMU
joint encoder feedback
急停開關或至少一個能立刻切電的實體開關
外接螢幕或 SSH
你的 policy.onnx
一條 USB serial 或你目前能跟 MCU 溝通的線
穩固架高用的支架，讓六足離地測試
```

最重要的規則：

```text
一開始不要插馬達主電源。
一開始不要讓 policy 接管。
一開始不要測全身旋轉。
先 mock，再 heartbeat，再單顆馬達，再架空全機，最後才落地。
```

### 1. 在 Jetson 安裝基本環境

先確認 ROS2。以 Humble 為例：

```bash
source /opt/ros/humble/setup.bash
ros2 --version
```

如果你的 Jetson 是 ROS2 Jazzy，把上面 `humble` 改成 `jazzy`。接著安裝工具：

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-pip
python3 -m pip install "numpy<2" onnx onnxruntime pyserial
```

如果你用 conda，ROS2 message generation 可能找不到 `em` / `empy`。初學階段建議不要用 conda build ROS2。若真的遇到 `ModuleNotFoundError: No module named 'em'`，先用系統 Python：

```bash
cd /home/jetson/RedRhex/ros2_ws
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

### 2. 把 policy.onnx 放到固定位置

```bash
mkdir -p /home/jetson/redrhex_models
cp /你自己的路徑/policy.onnx /home/jetson/redrhex_models/policy.onnx
ls -lh /home/jetson/redrhex_models/policy.onnx
```

然後確認 YAML：

```bash
nano /home/jetson/RedRhex/ros2_ws/src/redrhex_rl_controller/config/redrhex_policy.yaml
```

確認這行存在：

```yaml
policy:
  onnx_path: "/home/jetson/redrhex_models/policy.onnx"
```

### 3. 檢查 ONNX，不接任何硬體

這一步只是在 Jetson CPU 上跑一次 zero observation。

```bash
cd /home/jetson/RedRhex
python3 ros2_ws/src/redrhex_rl_controller/scripts/check_onnx_io.py \
  /home/jetson/redrhex_models/policy.onnx
```

你要看到類似：

```text
input shape: [1, 56]   或 [1, 280]
output shape: [1, 12]
zero-observation action finite: True
ONNX I/O check OK
```

判斷方式：

```text
input 56：policy 吃單幀 observation。
input 280：policy 吃 5 幀 history，本 controller 會自動堆 history。
output 不是 12：先停止，這不是目前 RedRhex 12-action 部署 policy。
有 NaN / Inf：先停止，重新確認 ONNX export。
```

### 4. Build ROS2 workspace

```bash
cd /home/jetson/RedRhex/ros2_ws
colcon build
source install/setup.bash
```

如果 build 成功，你會得到三個 package：

```text
redrhex_msgs
redrhex_rl_controller
redrhex_lowlevel_bridge
```

### 5. 第一次跑 ROS2 graph，只用 mock，不插馬達電源

Terminal A：

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/RedRhex/ros2_ws/install/setup.bash
ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py use_fake_sensors:=true
```

Terminal B：

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/RedRhex/ros2_ws/install/setup.bash
ros2 topic list
```

你至少要看到：

```text
/imu/data
/joint_states
/cmd_vel
/redrhex/motor_commands
/redrhex/state_machine_state
/redrhex/diagnostics
/redrhex/lowlevel_heartbeat
```

再看狀態：

```bash
ros2 topic echo /redrhex/state_machine_state
```

一開始會經過：

```text
BOOT -> SENSOR_CHECK -> MOTOR_IDLE -> INIT_STAND -> WARMUP -> POLICY_READY
```

如果卡在 `SENSOR_CHECK`，代表 IMU / joint_states / heartbeat 沒進來。mock mode 下通常不該卡住。

### 6. 確認 policy 不會自動接管

預設 controller 會停在 `POLICY_READY`，不會進 `POLICY_RUN`。這是故意的。要測 ONNX 閉環時才手動打開：

```bash
ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: true}"
```

然後看：

```bash
ros2 topic echo /redrhex/policy_action_raw
ros2 topic echo /redrhex/motor_commands
```

在 mock mode 看到 action 和 motor command 就夠了。這一步仍然不要接馬達。

### 7. 接低階板 heartbeat，但仍不要插馬達主電源

這一步只測 Jetson 到低階板通訊活著。先打開 bridge config：

```bash
nano /home/jetson/RedRhex/ros2_ws/src/redrhex_lowlevel_bridge/config/lowlevel_bridge.yaml
```

如果你還沒有 MCU protocol，保持：

```yaml
backend: "mock"
```

如果你要測 USB serial skeleton：

```yaml
backend: "serial"
serial:
  port: "/dev/ttyUSB0"
  baudrate: 921600
  timeout_s: 0.005
```

查 USB port：

```bash
ls /dev/ttyUSB*
ls /dev/ttyACM*
```

啟動 bridge：

```bash
ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py
```

看 heartbeat：

```bash
ros2 topic echo /redrhex/lowlevel_heartbeat
```

你要看到：

```text
data: true
```

如果不是 true，不要進下一步。

### 8. 手動送 disable，確認低階板收到「不出力」

這一步仍然不插馬達主電源。Terminal C：

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/RedRhex/ros2_ws/install/setup.bash
ros2 run redrhex_rl_controller motor_command_tool disable --duration 2.0
```

你應該在 bridge terminal 看到 mock 或 serial 正在收到 command。這個 command `enable=false`，是最安全的第一包。

### 9. 低功率 INIT_STAND，先架空或不插主電源

先測 command 是否合理：

```bash
ros2 run redrhex_rl_controller motor_command_tool init-stand --duration 2.0
```

注意上面沒有 `--enable`，只會送出 enable false。確認 topic 內容：

```bash
ros2 topic echo /redrhex/motor_commands
```

確認 joint order 正確後，才允許低功率：

```bash
ros2 run redrhex_rl_controller motor_command_tool init-stand --enable --duration 2.0
```

此時你要用非常低的電源/限流，手放在急停旁邊。期待現象：

```text
ABAD 回到 0 rad
main drive 回到右側 +45 deg、左側 -45 deg 附近
damper 保持初始角度
沒有抖動、沒有暴衝、沒有異常電流
```

如果任何一顆馬達方向相反，立刻急停，回來改 `main_drive_sign` / `abad_sign` / encoder zero offset。不要用 policy 硬扛方向錯誤。

### 10. 單顆 ABAD 測試

架空，低功率，先測第 0 顆 ABAD，小角度 0.10 rad：

```bash
ros2 run redrhex_rl_controller motor_command_tool single-abad \
  --enable \
  --index 0 \
  --position 0.10 \
  --kp 4.0 \
  --kd 0.3 \
  --effort-limit 1.0 \
  --duration 2.0
```

再回 0：

```bash
ros2 run redrhex_rl_controller motor_command_tool single-abad \
  --enable \
  --index 0 \
  --position 0.0 \
  --kp 4.0 \
  --kd 0.3 \
  --effort-limit 1.0 \
  --duration 2.0
```

依序測 `--index 0` 到 `5`。每一顆都確認：

```text
方向跟 IsaacLab 定義一致
encoder position 增減方向合理
沒有卡死
current 不過高
temperature 不快速上升
fault flag 是 false
```

### 11. 六顆 ABAD 小角度同步測試

只在單顆都通過後做：

```bash
ros2 run redrhex_rl_controller motor_command_tool all-abad \
  --enable \
  --position 0.08 \
  --kp 4.0 \
  --kd 0.3 \
  --effort-limit 1.0 \
  --duration 2.0
```

回 0：

```bash
ros2 run redrhex_rl_controller motor_command_tool all-abad \
  --enable \
  --position 0.0 \
  --kp 4.0 \
  --kd 0.3 \
  --effort-limit 1.0 \
  --duration 2.0
```

### 12. 單顆 main drive 低速測試

main drive 是旋轉腿，危險性比 ABAD 高。一定要架空，速度先低到 `0.3 rad/s`：

```bash
ros2 run redrhex_rl_controller motor_command_tool single-main-velocity \
  --enable \
  --index 0 \
  --velocity 0.3 \
  --kd 0.5 \
  --effort-limit 1.0 \
  --duration 2.0
```

停止：

```bash
ros2 run redrhex_rl_controller motor_command_tool disable --duration 1.0
```

依序測 `0..5`。確認：

```text
右側腿、左側腿的正方向是否符合訓練的 leg_direction_multiplier
encoder velocity 正負號是否符合 command
沒有機械干涉
沒有線材被捲入
```

### 13. 架空整機，先只跑 controller 的 INIT_STAND

此時還不要 enable policy。啟動完整 bringup：

```bash
ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py use_fake_sensors:=false
```

你需要真實 `/imu/data` 和 `/joint_states` 已經在跑。確認：

```bash
ros2 topic hz /imu/data
ros2 topic hz /joint_states
ros2 topic echo /redrhex/state_machine_state
```

期待 state 到：

```text
POLICY_READY
```

如果卡住，看 diagnostics：

```bash
ros2 topic echo /redrhex/diagnostics
```

### 14. 架空整機，低風險 policy run

先讓 command 是 0：

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

再 enable policy：

```bash
ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: true}"
```

看 action 與 motor command：

```bash
ros2 topic echo /redrhex/policy_action_raw
ros2 topic echo /redrhex/policy_action_safe
ros2 topic echo /redrhex/motor_commands
```

如果 action 有 NaN、motor command 突然很大、或任何馬達暴衝，急停。

### 15. 架空整機，給很小的 forward command

從小速度開始，不要直接用訓練最大速度：

```bash
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.10, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

注意：訓練 command range stage-5 forward 是 `0.22..0.45`，但真機第一次測試可以先低於訓練範圍，只看是否安全。真的要測 locomotion tracking 時再逐漸靠近訓練範圍。

### 16. 落地前 checklist

全部勾完才落地：

```text
[ ] E-stop 可用，按下後 motor command enable=false 或低階板立刻斷輸出
[ ] /imu/data rate 穩定
[ ] /joint_states rate 穩定
[ ] /motor_feedback 有 current / temperature / fault
[ ] /redrhex/diagnostics 無 ERROR
[ ] INIT_STAND 不抖
[ ] 六顆 ABAD 單顆測試通過
[ ] 六顆 main drive 單顆低速測試通過
[ ] 架空 policy run 沒有 NaN / Inf / deadline miss
[ ] 馬達方向與 encoder 方向已確認
[ ] 線材不會被旋轉腿捲入
[ ] base_lin_vel 來源已理解；若仍是 0，只能做非常保守低速測試
```

### 17. 第一次落地

第一次落地只做低速短時間：

```bash
ros2 bag record /imu/data /joint_states /cmd_vel /motor_feedback \
  /redrhex/observation /redrhex/policy_action_raw \
  /redrhex/policy_action_safe /redrhex/motor_commands \
  /redrhex/state_machine_state /redrhex/diagnostics
```

另一個 terminal：

```bash
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.10, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

只跑 2 到 3 秒。觀察：

```text
有沒有往預期方向移動
有沒有側翻趨勢
roll / pitch 是否接近 safety limit
current 是否暴增
腳是否打滑或相位混亂
```

第一次不要測側移、斜走、原地旋轉。前進穩定後再測 ABAD 功能。

### 18. 一出問題你該怎麼辦

立刻做這幾件事：

```bash
ros2 topic pub --once /estop std_msgs/msg/Bool "{data: true}"
ros2 run redrhex_rl_controller motor_command_tool disable --duration 1.0
```

然後切馬達主電源。不要在問題狀態下繼續 publish cmd_vel。

常見問題：

```text
一 enable 就暴衝：馬達方向或 encoder zero 錯，先不要跑 policy。
一直 SENSOR_CHECK：IMU 或 joint_states topic 沒進來，或 joint name 不匹配。
ONNX input 是 280：需要 history，controller 已支援，但你要確認 observation order。
落地完全不會走：base_lin_vel=0 造成 sim2real mismatch，需 odom/leg odometry。
側移很怪：ABAD sign 或 zero offset 很可能錯。
```

## 已確認

- Gym task：`Template-Redrhex-Direct-v0`
- `policy.onnx` 由 `scripts/rsl_rl/play.py` 匯出，程式呼叫 `export_policy_as_onnx(policy_nn, normalizer=normalizer, ...)`
- ONNX 若 policy 物件有 `actor_obs_normalizer` 或 `student_obs_normalizer`，export 會把 normalizer 包進 ONNX；Jetson 端不要重複 normalize
- action space：12
- single observation space：56
- 新版 RSL-RL config 可能使用 `policy + history`，ONNX input 可能是 56 或 280；請用 `check_onnx_io.py` 確認
- control dt：`sim.dt * decimation = (1/250) * 2 = 0.008 s`
- policy frequency：`125 Hz`
- repo 註解有些地方寫 120 Hz / 60 Hz，但實際程式參數是 250 Hz sim、decimation 2、policy 125 Hz
- exported ONNX 字串可見 `normalizer._mean` 時代表 normalizer 已在圖內
- repo 目前沒有 ROS2 package
- repo 目前沒有 MCU/low-level communication protocol

## 仍未知 / TODO

- 你的最終低階板通訊：UART / USB serial / CAN / micro-ROS / SPI / UDP 尚未確定
- 真實馬達方向、encoder zero offset、joint axis sign 尚未確認
- base linear velocity estimator 尚未完成；初期 `base_lin_vel=0` 只能 bench test
- 真機 IMU frame 是否與 IsaacLab base frame 一致尚未驗證
- 真機 ABAD/main drive torque/current/temperature limit 需依硬體標定

## Jetson 安裝假設

- Jetson Orin Nano
- Ubuntu 22.04 / JetPack 6.x
- ROS2 Humble
- Python 3.10

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-pip ros-humble-desktop
python3 -m pip install "numpy<2" onnx onnxruntime pyserial
```

若使用 GPU，依 JetPack / CUDA 相容性安裝 `onnxruntime-gpu`，再在 YAML 啟用：

```yaml
policy:
  use_cuda: true
  use_tensorrt: false
```

## policy.onnx 放置方式

建議：

```bash
mkdir -p /home/jetson/redrhex_models
cp /path/to/policy.onnx /home/jetson/redrhex_models/policy.onnx
```

修改：

```text
ros2_ws/src/redrhex_rl_controller/config/redrhex_policy.yaml
```

```yaml
policy:
  onnx_path: "/home/jetson/redrhex_models/policy.onnx"
```

## 你現在只有 .onnx，第一步怎麼做

先不要接馬達，先確認 ONNX I/O：

```bash
cd /home/jasonliao/RedRhex/RedRhex
python3 ros2_ws/src/redrhex_rl_controller/scripts/check_onnx_io.py /path/to/policy.onnx
```

期待：

- input dim 是 56 或 280
- output dim 是 12
- zero observation 可以推論
- 沒有 NaN / Inf

若 input 是 280，代表 policy 需要 5 幀 history，本 controller 會自動堆疊 `[current, prev1, prev2, prev3, prev4]`。

## Build

```bash
cd /home/jasonliao/RedRhex/RedRhex/ros2_ws
colcon build
source install/setup.bash
```

## Mock mode 測試

不接真機：

```bash
ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py use_fake_sensors:=true
```

開另一個 terminal：

```bash
source /home/jasonliao/RedRhex/RedRhex/ros2_ws/install/setup.bash
ros2 topic echo /redrhex/state_machine_state
ros2 topic echo /redrhex/motor_commands
ros2 topic echo /redrhex/diagnostics
```

預設不會進 `POLICY_RUN`。要明確允許：

```bash
ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: true}"
```

## 切換 low-level bridge

Mock：

```yaml
redrhex_lowlevel_bridge:
  ros__parameters:
    backend: "mock"
```

Serial skeleton：

```yaml
redrhex_lowlevel_bridge:
  ros__parameters:
    backend: "serial"
    serial:
      port: "/dev/ttyUSB0"
      baudrate: 921600
      timeout_s: 0.005
```

Serial 封包目前是 provisional，包含 magic header、version、sequence、timestamp、joint count、target arrays、enable、CRC。MCU protocol 確定後請替換 `serial_bridge.py`。

## Observation order

來自 `RedrhexEnv._get_observations()`：

```text
0:3    base_lin_vel
3:6    base_ang_vel
6:9    projected_gravity
9:15   main_drive_pos_sin
15:21  main_drive_pos_cos
21:27  main_drive_vel / base_gait_angular_vel
27:33  abad_pos / abad_pos_scale
33:39  abad_vel
39:42  velocity_command [vx, vy, wz]
42:43  sin(gait_phase)
43:44  cos(gait_phase)
44:56  last_actions
```

IsaacLab 端還做：

- observation domain randomization noise / latency during training
- `nan_to_num`
- clamp 到 `[-100, 100]`

真機端不加訓練噪音，但保留 finite check 與 clamp。

## Joint order

Main drive action `[0:6]`：

```text
0 Revolute_15  right front main drive
1 Revolute_7   right middle main drive
2 Revolute_12  right rear main drive
3 Revolute_18  left front main drive
4 Revolute_23  left middle main drive
5 Revolute_24  left rear main drive
```

ABAD action `[6:12]`：

```text
6  Revolute_14  right front ABAD
7  Revolute_6   right middle ABAD
8  Revolute_11  right rear ABAD
9  Revolute_17  left front ABAD
10 Revolute_22  left middle ABAD
11 Revolute_21  left rear ABAD
```

Damper joints, not policy-controlled:

```text
Revolute_5, Revolute_8, Revolute_13, Revolute_25, Revolute_26, Revolute_27
```

## Action decoding summary

來自 `RedrhexEnv._pre_physics_step()` 與 `_apply_action()`：

- raw policy action clamp 到 `[-1, 1]`
- first 6：main drive residual / velocity control
- last 6：ABAD position target / procedural blend
- main drive simulator actuator是 velocity control：`set_joint_velocity_target(...)`
- ABAD simulator actuator是 position control：`set_joint_position_target(...)`
- damper joints hold initial pose
- pure lateral mode 會 mask main-drive policy action
- forward mode 會 mask ABAD policy action
- lateral mode 有 GO_TO_STAND -> LATERAL_STEP FSM
- diag/yaw mode 有額外 ABAD bias / scale
- stage-5 deployment defaults已寫入 `redrhex_contract.py`

## 真機測試順序

A. 不插馬達電源，只測 ROS2 graph  
B. mock bridge 測試  
C. low-level board heartbeat 測試  
D. 單顆馬達命令測試  
E. 六顆 ABAD 測試  
F. 六顆 main drive 測試  
G. 架空整機測試  
H. 落地低速測試  
I. 最後才允許 policy 接管完整 locomotion

不能跳過 `INIT_STAND`，不能把 raw action 直接送馬達。

## 已知風險

- `base_lin_vel` 若長期設為 0，真機 locomotion 可能失效或出現嚴重 sim2real mismatch
- normalizer 是否已包含在 ONNX 必須用 `check_onnx_io.py` 或 Netron/strings 確認；若 ONNX 已包含 normalizer，不可重複 normalize
- policy frequency 必須與訓練一致，本 repo 實際是 125 Hz
- 真實馬達方向可能和 IsaacLab joint axis 相反
- encoder zero offset 可能導致 INIT_STAND 姿態錯誤
- low-level board protocol 尚未確認
- latency、packet loss、sensor delay 會造成 sim2real gap
- IMU frame 若不是 body frame，需要加 TF transform hook
- policy output 必須經 ActionDecoder 和 SafetyFilter，不可 raw action 直送

## Offline consistency

若同一個 export folder 有 `policy.pt`：

```bash
python3 ros2_ws/src/redrhex_rl_controller/scripts/compare_onnx_with_torch.py \
  --onnx /path/to/exported/policy.onnx \
  --torchscript /path/to/exported/policy.pt
```

若誤差過大，檢查 normalizer、export path、eval mode、input observation order。
