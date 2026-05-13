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

## Lite3_rl_deploy 參考點

DeepRobotics `Lite3_rl_deploy` 的有用設計是分層，而不是它的 Lite3 關節定義或 UDP protocol：

```text
Lite3 interface      -> RedRhex LowLevelBridge abstraction
Lite3 state_machine -> RedRhexStateMachine
Lite3 run_policy    -> PolicyONNXRunner + ObservationBuilder + ActionDecoder
Lite3 sim-to-sim    -> RedRhex mock/fake sensor test
Lite3 sim-to-real   -> RedRhex heartbeat、單顆馬達、架空、落地分階段測試
```

它的 README 也強調 ONNX 轉換後要檢查模型輸出一致性，所以這裡提供 `check_onnx_io.py` 和 `compare_onnx_with_torch.py`。但 RedRhex 的 observation order、action scaling、gait phase、ABAD gating 都不能照 Lite3 猜，必須以本 repo 的 IsaacLab `RedrhexEnv` 為準。

## 明天一步一步操作手冊

這一段是給「我現在只有 `policy.onnx`，不知道怎麼讓真機動起來」的完整流程。不要跳步。訓練模型裡有 18 個 joint 名稱，但真機只有 12 顆可控馬達：6 顆 main drive 旋轉腿馬達、6 顆 ABAD 馬達。damper joints 只是模擬彈簧腳用的 joint，不是實體馬達，不會送 command。

## 你的真機硬體回授假設

目前程式已依照你描述的真機改成這個模式：

```text
可取得真實回授：
  IMU orientation / gyro
  6 顆 main drive encoder position

可估算：
  main drive velocity：由 main encoder position 差分估算
  ABAD position：使用上一個下達的 ABAD position command 當估計值
  ABAD velocity：由 ABAD command 差分估算

不可取得：
  ABAD encoder feedback
  damper feedback
  damper command
  穩定 base linear velocity
```

因此 `/joint_states` 對真機最低需求是 6 顆 main drive encoder position。ABAD 不需要出現在 `/joint_states`，因為 YAML 預設：

```yaml
observation:
  abad_feedback_source: "commanded"
  estimate_missing_joint_velocity: true

action:
  include_damper_command: false
  main_drive_init_control_mode: "velocity_to_pose"
```

這是可跑 MVP 的做法，但要知道：ABAD 沒有回授會造成 sim2real mismatch。只要 ABAD 實際位置跟 command 不一致，policy 看到的 observation 就會錯。

### 0. 明天出門前先準備

請先準備這些東西：

```text
Jetson Orin Nano
RedRhex 本體
低階控制板：sbRIO
IMU
6 顆 main drive encoder feedback
急停開關或至少一個能立刻切電的實體開關
外接螢幕或 SSH
你的 policy.onnx
一條 Jetson 到 sbRIO 的 Ethernet 線
穩固架高用的支架，讓六足離地測試
```

最重要的規則：

```text
一開始不要插馬達主電源。
一開始不要讓 policy 接管。
一開始不要測全身旋轉。
先 mock，再 heartbeat，再單顆馬達，再架空全機，最後才落地。
```

明天的成功標準不是「馬上走起來」，而是安全通過這條最短路線：

```text
ONNX I/O 正確
ROS2 mock graph 正常
low-level heartbeat 正常
manual disable / dry-run 正常
單顆 ABAD 小角度正常
單顆 main drive 低速正常
架空 INIT_STAND 正常
架空 policy dry-run 正常
最後才做落地 2 秒低速測試
```

controller 有兩個開關，請分清楚：

```text
/redrhex/enable_policy：只允許 ONNX policy 進入閉環計算。
/redrhex/enable_motors：才允許 /redrhex/motor_commands 的 enable 變成 true。
```

也就是說，就算你打開 `/redrhex/enable_policy`，只要 `/redrhex/enable_motors` 還是 false，馬達仍不應出力。這是故意加上的第二道保險。

開關允許表：

```text
BOOT / SENSOR_CHECK / MOTOR_IDLE：兩個開關都會被拒絕。
INIT_STAND / WARMUP：只允許 enable_motors，讓機器人回站姿，不允許 policy。
POLICY_READY：允許 enable_policy，也允許 enable_motors。
POLICY_RUN：允許兩者維持開啟。
PROTECTIVE_STOP / FALL_DETECTED / RECOVER：兩個 latch 會自動關掉。
```

如果你 publish 了 enable 但沒生效，先看：

```bash
ros2 topic echo /redrhex/state_machine_state
ros2 topic echo /redrhex/diagnostics
ros2 topic echo /redrhex/motor_commands
```

不要一直重送 enable。先把 state 和 diagnostics 看懂。

`/redrhex/diagnostics` 會帶這些現場最常用欄位：

```text
state
last_transition_reason
policy_loaded
policy_enabled
motor_output_enabled
estop
imu_age_s
joint_state_age_s
motor_feedback_age_s
heartbeat_age_s
control_loop_dt_s
roll_rad
pitch_rad
cmd_vel
```

如果 `imu_age_s` 或 `joint_state_age_s` 一直變大，表示 sensor topic 斷了；如果 `control_loop_dt_s` 大於 YAML 的 `max_control_loop_dt_s`，表示 Jetson 控制迴圈已經 miss deadline，不要落地。

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
python3 -m pip install "numpy<2" onnx onnxruntime pyserial pyyaml
```

如果跑 script 出現 `ModuleNotFoundError: No module named 'numpy'`，代表你目前 terminal 用的 Python 環境沒有裝到套件。先確認：

```bash
which python3
python3 -m pip show numpy
/usr/bin/python3 -m pip show numpy
```

初學階段建議讓 ROS2 和測試 script 都用同一個系統 Python，少碰 conda。

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

這一步只是在 Jetson CPU 上跑一次 zero observation。這個 script 不需要先 `colcon build`，所以它是最早可以做的檢查。

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

Build 後再跑整合版 preflight。它會檢查 ONNX 是否存在、I/O shape、zero observation 推論、repo 推導出的 policy Hz，並提醒你目前是不是只能做 bench test：

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/RedRhex/ros2_ws/install/setup.bash
ros2 run redrhex_rl_controller preflight_check \
  --onnx /home/jetson/redrhex_models/policy.onnx \
  --config /home/jetson/RedRhex/ros2_ws/src/redrhex_rl_controller/config/redrhex_policy.yaml
```

看到 JSON 裡所有 `checks.ok` 都是 `true`，才進下一步。這個 preflight 會檢查：

```text
policy.onnx 是否存在
ONNX input/output shape
zero observation 能否推論
joint name 常數是否是 6+6+6 且不重複
YAML 裡 observation/action/safety 參數是否合法
ABAD 是否設定為 commanded feedback
damper 是否已排除 motor command
enable_policy_on_start / enable_motor_output_on_start 是否誤開
base_lin_vel_source 是否仍是 zero
```

若 `warnings` 提到 `base_lin_vel`，這不是 build 失敗，而是提醒你還沒有真機線速度估測；這時只能做 mock、架空、低速 bench test，不應直接落地跑 policy。

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
/redrhex/enable_policy
/redrhex/enable_motors
```

再看狀態：

```bash
ros2 topic echo /redrhex/state_machine_state
```

一開始會經過：

```text
BOOT -> SENSOR_CHECK -> MOTOR_IDLE -> INIT_STAND -> WARMUP -> POLICY_READY
```

如果卡在 `SENSOR_CHECK`，代表 IMU / joint_states / heartbeat 沒進來。真機 `/joint_states` 至少要有 6 顆 main drive encoder position；ABAD 沒有 feedback 沒關係，因為目前使用 commanded estimate。mock mode 下通常不該卡住。

fake sensor 預設也模擬你的真機，只發布 6 顆 main drive joint：

```bash
ros2 topic echo --once /joint_states
```

你應該看到 `name` 只有：

```text
Revolute_15, Revolute_7, Revolute_12, Revolute_18, Revolute_23, Revolute_24
```

如果你只是想測舊的完整 sim-style joint list，可以 launch 時加 `fake_publish_abad_joints:=true`、`fake_publish_damper_joints:=true`，但真機 bringup 不需要。

### 6. 確認 policy 和馬達輸出都不會自動接管

預設 controller 會停在 `POLICY_READY`，不會進 `POLICY_RUN`，而且 `enable_motors=false`。這是故意的。要測 ONNX 閉環計算時才手動打開 policy。注意：如果 state 還不是 `POLICY_READY` 或 `POLICY_RUN`，controller 會拒絕這個 request。

```bash
ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: true}"
```

然後看：

```bash
ros2 topic echo /redrhex/policy_action_raw
ros2 topic echo /redrhex/motor_commands
```

在 mock mode 看到 action 和 motor command 就夠了。這一步仍然不要接馬達，而且 `/redrhex/motor_commands.enable` 預期仍是 `false`，除非你另外明確打開 `/redrhex/enable_motors`。

若看到 log 顯示 `Rejecting policy enable`，代表你太早 enable。等 state 進 `POLICY_READY` 後再送一次。

### 7. 接 RhexROS2 / sbRIO heartbeat，但仍不要插馬達主電源

你提供的真機 ROS2 repo 是 `JasonLiaoJCS/RhexROS2`，它已經定義好 sbRIO 那邊使用的 ROS2 interface。這份 RedRhex deployment stack 現在支援 `rinbo_ros` backend，專門對接你的 RhexROS2：

```text
RedRhex RL controller
  -> /redrhex/motor_commands              redrhex_msgs/RedRhexMotorCommand
  -> redrhex_lowlevel_bridge backend=rinbo_ros
  -> /motor/command                       rinbo_msgs/MotorCmdStamped
  -> RhexROS2 rinbo_ros_bridge
  -> sbRIO / CORE / motor drivers

RhexROS2 /motor/state                     rinbo_msgs/MotorStateStamped
  -> redrhex_lowlevel_bridge backend=rinbo_ros
  -> /joint_states                        sensor_msgs/JointState, 只含 6 顆 main encoder
  -> ObservationBuilder
```

先把 RhexROS2 workspace 準備好。若你 Jetson 上已經有這個 workspace，只要 source 它即可；若還沒有：

```bash
cd /home/jetson
git clone https://github.com/JasonLiaoJCS/RhexROS2.git
cd RhexROS2
source /opt/ros/humble/setup.bash
colcon build --packages-select rinbo_msgs
source install/setup.bash
ros2 interface show rinbo_msgs/msg/MotorCmdStamped
ros2 interface show rinbo_msgs/msg/MotorStateStamped
```

如果只是在測 RedRhex adapter，先 build `rinbo_msgs` 就夠了。要真的連 sbRIO，還需要你的 RhexROS2 `rinbo_ros_bridge` 能成功 build 和執行；這可能需要原本 repo 裡的 gRPC / CORE 環境。

```bash
cd /home/jetson/RhexROS2
source /opt/ros/humble/setup.bash
source install/setup.bash
export CORE_MASTER_ADDR="192.168.30.12:50051"
export CORE_LOCAL_IP="192.168.30.164"
ros2 run rinbo_ros_bridge rinbo_ros_bridge
```

如果你使用 RhexROS2 的 `start_bridge.sh`，請先確認裡面的 workspace 路徑符合你自己的 Jetson。原始 script 會 source `~/rinbo_ros_ws/install/setup.bash`，如果你的路徑是 `/home/jetson/RhexROS2/install/setup.bash`，請改掉或直接用上面的手動指令。

RhexROS2 bridge 起來後，先確認 topic 有出現：

```bash
ros2 topic list | grep motor
ros2 topic echo /motor/state --once
```

然後打開 RedRhex bridge config：

```bash
nano /home/jetson/RedRhex/ros2_ws/src/redrhex_lowlevel_bridge/config/lowlevel_bridge.yaml
```

把 backend 改成 `rinbo_ros`：

```yaml
redrhex_lowlevel_bridge:
  ros__parameters:
    backend: "rinbo_ros"
    rinbo:
      command_topic: "/motor/command"
      state_topic: "/motor/state"
      joint_state_topic: "/joint_states"
      preview_topic: "/redrhex/rinbo_motor_command_preview"
      publish_preview: true
      allow_enable: false
      publish_when_disabled: false
      disabled_servo_control_mode: 0
      require_state: true
      state_timeout_s: 0.25
      main_position_counts_per_rev: 54984.83
      main_pwm_per_rad_s: 120.0
      main_max_pwm: 500.0
      main_encoder_zero_counts_rinbo_order: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      main_encoder_sign_rinbo_order: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
      main_velocity_sign_policy_order: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      main_direction_positive_rinbo_order: [false, false, false, true, true, true]
      abad_encoder_zero_rinbo_order: [739, 2566, 3283, 1945, 2070, 987]
      abad_encoder_counts_per_rad: 1000.0
      abad_encoder_min: 0
      abad_encoder_max: 65535
      abad_sign_rinbo_order: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      servo_control_mode: 2
```

`allow_enable: false` 一定先保持 false。這代表 adapter 會接收 RL command、做轉換、發布 diagnostics，但拒絕任何 enabled motor command。等你完成急停、限流、單顆馬達測試後，才可以改成 true。

`preview_topic` 是安全預覽 topic。當 `publish_preview: true` 時，adapter 會把轉換後的 `rinbo_msgs/MotorCmdStamped` 發到 `/redrhex/rinbo_motor_command_preview`，讓你在 `allow_enable=false` 時也能檢查 `l1/l2/.../sr3` 的 PWM、direction、servo encoder。RhexROS2 bridge 不會訂閱這個 topic，所以它不會讓馬達動。

`publish_when_disabled: false` 也先保持 false。RhexROS2 的 ABAD servo command 沒有 per-servo enable 欄位；如果 disabled command 也 publish，servo 仍可能吃到 `position_encoder`。所以 dry-run 階段只看 `/redrhex/motor_commands`，不要讓 adapter publish `/motor/command`。如果 adapter 前一包已經是 enabled，下一包 disabled 仍會被送出一次，確保 main legs release。等你確認伺服電源斷開、或確認 `servo_control_mode=0` 真的不會動，再暫時打開它做 message-level 測試。

`require_state: true` 代表 `/redrhex/lowlevel_heartbeat` 只有在真的收到 `/motor/state` 時才會是 true。如果你只是離線測 message conversion，沒有啟動 RhexROS2 bridge，可以暫時改成 false，但真機測試要改回 true。

請用三個 terminal 分開跑。

Terminal A：RhexROS2 bridge

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/RhexROS2/install/setup.bash
export CORE_MASTER_ADDR="192.168.30.12:50051"
export CORE_LOCAL_IP="192.168.30.164"
ros2 run rinbo_ros_bridge rinbo_ros_bridge
```

Terminal B：RedRhex low-level adapter

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/RhexROS2/install/setup.bash
source /home/jetson/RedRhex/ros2_ws/install/setup.bash
ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py
```

Terminal C：檢查 heartbeat 和 encoder 是否進來

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/RhexROS2/install/setup.bash
source /home/jetson/RedRhex/ros2_ws/install/setup.bash
ros2 topic echo /redrhex/lowlevel_heartbeat
ros2 topic echo /joint_states --once
ros2 topic echo /motor_feedback --once
ros2 topic echo /redrhex/rinbo_motor_command_preview --once
```

你在 `/joint_states` 應該只看到 6 個 main drive joint：

```text
Revolute_15, Revolute_7, Revolute_12, Revolute_18, Revolute_23, Revolute_24
```

這是正確的。ABAD 沒有 feedback，所以 observation 用上一筆 ABAD command 當 estimate；damper 是模擬彈簧腳，不會出現在真機 command。

RhexROS2 的 leg/servo order 和 policy order 不一樣，目前 adapter 內部這樣轉：

```text
RhexROS2 main leg order: l1, l2, l3, r1, r2, r3
Policy main order:      RF, RM, RR, LF, LM, LR

RhexROS2 servo order:   sl1, sl2, sl3, sr1, sr2, sr3
Policy ABAD order:      RF, RM, RR, LF, LM, LR
```

main drive policy velocity 會轉成 RhexROS2 的 `LegCmd.voltage` 和 `LegCmd.direction`。你目前的 RhexROS2 bridge 只把 `enable/direction/voltage/state/reset_position` 傳到 gRPC；`target_velocity` 欄位在 bridge 裡沒有送出去，所以 RL 端不能假設 sbRIO 會做速度閉環。現在的 `main_pwm_per_rad_s` 是保守初值，真正走路前一定要架空調整。`main_velocity_sign_policy_order`、`main_direction_positive_rinbo_order`、`main_encoder_sign_rinbo_order` 都是為真機方向校正準備的，不要在 Python 裡硬改。

ABAD position 會轉成 `ServoCmd.position_encoder`。`abad_encoder_zero_rinbo_order` 目前使用 RhexROS2 既有站姿附近數值作初始值，不等於你的最終機械零點；`abad_encoder_counts_per_rad` 和 `abad_sign_rinbo_order` 也必須實測。ABAD 沒有 feedback 時，這些數字錯了，policy 看到的 ABAD observation 就會錯。

#### RhexROS2 adapter 校正順序

不要一開始就跑 policy。先只校正 adapter：

1. `allow_enable=false`、`publish_when_disabled=false`，先確認 `/redrhex/motor_commands` 與 `/redrhex/rinbo_motor_command_preview`，不讓 adapter publish enabled command 到 `/motor/command`。
2. 啟動 RhexROS2 bridge，只看 `/motor/state` 是否穩定，確認 `require_state=true` 時 `/redrhex/lowlevel_heartbeat` 會變 true。
3. 手轉每一顆 main drive 一小段，檢查 `/joint_states.position` 對應的 policy joint 是否變動。如果符號反了，只改 `main_encoder_sign_rinbo_order`。
4. 架空、限流、只測單顆 main drive。若 RL/工具命令正速度時實體腿方向反了，優先改 `main_velocity_sign_policy_order`；若只是 `LegCmd.direction` boolean 與 sbRIO 韌體定義相反，再改 `main_direction_positive_rinbo_order`。
5. 斷 main drive，只測單顆 ABAD 小角度。如果正角度反向，改 `abad_sign_rinbo_order`；如果中立角不對，改 `abad_encoder_zero_rinbo_order`；如果角度比例不對，改 `abad_encoder_counts_per_rad`。
6. 每改一次 YAML 都重新 build/source 或重啟 launch，並看 `/redrhex/lowlevel_diagnostics` 裡的 `rinbo_actual_publish_state`、`rinbo_last_pwm_l1_l2_l3_r1_r2_r3` 與 `rinbo_last_abad_sl1_sl2_sl3_sr1_sr2_sr3`。

如果你 publish 了 `--enable` 的手動命令，但 `allow_enable=false`，你會看到：

```text
/redrhex/rinbo_motor_command_preview：有轉換後的 PWM / servo target
/motor/command：不會收到 enabled command
/redrhex/lowlevel_diagnostics：rinbo_actual_publish_state=blocked_allow_enable
```

這是正確的安全狀態。

不要同時跑 RhexROS2 既有的 `rinbo_fsm` tripod/standing 節點和 RedRhex RL controller。兩邊都會 publish `/motor/command`，會互相搶控制權。

如果你暫時不走 RhexROS2，而是要測本 repo 內的 sbRIO UDP skeleton：

```yaml
backend: "sbrio_udp"
sbrio:
  remote_host: "192.168.0.2"
  command_port: 15000
  bind_host: "0.0.0.0"
  feedback_port: 15001
  timeout_s: 0.002
  heartbeat_timeout_s: 0.25
  allow_enable: false
  require_feedback: false
```

這條 UDP skeleton 只適合你未來自己寫 LabVIEW RT/FPGA protocol 時使用。既然你目前已有 RhexROS2，明天優先用 `rinbo_ros`。

raw UDP skeleton 的建議資料流：

```text
Jetson ROS2 /redrhex/motor_commands
  -> redrhex_lowlevel_bridge sbrio_udp
  -> Ethernet UDP
  -> sbRIO LabVIEW RT
  -> FPGA / motor driver interface
  -> motor drivers
```

先讓 Jetson 和 sbRIO 在同一個 subnet，例如 Jetson `192.168.0.10`、sbRIO `192.168.0.2`。先確認：

```bash
ping 192.168.0.2
```

`allow_enable` 預設一定要保持 `false`。這代表 sbRIO UDP skeleton 只允許你測封包與 heartbeat，不允許透過尚未確認的 protocol 讓馬達真的出力。等 sbRIO 端已經完成 watchdog、CRC 驗證、急停路徑、單顆馬達測試後，才可以把它改成 `true`。

`require_feedback` 一開始可以是 `false`，只測 Jetson UDP send path。等 sbRIO 端會回 heartbeat 或 feedback packet 後，再改成 `true`，這樣 `/redrhex/lowlevel_heartbeat` 才代表真的收到 sbRIO 回覆。

如果你要測 USB serial skeleton：

```yaml
backend: "serial"
serial:
  port: "/dev/ttyUSB0"
  baudrate: 921600
  timeout_s: 0.005
  allow_enable: false
```

`allow_enable` 預設一定要保持 `false`。這代表 serial skeleton 只允許你測封包與 heartbeat，不允許透過尚未確認的 provisional protocol 讓馬達真的出力。等 MCU 端已經完成 watchdog、CRC 驗證、急停路徑、單顆馬達測試後，才可以把它改成 `true`。

查 USB port：

```bash
ls /dev/ttyUSB*
ls /dev/ttyACM*
```

如果你沒有照上面 Terminal B 啟動，這裡也可以單獨啟動 RedRhex low-level bridge：

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

你應該在 bridge terminal 看到 mock、rinbo_ros、sbrio_udp 或 serial 正在收到 command。這個 command `enable=false`，是最安全的第一包。

先確認 joint order：

```bash
ros2 run redrhex_rl_controller motor_command_tool list-joints
```

你也可以先 dry-run 看即將送出的 JSON，不 publish：

```bash
ros2 run redrhex_rl_controller motor_command_tool init-stand --dry-run
```

任何會讓馬達真的出力的手動指令都需要同時加：

```text
--enable --confirm-risk
```

這是故意設計的，避免你複製指令時誤觸真機。

### 9. 低功率 INIT_STAND，先架空或不插主電源

先測 command 是否合理：

```bash
ros2 run redrhex_rl_controller motor_command_tool init-stand --duration 2.0
```

注意上面沒有 `--enable`，只會送出 enable false。確認 topic 內容：

```bash
ros2 topic echo /redrhex/motor_commands
```

手動工具的 `init-stand` 沒有讀 encoder，所以它只能送一包安全的 12-motor target preview；它不會像 controller 一樣用 main encoder 做 `velocity_to_pose` 閉環。main drive 回站姿請以第 13 步的 controller INIT_STAND 為準。

確認 joint order 正確後，才允許低功率：

```bash
ros2 run redrhex_rl_controller motor_command_tool init-stand \
  --enable \
  --confirm-risk \
  --duration 2.0
```

此時你要用非常低的電源/限流，手放在急停旁邊。期待現象：

```text
ABAD 回到 0 rad
main drive 若由 controller 執行 INIT_STAND，會用低速 velocity command 回到右側 +45 deg、左側 -45 deg 附近
damper 不會出現在 motor command，因為真機沒有 damper 馬達
沒有抖動、沒有暴衝、沒有異常電流
```

如果任何一顆馬達方向相反，立刻急停，回來改 `main_drive_sign` / `abad_sign` / encoder zero offset。不要用 policy 硬扛方向錯誤。

### 10. 單顆 ABAD 測試

架空，低功率，先測第 0 顆 ABAD，小角度 0.10 rad：

```bash
ros2 run redrhex_rl_controller motor_command_tool single-abad \
  --enable \
  --confirm-risk \
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
  --confirm-risk \
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
  --confirm-risk \
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
  --confirm-risk \
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
  --confirm-risk \
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

此時還不要 enable policy。先確認機器人已經架空、低階板限流、急停在手邊，再啟動完整 bringup：

```bash
ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py use_fake_sensors:=false
```

你需要真實 `/imu/data` 和 `/joint_states` 已經在跑。確認：

```bash
ros2 topic hz /imu/data
ros2 topic hz /joint_states
ros2 topic echo /redrhex/state_machine_state
```

你的 `/joint_states` 可以只包含 main drive 6 顆：

```text
Revolute_15
Revolute_7
Revolute_12
Revolute_18
Revolute_23
Revolute_24
```

如果 `velocity` 欄位沒有資料，controller 會用 encoder position 差分估算 main drive velocity。ABAD 沒有 encoder feedback 時不要硬塞假 encoder topic；讓 `abad_feedback_source: "commanded"` 保持預設即可。

期待 state 到：

```text
POLICY_READY
```

如果卡住，看 diagnostics：

```bash
ros2 topic echo /redrhex/diagnostics
```

預設馬達輸出仍然是關的。你應該先看 command 內容：

```bash
ros2 topic echo /redrhex/motor_commands
```

確認 joint order、position target、velocity target、kp/kd 都合理後，才打開馬達輸出，讓 controller 持續送 INIT_STAND / POLICY_READY 的站姿命令。因為 main drive 真機是 PWM/速度控制，INIT_STAND 對 main drive 不是直接 position control，而是用 main encoder 位置誤差轉成低速 velocity command；ABAD 則送 position command。

```bash
ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: true}"
```

如果這個 request 被拒絕，通常是 state 還在 `SENSOR_CHECK`、`MOTOR_IDLE`、`PROTECTIVE_STOP`，或 `/estop=true`。不要硬重送，先修 sensor / heartbeat / diagnostics。

如果任何一顆馬達抖動、方向錯、電流暴增，立刻：

```bash
ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: false}"
ros2 topic pub --once /estop std_msgs/msg/Bool "{data: true}"
```

### 13.5 如果馬達方向或 encoder zero 錯了

不要用 policy 硬跑。先回 YAML 改 calibration hook：

```bash
nano /home/jetson/RedRhex/ros2_ws/src/redrhex_rl_controller/config/redrhex_policy.yaml
```

可調欄位：

```yaml
action:
  main_drive_sign: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  abad_sign: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  main_drive_zero_offset_rad: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  abad_zero_offset_rad: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

例子：

```text
第 0 顆 ABAD 正方向相反：把 abad_sign 第一個改成 -1.0
第 2 顆 main drive encoder zero 差 +0.12 rad：把 main_drive_zero_offset_rad 第三個改成 0.12
```

改完要重新 launch controller。不要在 controller 正在跑時改 YAML 期待它自動生效。

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

如果你只是要看 policy 計算，不想讓馬達出力，先關掉 motor output：

```bash
ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: false}"
```

等 `/redrhex/policy_action_safe` 和 `/redrhex/motor_commands` 都看起來合理，架空且限流後，再打開：

```bash
ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: true}"
```

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
[ ] /joint_states rate 穩定，至少包含 6 顆 main drive encoder position
[ ] main drive velocity 若無硬體回傳，encoder 差分估算正常
[ ] ABAD 無 feedback 的事實已接受，observation 使用 commanded ABAD estimate
[ ] /motor_feedback 若暫時沒有，YAML 保持 require_motor_feedback=false
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

收到 `/estop=true` 後，controller 會自動把 policy enable 和 motor output latch 都關掉。排除問題後，不要直接落地重試；先回到架空 INIT_STAND，再重新逐步 enable。

常見問題：

```text
一 enable 就暴衝：馬達方向或 encoder zero 錯，先不要跑 policy。
一直 SENSOR_CHECK：IMU 或 main drive joint_states topic 沒進來，或 main drive joint name 不匹配。
ONNX input 是 280：需要 history，controller 已支援，但你要確認 observation order。
落地完全不會走：base_lin_vel=0 造成 sim2real mismatch，需 odom/leg odometry。
側移很怪：ABAD sign 或 zero offset 很可能錯。
ABAD 看起來漂掉：因為 ABAD 無 encoder feedback，controller 只能相信上一個 command，需檢查 ABAD 低階位置控制是否真的到位。
```

快速判斷表：

```text
/redrhex/motor_commands.enable 一直是 false：
  正常情況：你還沒開 /redrhex/enable_motors。
  異常情況：state 不允許、E-stop active、或 controller 剛進 protective stop。

policy_action_raw 沒資料：
  代表還沒進 POLICY_RUN。檢查 /redrhex/enable_policy 和 state_machine_state。

policy_action_raw 有資料但 safe action / motor command 很怪：
  先關 /redrhex/enable_motors，不要讓馬達出力，再檢查 observation order 和 joint sign。

diagnostics 有 control loop deadline miss：
  降低 policy_hz 或檢查 Jetson CPU 負載，不要在 deadline miss 狀態落地。

motor_feedback 有 NaN / Inf：
  低階板 feedback parser 錯或 sensor 資料壞掉，controller 會進保護停止。
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
- 原始訓練 repo 先前沒有 ROS2 deployment package；本 MVP 已新增 `redrhex_msgs`、`redrhex_rl_controller`、`redrhex_lowlevel_bridge`
- 原始訓練 repo 目前沒有 ROS2 真機通訊層；本 MVP 已新增 mock、serial、sbRIO UDP skeleton，並新增可對接 `JasonLiaoJCS/RhexROS2` 的 `rinbo_ros` backend
- 依你目前硬體描述，真機只有 IMU 與 6 顆 main drive encoder 是真實回授；ABAD observation 使用 commanded estimate；damper 不送 motor command

## 仍未知 / TODO

- 你已提供真機 ROS2 repo `JasonLiaoJCS/RhexROS2`；目前優先使用 `rinbo_ros` backend 對接 `/motor/command` 與 `/motor/state`
- 若未來繞過 RhexROS2 直接連 sbRIO，LabVIEW RT/FPGA packet format 仍未最終確認，請使用 `sbrio_udp` skeleton 重新定義 protocol
- sbRIO 到馬達驅動器的最終介面仍需確認：FPGA PWM / CAN / EtherCAT / custom IO / 其他
- 真實馬達方向、encoder zero offset、joint axis sign 尚未確認
- base linear velocity estimator 尚未完成；初期 `base_lin_vel=0` 只能 bench test
- 真機 IMU frame 是否與 IsaacLab base frame 一致尚未驗證
- 真機 ABAD/main drive torque/current/temperature limit 若 sbRIO 或 motor driver 可提供，需依硬體標定；若無回授，必須靠 sbRIO 端 watchdog/限流保護

## Jetson 安裝假設

- Jetson Orin Nano
- Ubuntu 22.04 / JetPack 6.x
- ROS2 Humble
- Python 3.10

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-pip ros-humble-desktop
python3 -m pip install "numpy<2" onnx onnxruntime pyserial pyyaml
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

預設不會進 `POLICY_RUN`，而且 motor output 是關的。要明確允許 policy 計算：

```bash
ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: true}"
```

mock mode 下通常不需要打開馬達輸出；你可以確認 `/redrhex/motor_commands.enable` 仍是 `false`。只有在架空真機、限流、急停就緒後，才使用：

```bash
ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: true}"
```

注意：當 `/redrhex/enable_motors=false` 時，controller 只做 policy 計算與 command preview，不會更新 ABAD commanded observation estimate。這避免 ABAD 沒有真的動時，observation 卻假裝它已經到位。

## 切換 low-level bridge

Mock：

```yaml
redrhex_lowlevel_bridge:
  ros__parameters:
    backend: "mock"
```

RhexROS2 / rinbo_ros adapter，這是你目前真機最應該先用的 backend：

```yaml
redrhex_lowlevel_bridge:
  ros__parameters:
    backend: "rinbo_ros"
    rinbo:
      command_topic: "/motor/command"
      state_topic: "/motor/state"
      joint_state_topic: "/joint_states"
      preview_topic: "/redrhex/rinbo_motor_command_preview"
      publish_preview: true
      allow_enable: false
      publish_when_disabled: false
      disabled_servo_control_mode: 0
      require_state: true
      state_timeout_s: 0.25
      main_position_counts_per_rev: 54984.83
      main_pwm_per_rad_s: 120.0
      main_max_pwm: 500.0
      main_encoder_zero_counts_rinbo_order: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      main_encoder_sign_rinbo_order: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
      main_velocity_sign_policy_order: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      main_direction_positive_rinbo_order: [false, false, false, true, true, true]
      abad_encoder_zero_rinbo_order: [739, 2566, 3283, 1945, 2070, 987]
      abad_encoder_counts_per_rad: 1000.0
      abad_encoder_min: 0
      abad_encoder_max: 65535
      abad_sign_rinbo_order: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      servo_control_mode: 2
```

`rinbo_ros` 需要先 build/source 你的 `RhexROS2`，因為 runtime 需要 `rinbo_msgs`。它會把 `/redrhex/motor_commands` 轉成 `/motor/command`，並把 `/motor/state` 轉成只含 6 顆 main encoder 的 `/joint_states`。RhexROS2 的 `MotorCmdStamped` leg 欄位目前實際使用 PWM 式 `voltage/direction`，不是標準速度閉環，所以 `main_pwm_per_rad_s` 一定要架空慢慢調。方向、encoder sign、ABAD servo sign 都從 YAML 調，不要改 policy。

Serial skeleton：

```yaml
redrhex_lowlevel_bridge:
  ros__parameters:
    backend: "serial"
    serial:
      port: "/dev/ttyUSB0"
      baudrate: 921600
      timeout_s: 0.005
      allow_enable: false
```

Serial 封包目前是 provisional，包含 magic header、version、sequence、timestamp、joint count、target arrays、enable、CRC。`allow_enable=false` 時，serial backend 會拒絕 enabled motor command；MCU protocol 確定後請替換 `serial_bridge.py`，再有意識地打開 `allow_enable`。

sbRIO UDP skeleton：

```yaml
redrhex_lowlevel_bridge:
  ros__parameters:
    backend: "sbrio_udp"
    sbrio:
      remote_host: "192.168.0.2"
      command_port: 15000
      bind_host: "0.0.0.0"
      feedback_port: 15001
      timeout_s: 0.002
      heartbeat_timeout_s: 0.25
      allow_enable: false
      require_feedback: false
```

sbRIO 這條路線建議你在 LabVIEW RT 端實作兩種封包：

```text
Command packet Jetson -> sbRIO：
  magic RRHX
  protocol version
  packet type = command
  sequence id
  timestamp
  joint count
  enable
  target_position_rad[]
  target_velocity_rad_s[]
  kp[]
  kd[]
  effort_limit_nm[]
  CRC32

Feedback / heartbeat packet sbRIO -> Jetson：
  magic RRHX
  protocol version
  packet type = heartbeat 或 feedback
  sequence id
  timestamp
  joint count
  position_rad[]
  velocity_rad_s[]
  effort_nm[]
  current_a[]
  temperature_c[]
  fault[]
  CRC32
```

目前 `sbrio_udp_bridge.py` 是 provisional packet skeleton，不是最終硬體 protocol。真機前 sbRIO 端一定要有自己的 watchdog：如果超過 100 ms 沒收到新 command、CRC 錯、sequence 跳太多、E-stop active、或 command enable false，就必須讓馬達進安全狀態。Jetson 端的 safety filter 不能取代 sbRIO 端 watchdog。

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

真機端不加訓練噪音，也不再靜默 clamp observation；若 NaN/Inf 會直接進 protective stop。你的真機 observation 來源如下：

```text
base_lin_vel：目前 zero，僅 bench/低速測試可接受
base_ang_vel：IMU gyro
projected_gravity：IMU orientation
main_drive_pos_sin/cos：6 顆 main encoder position
main_drive_vel：若 /joint_states.velocity 沒有，使用 main encoder position 差分
abad_pos：上一個下達的 ABAD position command
abad_vel：ABAD command 差分
velocity_command：/cmd_vel saturation
gait_phase：controller oscillator
last_actions：上一筆 policy raw action
```

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

Damper joints, simulation-only spring legs, not real motors and not sent in `/redrhex/motor_commands`:

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
- damper joints are simulation-only spring-leg joints on the real robot and are excluded from motor command
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
- RhexROS2 ROS2 message protocol 已可對接，但 sbRIO/CORE 內部 watchdog、限流、PWM 標定、servo encoder scale 仍需真機確認
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
