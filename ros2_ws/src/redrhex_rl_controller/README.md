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

## 現場快速總覽

如果你已經照前面步驟 build 好 `~/rinbo_ros_ws` 和本 repo 的 `ros2_ws`，現場順序永遠照這張表，不要跳：

```text
1. sbRIO terminal：只啟動一個 grpccore + 一個 fpga_driver
2. Orin terminal A：source rinbo_ros_ws，設定 CORE_MASTER_ADDR/CORE_LOCAL_IP，啟動 rinbo_ros_bridge
3. Orin terminal B：source 兩個 workspace，跑 biorola_bringup_check
4. Orin terminal B：用 biorola_power_tool digital -> sensors -> relay
5. Orin terminal C：跑 rinbo_cali，再跑 rinbo_standing，跑完關掉 rinbo_fsm
6. Orin terminal D：啟動 redrhex_lowlevel_bridge backend=biorola_ros，只看 preview 和 heartbeat
7. Orin terminal E：啟動 redrhex_rl_controller，先只 policy dry-run，不開 enable_motors
8. 架空、限流、E-stop 在手上，才允許單顆馬達測試
```

最常用的檢查指令：

```bash
ros2 run redrhex_lowlevel_bridge biorola_bringup_check --message-timeout-s 5.0
ros2 run redrhex_lowlevel_bridge biorola_power_tool status
ros2 topic echo /redrhex/rinbo_motor_command_preview --once
ros2 topic echo /redrhex/lowlevel_diagnostics --once
ros2 topic echo /redrhex/state_machine_state
```

今天現場如果只想先走最短安全路線，照這 6 行判斷，不要急著讓 policy 接管：

```text
1. biorola_bringup_check 沒有 ERROR，才碰 power。
2. biorola_power_tool status 讀得到 digital/signal/power，才碰 calibration。
3. rinbo_cali / rinbo_standing 完成，而且已經停止，才啟動 RedRhex bridge。
4. allow_enable=false 時先看 /redrhex/rinbo_motor_command_preview，確認 mapping。
5. allow_enable=true 後只測單顆 ABAD 或單顆 main，機器人必須架空。
6. 單顆測試正確後，才允許 /redrhex/enable_policy；最後才允許 /redrhex/enable_motors。
```

建議每個 Orin terminal 一開始先設定這些路徑變數，後面複製指令比較不會混亂：

```bash
export RINBO_WS=~/rinbo_ros_ws
export REDRHEX_WS=~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source $RINBO_WS/install/setup.bash
source $REDRHEX_WS/install/setup.bash
```

如果你明天到現場只想先產生一份「照抄指令表」，可以用新工具：

```bash
source /opt/ros/humble/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
ros2 run redrhex_lowlevel_bridge biorola_bringup_plan \
  --sbrio-ip 192.168.30.12 \
  --orin-ip 192.168.30.164 \
  --onnx-path /home/jetson/redrhex_models/policy.onnx
```

它只會印出分 terminal 的操作命令，不會 publish ROS topic，也不會碰馬達。確定急停、限流、架空都準備好之後，才加：

```bash
ros2 run redrhex_lowlevel_bridge biorola_bringup_plan \
  --sbrio-ip 192.168.30.12 \
  --orin-ip 192.168.30.164 \
  --onnx-path /home/jetson/redrhex_models/policy.onnx \
  --include-relay \
  --enable-hardware-snippets
```

現場建議開這幾個 PowerShell / terminal 分頁：

```text
sbRIO   ：ssh admin@<SBRIO_IP>，只管 grpccore / fpga_driver
Orin1   ：ssh jetson@<ORIN_IP>，只跑 rinbo_ros_bridge，啟動後不要關
Orin2   ：power、bringup_check、rinbo_cali、rinbo_standing
Orin3   ：redrhex_lowlevel_bridge + redrhex_rl_controller
Monitor ：只 echo diagnostics / state / preview，不送 command
```

如果你不知道下一步能不能做，用這張判斷表：

```text
ONNX check 沒過：不要開 ROS graph
biorola_bringup_check 沒看到 /motor/state：不要開 relay
power status 不是 digital=true signal=true power=true：不要校正
rinbo_cali / rinbo_standing 還在跑：不要啟動 RL controller
/motor/command publisher 超過 1：不要讓機器人動
/redrhex/lowlevel_heartbeat=false：不要 enable_motors
機器人沒有架空：不要做第一次 main drive 測試
```

最重要的禁止事項：

```text
不要同時開兩個 grpccore / fpga_driver
不要同時跑 rinbo_tripod 和 RL controller
不要在 allow_enable=false 還沒驗證 preview 前打開 enable_motors
不要把 policy raw action 直接送 /motor/command
```

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

還有一個 ROS2 軟體急停 topic：

```text
/estop：true 時 controller 會丟掉 policy / motor enable latch，並進入保護邏輯。
```

它不能取代實體急停，也不能取代 sbRIO / motor driver 自己的 watchdog；它只是 ROS2 高階控制器的保護輸入。現場任何 terminal 都可以先準備這兩行：

```bash
ros2 run redrhex_rl_controller estop_tool assert
ros2 run redrhex_rl_controller estop_tool clear --confirm-clear
```

`clear` 故意需要 `--confirm-clear`，避免你手滑解除軟體急停。第一次上真機時，實體急停永遠比軟體急停重要。

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
echo $ROS_DISTRO
which ros2
ros2 pkg list > /tmp/ros2_pkg_list.txt
sed -n '1,10p' /tmp/ros2_pkg_list.txt
```

你應該看到 `humble`，而且 `ros2 pkg list` 會列出一些 ROS2 package。注意：ROS2 Humble 的 `ros2` CLI 沒有 `ros2 --version`，出現 `unrecognized arguments: --version` 不是安裝壞掉。

也不要用 `ros2 pkg list | head` 當檢查指令；Humble 有時會因為 `head` 提早關閉 pipe 而顯示 `BrokenPipeError`，那只是輸出被截斷，不代表 ROS2 壞掉。

如果你的 Jetson 是 ROS2 Jazzy，把上面 `humble` 改成 `jazzy`，並確認 `echo $ROS_DISTRO` 顯示 `jazzy`。接著安裝工具：

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
cd ~/RedRhex/RedRhex/ros2_ws
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
nano ~/RedRhex/RedRhex/ros2_ws/src/redrhex_rl_controller/config/redrhex_policy.yaml
```

確認這行存在：

```yaml
policy:
  onnx_path: "/home/jetson/redrhex_models/policy.onnx"
```

### 3. 檢查 ONNX，不接任何硬體

這一步只是在 Jetson CPU 上跑一次 zero observation。這個 script 不需要先 `colcon build`，所以它是最早可以做的檢查。

```bash
cd ~/RedRhex/RedRhex
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
cd ~/RedRhex/RedRhex/ros2_ws
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
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
ros2 run redrhex_rl_controller preflight_check \
  --onnx /home/jetson/redrhex_models/policy.onnx \
  --config ~/RedRhex/RedRhex/ros2_ws/src/redrhex_rl_controller/config/redrhex_policy.yaml
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
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py \
  use_fake_sensors:=true \
  bridge_backend:=mock \
  onnx_path:=/home/jetson/redrhex_models/policy.onnx \
  enable_policy_on_start:=false \
  enable_motor_output_on_start:=false
```

Terminal B：

```bash
source /opt/ros/humble/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
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

fake command 也可以直接從 launch 覆蓋，不用另外 publish `/cmd_vel`：

```bash
ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py \
  use_fake_sensors:=true \
  bridge_backend:=mock \
  fake_cmd_vx:=0.10 \
  fake_cmd_vy:=0.00 \
  fake_cmd_wz:=0.00
```

想確認 controller launch 目前支援哪些覆蓋參數，可以跑：

```bash
ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py --show-args
```

現場最常用的是：

```text
onnx_path
use_fake_sensors
start_bridge
bridge_backend
enable_policy_on_start
enable_motor_output_on_start
base_lin_vel_source
abad_feedback_source
require_lowlevel_heartbeat
require_motor_feedback
```

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

### 7. 接 BioRoLaROS2 / sbRIO heartbeat，但仍不要插馬達主電源

你提供的真機 ROS2 repo 是 `JasonLiaoJCS/BioRoLaROS2`，它已經定義好 sbRIO 那邊使用的 ROS2 interface。這份 RedRhex deployment stack 現在支援 `biorola_ros` backend，專門對接你的 BioRoLaROS2。底層 message package 名仍叫 `rinbo_msgs`，所以工具內部仍使用 `/motor/*`、`/power/*` 和 `rinbo_msgs`。

```text
RedRhex RL controller
  -> /redrhex/motor_commands              redrhex_msgs/RedRhexMotorCommand
  -> redrhex_lowlevel_bridge backend=biorola_ros
  -> /motor/command                       rinbo_msgs/MotorCmdStamped
  -> BioRoLaROS2 rinbo_ros_bridge
  -> sbRIO / CORE / motor drivers

BioRoLaROS2 /motor/state                     rinbo_msgs/MotorStateStamped
  -> redrhex_lowlevel_bridge backend=biorola_ros
  -> /joint_states                        sensor_msgs/JointState, 只含 6 顆 main encoder
  -> ObservationBuilder
```

先把 BioRoLaROS2 workspace 準備好。你目前 Jetson 上的工作區是 `~/rinbo_ros_ws`，建議固定使用這個路徑。若已經有 `src/rinbo_msgs/package.xml`，可以直接 build；若還沒有完整 repo，才重新 clone：

```bash
cd ~/rinbo_ros_ws
source /opt/ros/humble/setup.bash
find src -maxdepth 6 -name package.xml | sort
```

如果 `rinbo_msgs`、`rinbo_ros_bridge`、`rinbo_fsm` 都存在，再 build：

```bash
cd ~/rinbo_ros_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ros2 interface show rinbo_msgs/msg/MotorCmdStamped
ros2 interface show rinbo_msgs/msg/MotorStateStamped
```

如果 `src` 裡沒有完整 package，先開一個乾淨的新 workspace 重新 clone，不要直接覆蓋你現場正在用的 `~/rinbo_ros_ws`：

```bash
mkdir -p ~/biorola_tmp_ws/src
cd ~/biorola_tmp_ws/src
git clone https://github.com/JasonLiaoJCS/BioRoLaROS2.git
cd ~/biorola_tmp_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

我已依照 `https://github.com/JasonLiaoJCS/BioRoLaROS2.git` 檢查目前 interface，部署端採用以下事實：

```text
ROS2 package 名稱：rinbo_msgs, rinbo_ros_bridge, rinbo_fsm
command topic：/motor/command，型別 rinbo_msgs/msg/MotorCmdStamped
state topic：/motor/state，型別 rinbo_msgs/msg/MotorStateStamped
power command：/power/command，型別 rinbo_msgs/msg/PowerCmdStamped
power state：/power/state，型別 rinbo_msgs/msg/PowerStateStamped
main leg order：l1, l2, l3, r1, r2, r3
servo order：sl1, sl2, sl3, sr1, sr2, sr3
main motor command：LegCmd.enable + direction + voltage，不是 target_velocity
ABAD/servo command：ServoCmd.position_encoder
BioRoLaROS2 rinbo_tripod counts/rev：54984.83
BioRoLaROS2 rinbo_cali/rinbo_standing local PID counts/rev：55296.0
stand-like servo targets：[740, 2565, 3283, 1944, 2071, 989]
```

因此本 repo 的 `biorola_ros` backend 仍使用 `rinbo.*` YAML namespace，因為外部 ROS2 message/package 名真的叫 `rinbo_msgs`。`rinbo_ros` 也保留為相容 alias。

我也在 BioRoLaROS2 repo 目前版本確認到一個重要細節：

```text
src/rinbo_ros_bridge/src/rinbo_ros_bridge.cpp
  setenv("CORE_IP", "192.168.30.12", 1);
```

也就是說，如果你的 sbRIO IP 不是 `192.168.30.12`，只在 terminal export `CORE_MASTER_ADDR` / `CORE_LOCAL_IP` 可能還不夠，因為 bridge 程式本身可能覆蓋 `CORE_IP`。如果 `biorola_bringup_check` 的 `TCP 50051` 或 `/motor/state` 一直失敗，請回 BioRoLaROS2 檢查這一行，改成你的 sbRIO IP 後重新 build：

```bash
cd ~/rinbo_ros_ws
nano src/rinbo_ros_bridge/src/rinbo_ros_bridge.cpp
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

如果你使用的是 `~/rinbo_ros_ws`，但 `find src -maxdepth 6 -name package.xml` 完全沒有輸出，代表 `src/rinbo_msgs` 只是空資料夾或不完整資料夾，還不是 ROS2 package。先檢查：

```bash
cd ~/rinbo_ros_ws
find src/rinbo_msgs -maxdepth 3 -type f | sort
ls -la src/rinbo_msgs
```

正常的 `rinbo_msgs` 至少要有：

```text
package.xml
CMakeLists.txt
msg/MotorCmdStamped.msg
msg/MotorStateStamped.msg
```

如果沒有這些檔案，請重新把 BioRoLaROS2 repo 正確 clone 到新的 workspace，或把完整的 `rinbo_msgs` package 複製進 `~/rinbo_ros_ws/src/rinbo_msgs` 後再 build。

如果只是在測 RedRhex adapter，先 build `rinbo_msgs` 就夠了。要真的連 sbRIO，還需要你的 BioRoLaROS2 `rinbo_ros_bridge` 能成功 build 和執行；這可能需要原本 repo 裡的 gRPC / CORE 環境。

```bash
cd ~/rinbo_ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export CORE_MASTER_ADDR="192.168.30.12:50051"
export CORE_LOCAL_IP="192.168.30.164"
ros2 run rinbo_ros_bridge rinbo_ros_bridge
```

如果你使用 BioRoLaROS2 的 `start_bridge.sh`，請先確認裡面的 workspace 路徑就是 `~/rinbo_ros_ws/install/setup.bash`。不確定時先不用 script，直接照上面的手動指令啟動，比較容易看出是哪一步錯。

#### R-Slip / sbRIO 真機啟動順序

你現有 R-Slip 流程裡真正重要的依賴順序是：

```text
sbRIO: grpccore + fpga_driver
  -> Orin: rinbo_ros_bridge
  -> Orin: /power/command digital
  -> Orin: /power/command signal
  -> Orin: /power/command power relay
  -> calibration / standing
  -> RL controller 接管
```

一次只能有一個 `grpccore` 和一個 `fpga_driver`。如果你不確定是否有殘留，先在 sbRIO 上查：

```bash
ps -ef | egrep "grpccore|fpga_driver" | grep -v grep
```

如果真的有殘留且你確定要清掉：

```bash
pkill -f grpccore
pkill -f fpga_driver
```

在 sbRIO terminal：

```bash
ssh admin@<SBRIO_IP>
cd ~/rinbo_sbRIO_ws/rinbo_fpga_driver/build
export CORE_LOCAL_IP=<SBRIO_IP>
export CORE_MASTER_ADDR=<SBRIO_IP>:50051
nohup /home/admin/rinbo_sbRIO_ws/install/bin/grpccore >/tmp/grpccore.log 2>&1 &
nohup /home/admin/rinbo_sbRIO_ws/rinbo_fpga_driver/build/fpga_driver >/tmp/fpga_driver.log 2>&1 &
ps -ef | egrep "grpccore|fpga_driver" | grep -v grep
netstat -tn | grep 50051 || ss -tn | grep 50051 || echo "NO TCP on 50051"
```

如果你的 sbRIO 已經有 `start_fpga_driver.sh`，可以用它取代上面手動指令；但要記得 `chmod +x start_fpga_driver.sh`，而且 IP 變動時要更新 script。編輯 script 時不要在 `export` 的等號旁邊加空白。

在 Orin terminal 啟動 BioRoLaROS2 bridge：

```bash
cd ~/rinbo_ros_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
export CORE_MASTER_ADDR=<SBRIO_IP>:50051
export CORE_LOCAL_IP=<ORIN_IP>
ros2 run rinbo_ros_bridge rinbo_ros_bridge
```

注意 `export CORE_LOCAL_IP=<ORIN_IP>` 中間不能有空白，不要寫成 `CORE_LOCAL_IP = ...`。Orin IP 可先用：

```bash
hostname -I
```

BioRoLaROS2 bridge 啟動後，先在另一個 Orin terminal 跑本 repo 的檢查工具。它不會送任何馬達命令，只會檢查環境、topic 和 state message：

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
export CORE_MASTER_ADDR=<SBRIO_IP>:50051
export CORE_LOCAL_IP=<ORIN_IP>
ros2 run redrhex_lowlevel_bridge biorola_bringup_check --message-timeout-s 5.0
```

你希望看到：

```text
[OK   ] CORE_MASTER_ADDR
[OK   ] rinbo_msgs
[OK   ] BioRoLaROS2 message contract
[OK   ] topic /motor/state
[OK   ] subscribers /power/command
[OK   ] publishers /motor/state
[OK   ] /motor/state: message received
```

`BioRoLaROS2 message contract` 會檢查你 source 到的 `rinbo_msgs` 是否真的有：

```text
MotorCmdStamped.l1..r3.enable/direction/voltage/state/reset_position
MotorCmdStamped.sl1..sr3.position_encoder
MotorStateStamped.l1..r3.position/tick_count/hall_effect
MotorStateStamped.sl1..sr3.position_encoder
PowerCmdStamped.digital/signal/power/clean/trigger
PowerStateStamped.digital/signal/power/clean/v_0/i_0 ... v_7/i_7
```

如果這一項是 ERROR，代表你 source 到的 workspace 不是目前這份 BioRoLaROS2 interface，先不要跑 bridge 或 RL。

`biorola_bringup_check` 也會檢查 `/motor/command` 的 publisher 數量。如果看到 `publishers /motor/command` 超過 1，代表很可能同時開了 `rinbo_tripod`、`rinbo_standing`、RedRhex low-level bridge 或其他控制節點。這時不要繼續，先把多餘節點關掉，因為兩個節點同時送 `/motor/command` 會讓機器人行為不可預測。

如果 `TCP 50051` 是 WARN，先確認 sbRIO 上 `grpccore` 是否只有一個、`fpga_driver` 是否只有一個、IP 是否正確。如果 `/motor/state` 沒有 message，不要進 power relay，也不要跑 RL。

如果 `CORE_LOCAL_IP` 顯示 WARN，通常是 Orin IP 填錯。先在 Orin 跑：

```bash
hostname -I
```

把看到的 Orin Wi-Fi / Ethernet IP 填到：

```bash
export CORE_LOCAL_IP=<ORIN_IP>
```

等號左右不要加空格。

如果 `CORE_IP` 顯示 WARN，通常是 BioRoLaROS2 bridge 內部的 `CORE_IP` 與你設定的 sbRIO IP 不一致。這時請先檢查 `rinbo_ros_bridge.cpp` 的 `setenv("CORE_IP", "...")`，不要靠重開 terminal 解決。

現在 `biorola_bringup_check` 也會自動掃這個檔案：

```text
~/rinbo_ros_ws/src/rinbo_ros_bridge/src/rinbo_ros_bridge.cpp
```

如果它發現 `setenv("CORE_IP", "...")` 和 `CORE_MASTER_ADDR` 的 IP 不同，會直接顯示 `BioRoLaROS2 hardcoded CORE_IP: ERROR`。這個 ERROR 不能忽略，因為 terminal export 的 IP 可能被 bridge 程式本身覆蓋。你的 workspace 路徑如果不同，可以手動指定：

```bash
ros2 run redrhex_lowlevel_bridge biorola_bringup_check \
  --message-timeout-s 5.0 \
  --bridge-source ~/你的_ws/src/rinbo_ros_bridge/src/rinbo_ros_bridge.cpp
```

Orin 另一個 terminal 開 power，建議用本 repo 的 helper，避免手打很長的 YAML：

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash

# Step 1: digital=true, signal=false, power=false
ros2 run redrhex_lowlevel_bridge biorola_power_tool digital

# Step 2: digital=true, signal=true, power=false
ros2 run redrhex_lowlevel_bridge biorola_power_tool sensors

# Step 3: digital=true, signal=true, power=true，這步才會開繼電器
ros2 run redrhex_lowlevel_bridge biorola_power_tool relay --confirm-relay

# Step 4: 讀回 /power/state，確認真實狀態
ros2 run redrhex_lowlevel_bridge biorola_power_tool status
```

`biorola_power_tool` 會先等 `/power/command` 有 subscriber，通常就是 `rinbo_ros_bridge`。如果你看到 `No subscriber on /power/command`，代表 bridge 沒起來或 workspace 沒 source 對，不要硬加 `--allow-no-subscriber`。

也可以用 sequence；預設只開 digital + sensors，不開 relay：

```bash
ros2 run redrhex_lowlevel_bridge biorola_power_tool sequence
ros2 run redrhex_lowlevel_bridge biorola_power_tool sequence --include-relay --confirm-relay
```

你原本的手動 power 指令仍然可以用，但不建議每天手打。若要 dry-run 查看即將送出的 power command：

```bash
ros2 run redrhex_lowlevel_bridge biorola_power_tool relay --dry-run
ros2 run redrhex_lowlevel_bridge biorola_power_tool sequence --include-relay --dry-run
```

`--dry-run` 不需要 `rinbo_msgs` 已經 source；它只印出即將發布的欄位，不會碰 ROS topic。真正發布 `relay` 時才需要 `--confirm-relay`。

`biorola_power_tool status` 會讀 `/power/state`，除了 digital/signal/power，也會列出 BioRoLaROS2 `PowerStateStamped` 裡 8 路電壓/電流欄位 `v_0/i_0 ... v_7/i_7`。如果 relay 已開但所有 voltage/current 都是 0，先確認 sbRIO bridge 是否真的在回 power state，不要直接進馬達測試。

開完 power 後可以再跑一次檢查，這次要求 `/power/state`：

```bash
ros2 run redrhex_lowlevel_bridge biorola_bringup_check --message-timeout-s 5.0 --require-power-state
```

校正和站姿仍可使用既有 BioRoLaROS2 FSM：

```bash
ros2 run rinbo_fsm rinbo_cali
ros2 run rinbo_fsm rinbo_standing
```

`rinbo_tripod` 是原本 R-Slip/手寫步態測試，不要和 RL controller 同時跑，因為它也會 publish `/motor/command`。要測 RL 時，完成 calibration / standing 後，關掉 `rinbo_fsm` 節點，只保留 `rinbo_ros_bridge` 與本 repo 的 low-level bridge/controller。

BioRoLaROS2 bridge 起來後，先確認 topic 有出現：

```bash
ros2 topic list | grep motor
ros2 topic echo /motor/state --once
```

然後打開 RedRhex bridge config：

```bash
nano ~/RedRhex/RedRhex/ros2_ws/src/redrhex_lowlevel_bridge/config/lowlevel_bridge.yaml
```

把 backend 改成 `biorola_ros`：

```yaml
redrhex_lowlevel_bridge:
  ros__parameters:
    backend: "biorola_ros"
    rinbo:
      command_topic: "/motor/command"
      state_topic: "/motor/state"
      joint_state_topic: "/joint_states"
      preview_topic: "/redrhex/rinbo_motor_command_preview"
      publish_preview: true
      allow_enable: false
      publish_when_disabled: false
      disabled_servo_control_mode: 0
      publish_shutdown_disable: true
      shutdown_disable_repeats: 5
      shutdown_disable_period_s: 0.02
      require_state: true
      block_if_duplicate_command_publishers: true
      state_timeout_s: 0.25
      main_position_counts_per_rev: 54984.83
      main_pwm_per_rad_s: 120.0
      main_max_pwm: 500.0
      main_encoder_zero_counts_rinbo_order: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      main_encoder_sign_rinbo_order: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
      main_velocity_sign_policy_order: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      main_direction_positive_rinbo_order: [true, true, true, false, false, false]
      main_velocity_filter_alpha: 0.35
      main_velocity_max_dt_s: 0.20
      main_velocity_clip_rad_s: 80.0
      abad_encoder_zero_rinbo_order: [740, 2565, 3283, 1944, 2071, 989]
      abad_encoder_counts_per_rad: 1000.0
      abad_encoder_min: 0
      abad_encoder_max: 65535
      abad_sign_rinbo_order: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      servo_control_mode: 2
```

`allow_enable: false` 一定先保持 false。這代表 adapter 會接收 RL command、做轉換、發布 diagnostics，但拒絕任何 enabled motor command。等你完成急停、限流、單顆馬達測試後，才可以改成 true。

`preview_topic` 是安全預覽 topic。當 `publish_preview: true` 時，adapter 會把轉換後的 `rinbo_msgs/MotorCmdStamped` 發到 `/redrhex/rinbo_motor_command_preview`，讓你在 `allow_enable=false` 時也能檢查 `l1/l2/.../sr3` 的 PWM、direction、servo encoder。BioRoLaROS2 bridge 不會訂閱這個 topic，所以它不會讓馬達動。

`publish_when_disabled: false` 也先保持 false。BioRoLaROS2 的 ABAD servo command 沒有 per-servo enable 欄位；如果 disabled command 也 publish，servo 仍可能吃到 `position_encoder`。所以 dry-run 階段只看 `/redrhex/motor_commands`，不要讓 adapter publish `/motor/command`。如果 adapter 前一包已經是 enabled，下一包 disabled 仍會被送出一次，確保 main legs release。等你確認伺服電源斷開、或確認 `servo_control_mode=0` 真的不會動，再暫時打開它做 message-level 測試。

`publish_shutdown_disable: true` 代表你 Ctrl-C 關掉 RedRhex bridge 時，adapter 會補送幾包 disabled command 到 `/motor/command`。這是軟體保險，不能取代 sbRIO watchdog 或實體急停；但它可以降低「節點關掉後低階端保留上一包 PWM」的風險。

`require_state: true` 代表 `/redrhex/lowlevel_heartbeat` 只有在真的收到 `/motor/state` 時才會是 true。如果你只是離線測 message conversion，沒有啟動 BioRoLaROS2 bridge，可以暫時改成 false，但真機測試要改回 true。

`block_if_duplicate_command_publishers: true` 代表只要 `/motor/command` 上同時有超過一個 publisher，adapter 會擋住 enabled command。這是專門防止 `rinbo_tripod`、`rinbo_standing` 或另一個 RL bridge 還活著時，兩個節點同時搶 sbRIO 控制權。

請用三個 terminal 分開跑。

Terminal A：BioRoLaROS2 bridge

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
export CORE_MASTER_ADDR="192.168.30.12:50051"
export CORE_LOCAL_IP="192.168.30.164"
ros2 run rinbo_ros_bridge rinbo_ros_bridge
```

Terminal B：RedRhex low-level adapter

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py \
  backend:=biorola_ros \
  rinbo_allow_enable:=false \
  rinbo_require_state:=true \
  rinbo_block_if_duplicate_command_publishers:=true
```

Terminal C：檢查 heartbeat 和 encoder 是否進來

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
ros2 topic echo /redrhex/lowlevel_heartbeat
ros2 topic echo /joint_states --once
ros2 topic echo /motor_feedback --once
ros2 topic echo /redrhex/rinbo_motor_command_preview --once
ros2 topic echo /redrhex/lowlevel_diagnostics --once
```

你在 `/joint_states` 應該只看到 6 個 main drive joint：

```text
Revolute_15, Revolute_7, Revolute_12, Revolute_18, Revolute_23, Revolute_24
```

這是正確的。ABAD 沒有 feedback，所以 observation 用上一筆 ABAD command 當 estimate；damper 是模擬彈簧腳，不會出現在真機 command。

BioRoLaROS2 的 leg/servo order 和 policy order 不一樣，目前 adapter 內部這樣轉：

```text
BioRoLaROS2 main leg order: l1, l2, l3, r1, r2, r3
Policy main order:      RF, RM, RR, LF, LM, LR

BioRoLaROS2 servo order:   sl1, sl2, sl3, sr1, sr2, sr3
Policy ABAD order:      RF, RM, RR, LF, LM, LR
```

main drive policy velocity 會轉成 BioRoLaROS2 的 `LegCmd.voltage` 和 `LegCmd.direction`。你目前的 BioRoLaROS2 bridge 只把 `enable/direction/voltage/state/reset_position` 傳到 gRPC；`target_velocity` 欄位在 bridge 裡沒有送出去，所以 RL 端不能假設 sbRIO 會做速度閉環。現在的 `main_pwm_per_rad_s` 是保守初值，真正走路前一定要架空調整。`main_velocity_sign_policy_order`、`main_direction_positive_rinbo_order`、`main_encoder_sign_rinbo_order` 都是為真機方向校正準備的，不要在 Python 裡硬改。

BioRoLaROS2 裡有一個容易混淆的地方：`rinbo_tripod.cpp` 對 `LegCmd.direction` 使用 `[true,true,true,false,false,false]`，但 `rinbo_cali.cpp` / `rinbo_standing.cpp` 內部 PID helper 使用相反 convention。本 repo 的 `biorola_ros` 預設先採用 `rinbo_tripod` 的 locomotion convention：

```yaml
main_direction_positive_rinbo_order: [true, true, true, false, false, false]
```

這不是保證一定符合你當天 sbRIO firmware 的真實方向，所以 `allow_enable=false`、preview、單顆馬達架空測試不能跳過。

adapter 會用 `/motor/state` 的 encoder position 差分估算 main drive velocity，並把它放進 `/joint_states.velocity` 和 `/motor_feedback.velocity_rad_s`。如果你看到 velocity 抖很大，可以先調：

```yaml
rinbo:
  main_velocity_filter_alpha: 0.35
  main_velocity_max_dt_s: 0.20
  main_velocity_clip_rad_s: 80.0
```

`main_velocity_filter_alpha` 越大越相信最新差分，反應快但更抖；越小越平滑但延遲變大。真機第一次測試先不要靠調大 alpha 追求漂亮反應，先確認 encoder sign 和方向正確。

ABAD position 會轉成 `ServoCmd.position_encoder`。`abad_encoder_zero_rinbo_order` 目前使用 BioRoLaROS2 既有站姿附近數值作初始值，不等於你的最終機械零點；`abad_encoder_counts_per_rad` 和 `abad_sign_rinbo_order` 也必須實測。ABAD 沒有 feedback 時，這些數字錯了，policy 看到的 ABAD observation 就會錯。

#### BioRoLaROS2 adapter 校正順序

不要一開始就跑 policy。先只校正 adapter：

1. `allow_enable=false`、`publish_when_disabled=false`，先確認 `/redrhex/motor_commands` 與 `/redrhex/rinbo_motor_command_preview`，不讓 adapter publish enabled command 到 `/motor/command`。
2. 啟動 BioRoLaROS2 bridge，只看 `/motor/state` 是否穩定，確認 `require_state=true` 時 `/redrhex/lowlevel_heartbeat` 會變 true。
3. 手轉每一顆 main drive 一小段，檢查 `/joint_states.position` 對應的 policy joint 是否變動，也看 `/joint_states.velocity` 是否方向合理。如果 position 符號反了，只改 `main_encoder_sign_rinbo_order`；如果 command 正速度時實體方向反了，再改 velocity/PWM 方向設定。
4. 架空、限流、只測單顆 main drive。若 RL/工具命令正速度時實體腿方向反了，優先改 `main_velocity_sign_policy_order`；若只是 `LegCmd.direction` boolean 與 sbRIO 韌體定義相反，再改 `main_direction_positive_rinbo_order`。
5. 斷 main drive，只測單顆 ABAD 小角度。如果正角度反向，改 `abad_sign_rinbo_order`；如果中立角不對，改 `abad_encoder_zero_rinbo_order`；如果角度比例不對，改 `abad_encoder_counts_per_rad`。
6. 每改一次 YAML 都重新 build/source 或重啟 launch，並看 `/redrhex/lowlevel_diagnostics` 裡的 `rinbo_actual_publish_state`、`rinbo_last_pwm_l1_l2_l3_r1_r2_r3`、`rinbo_last_abad_sl1_sl2_sl3_sr1_sr2_sr3` 與 `rinbo_main_vel_policy_order_rad_s`。

如果你 publish 了 `--enable` 的手動命令，但 `allow_enable=false`，你會看到：

```text
/redrhex/rinbo_motor_command_preview：有轉換後的 PWM / servo target
/motor/command：不會收到 enabled command
/redrhex/lowlevel_diagnostics：rinbo_actual_publish_state=blocked_allow_enable
```

這是正確的安全狀態。

如果未來你已經把 `allow_enable=true`，但 `/motor/state` 中斷或超過 `rinbo.state_timeout_s`，backend 仍會拒絕 enabled command，diagnostics 會顯示：

```text
rinbo_actual_publish_state=blocked_no_recent_state
```

這代表 sbRIO/BioRoLaROS2 state feedback 沒有穩定進來。先修 heartbeat，不要提高 PWM 或重複 enable。

如果 diagnostics 顯示：

```text
rinbo_actual_publish_state=blocked_duplicate_publishers
```

代表 `/motor/command` 上不只一個 publisher。先跑：

```bash
ros2 topic info /motor/command -v
```

把 `rinbo_tripod`、`rinbo_standing`、重複啟動的 bridge 或其他控制節點關掉，只留下 RedRhex low-level bridge 一個 publisher。

如果 diagnostics 顯示：

```text
rinbo_actual_publish_state=blocked_no_command_subscriber
```

代表 BioRoLaROS2 的 `rinbo_ros_bridge` 沒有在訂閱 `/motor/command`。先修 bridge/source/IP，不要用 `allow_enable=true` 硬推。

不要同時跑 BioRoLaROS2 既有的 `rinbo_fsm` tripod/standing 節點和 RedRhex RL controller。兩邊都會 publish `/motor/command`，會互相搶控制權。

如果你暫時不走 BioRoLaROS2，而是要測本 repo 內的 sbRIO UDP skeleton：

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

這條 UDP skeleton 只適合你未來自己寫 LabVIEW RT/FPGA protocol 時使用。既然你目前已有 BioRoLaROS2，明天優先用 `biorola_ros`。

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

也可以不用改 YAML，直接在 launch 時覆蓋最常用的安全參數。這是現場比較推薦的方式，因為你可以從 terminal 歷史清楚看到當時有沒有允許硬體輸出：

```bash
ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py \
  backend:=biorola_ros \
  rinbo_allow_enable:=false \
  rinbo_require_state:=true \
  rinbo_block_if_duplicate_command_publishers:=true
```

第一次架空單顆馬達測試時，才把 `rinbo_allow_enable` 改成 true：

```bash
ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py \
  backend:=biorola_ros \
  rinbo_allow_enable:=true \
  rinbo_require_state:=true \
  rinbo_block_if_duplicate_command_publishers:=true \
  rinbo_main_pwm_per_rad_s:=80.0 \
  rinbo_main_max_pwm:=200.0
```

這裡故意把 `rinbo_main_pwm_per_rad_s` 和 `rinbo_main_max_pwm` 調低，因為第一次 main drive 測試要慢，不是要走。

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
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
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

`motor_command_tool` 預設會先等 `/redrhex/motor_commands` 有 subscriber，也就是 `redrhex_lowlevel_bridge` 已經啟動；如果你加了 `--enable`，它還會等待 `/redrhex/lowlevel_heartbeat=true`。如果它拒絕發布，先修 bridge/heartbeat，不要用 `--allow-no-subscriber` 或 `--skip-heartbeat-check` 硬跳。

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

如果你只是要看即將送出的 command，不碰 ROS graph：

```bash
ros2 run redrhex_rl_controller motor_command_tool single-main-velocity \
  --dry-run \
  --index 0 \
  --velocity 0.3
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
ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py \
  use_fake_sensors:=false \
  start_bridge:=false \
  onnx_path:=/home/jetson/redrhex_models/policy.onnx \
  enable_policy_on_start:=false \
  enable_motor_output_on_start:=false \
  base_lin_vel_source:=zero \
  abad_feedback_source:=commanded
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
nano ~/RedRhex/RedRhex/ros2_ws/src/redrhex_rl_controller/config/redrhex_policy.yaml
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
- 原始訓練 repo 目前沒有 ROS2 真機通訊層；本 MVP 已新增 mock、serial、sbRIO UDP skeleton，並新增可對接 `JasonLiaoJCS/BioRoLaROS2` 的 `biorola_ros` backend
- 依你目前硬體描述，真機只有 IMU 與 6 顆 main drive encoder 是真實回授；ABAD observation 使用 commanded estimate；damper 不送 motor command

## 仍未知 / TODO

- 你已提供真機 ROS2 repo `JasonLiaoJCS/BioRoLaROS2`；目前優先使用 `biorola_ros` backend 對接 `/motor/command` 與 `/motor/state`
- 若未來繞過 BioRoLaROS2 直接連 sbRIO，LabVIEW RT/FPGA packet format 仍未最終確認，請使用 `sbrio_udp` skeleton 重新定義 protocol
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

BioRoLaROS2 / biorola_ros adapter，這是你目前真機最應該先用的 backend：

```yaml
redrhex_lowlevel_bridge:
  ros__parameters:
    backend: "biorola_ros"
    rinbo:
      command_topic: "/motor/command"
      state_topic: "/motor/state"
      joint_state_topic: "/joint_states"
      preview_topic: "/redrhex/rinbo_motor_command_preview"
      publish_preview: true
      allow_enable: false
      publish_when_disabled: false
      disabled_servo_control_mode: 0
      publish_shutdown_disable: true
      shutdown_disable_repeats: 5
      shutdown_disable_period_s: 0.02
      require_state: true
      block_if_duplicate_command_publishers: true
      state_timeout_s: 0.25
      main_position_counts_per_rev: 54984.83
      main_pwm_per_rad_s: 120.0
      main_max_pwm: 500.0
      main_encoder_zero_counts_rinbo_order: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      main_encoder_sign_rinbo_order: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
      main_velocity_sign_policy_order: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      main_direction_positive_rinbo_order: [true, true, true, false, false, false]
      main_velocity_filter_alpha: 0.35
      main_velocity_max_dt_s: 0.20
      main_velocity_clip_rad_s: 80.0
      abad_encoder_zero_rinbo_order: [740, 2565, 3283, 1944, 2071, 989]
      abad_encoder_counts_per_rad: 1000.0
      abad_encoder_min: 0
      abad_encoder_max: 65535
      abad_sign_rinbo_order: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      servo_control_mode: 2
```

`biorola_ros` 需要先 build/source 你的 `BioRoLaROS2`，因為 runtime 需要 `rinbo_msgs`。它會把 `/redrhex/motor_commands` 轉成 `/motor/command`，並把 `/motor/state` 轉成只含 6 顆 main encoder 的 `/joint_states`。BioRoLaROS2 的 `MotorCmdStamped` leg 欄位目前實際使用 PWM 式 `voltage/direction`，不是標準速度閉環，所以 `main_pwm_per_rad_s` 一定要架空慢慢調。方向、encoder sign、ABAD servo sign 都從 YAML 調，不要改 policy。

BioRoLaROS2/sbRIO 檢查工具：

```bash
ros2 run redrhex_lowlevel_bridge biorola_bringup_check --message-timeout-s 5.0
ros2 run redrhex_lowlevel_bridge biorola_bringup_check --message-timeout-s 5.0 --require-power-state
```

Power helper：

```bash
ros2 run redrhex_lowlevel_bridge biorola_power_tool digital
ros2 run redrhex_lowlevel_bridge biorola_power_tool sensors
ros2 run redrhex_lowlevel_bridge biorola_power_tool relay --confirm-relay
ros2 run redrhex_lowlevel_bridge biorola_power_tool status
ros2 run redrhex_lowlevel_bridge biorola_power_tool off
```

`biorola_power_tool` 預設會等待 `/power/command` 有 subscriber，避免你把 power command 發到空氣中。若它說沒有 subscriber，先修 `rinbo_ros_bridge`，不要直接跳過。

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
- BioRoLaROS2 ROS2 message protocol 已可對接，但 sbRIO/CORE 內部 watchdog、限流、PWM 標定、servo encoder scale 仍需真機確認
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
