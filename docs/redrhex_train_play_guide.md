# RedRhex 終極操作指南

更新日期：2026-04-16

這份文件是 RedRhex 專案目前唯一的操作手冊，目標不是講理論，而是讓一個第一次碰這個專案的人也能知道：

- 要先做什麼檢查
- 怎麼開始 train
- 怎麼用 play 看模型
- 怎麼匯出 policy
- 怎麼跑五階段 curriculum
- 怎麼用 teacher / distillation
- 怎麼做 smoke test、command sweep、除錯

如果你要看「這次改革背後參考了哪些論文、理論、程式修改細節、reward 設計、驗證結果」，請看：

- `docs/2026_Midterm.md`
- `docs/redrhex_improvement_strategy_full.md`
- `docs/redrhex_forwardfast_professor_report.md`

這份文件只負責把操作流程講清楚。

---

## 0. 完全新手先看這裡

如果你現在最大的感覺是：

- 我不知道 `Train` 跟 `Play` 差在哪裡
- 我不知道 `Agent` 是什麼
- 我不知道 `Teacher` / `Student` 是什麼
- 我不知道 `Policy` / `History` / `Actor` / `Critic` 是什麼
- 我也不知道為什麼有些參數前面要加 `--`，有些卻寫成 `env.stage=1`

那你先看這一節，再往下看操作指令。

## 0.1 最白話的名詞解釋

### `Train` 是什麼

`Train` 就是「訓練模型」。

你執行 `python scripts/rsl_rl/train.py ...` 之後，程式會：

1. 建立很多個模擬環境
2. 讓機器人在模擬裡不斷嘗試動作
3. 根據 reward 判斷哪些動作比較好
4. 更新神經網路參數
5. 每隔一段時間存成 `model_*.pt`

所以 `Train` 的產物是：

- TensorBoard 訓練曲線
- `model_*.pt` checkpoint
- `params/env.yaml`、`params/agent.yaml`

### `Play` 是什麼

`Play` 就是「拿一個已經訓練好的 checkpoint 來播放、測試、觀察」。

你執行 `python scripts/rsl_rl/play.py ...` 之後，程式不會再學習，它只會：

1. 載入某個 `model_*.pt`
2. 建立模擬環境
3. 用這個模型持續輸出動作
4. 讓你看它會怎麼走
5. 順便自動匯出 `policy.pt` 和 `policy.onnx`

所以 `Play` 的用途是：

- 看訓練好的模型是不是會走
- 測試鍵盤控制
- 匯出部署用模型

### `Task` 是什麼

`Task` 可以理解成「你要用哪一種環境設定來訓練或播放」。

目前最常用兩個：

- `Template-Redrhex-Direct-v0`
  - 完整主任務
  - 包含 rough terrain、history、critic privileged obs、teacher obs、故障注入等完整改革
- `Template-Redrhex-ForwardFast-Direct-v0`
  - 快速直走版
  - 比較適合快速測試與現場 sim2real 直走調參

### `Agent` 是什麼

這裡的 `Agent` 不是指「機器人本體」，而是指「這次要用哪一種 RL 訓練設定與網路結構」。

也就是說，`--agent` 會決定：

- 用哪個網路輸入
- 用哪種 runner
- actor / critic 吃哪些 observation
- 是一般 PPO、teacher PPO，還是 distillation
- log 會存到哪個 experiment root

所以你可以把 `--agent` 想成：

- 「這次我要用哪個訓練模式」

### `Checkpoint` 是什麼

`Checkpoint` 就是訓練途中存下來的模型檔。

在這個專案裡，最重要的 checkpoint 檔名長這樣：

- `model_50.pt`
- `model_100.pt`
- `model_2500.pt`

你之後做 `resume`、`play`、`eval`，幾乎都是拿這種 `model_*.pt`。

### `Run` 是什麼

`Run` 就是某一次訓練實驗的資料夾。

例如：

```text
logs/rsl_rl/redrhex_wheg/2026-04-16_12-10-22_wheg_locomotion_reform_v1/
```

這整個資料夾就是一個 run。

裡面會有：

- `model_*.pt`
- TensorBoard event 檔
- `params/`
- `exported/`

### `Resume` 是什麼

`Resume` 就是「從舊 checkpoint 接著訓練，不是從零開始」。

---

## 0.2 `Actor`、`Critic`、`Policy`、`History` 到底是什麼

這些名詞很像，但不是同一件事。

### `Actor`

`Actor` 是真正負責「輸出動作」的網路。

它看到 observation 之後，會決定下一步關節要怎麼動。

如果你把整個機器人 policy 想成一個大腦：

- `Actor` = 負責下命令的部分

### `Critic`

`Critic` 不負責輸出動作，它負責評估：

- 「這個狀態值不值得」
- 「這條軌跡看起來有沒有前途」

在 PPO 訓練裡，`Critic` 幫助 `Actor` 學得更穩定。

非常重要的一點：

- `Critic` 通常只在訓練時使用
- 真正部署到機器人時，通常只需要 `Actor`

所以你可以把它理解成：

- `Actor` = 開車的人
- `Critic` = 副駕兼教練，在旁邊評估你開得好不好

### `Policy`

`Policy` 這個詞在 RL 裡通常是「整個決策規則」。

在這個專案裡你會看到兩種意思：

1. 一般概念上的 `policy`
   - 指整個控制策略，也就是模型學到的行為規則
2. observation group 裡的 `policy`
   - 指給 actor 用的「當前時刻基礎觀測」

所以看到 `policy` 時要分辨上下文。

### `History`

`History` 是「前幾個時間步的觀測歷史」。

為什麼要有它？

因為如果 actor 只看當前瞬間，很可能不知道：

- 腿剛剛是往前擺還是往後擺
- 機器人是正在加速還是剛剛打滑
- 當前動作是 gait 的哪個階段

加入 `history` 之後，actor 雖然不是 RNN，但仍能看到一小段時間脈絡。

你可以把它理解成：

- `policy` = 現在這一瞬間看到的資料
- `history` = 剛剛前幾幀發生了什麼

### `Policy + History` 為什麼常一起出現

因為目前這個專案的主力 deployable actor，就是吃：

- `policy`
- `history`

也就是：

- 當前觀測
- 加上一小段過去歷史

這樣在不依賴特權資訊的情況下，也能學到比較穩定的步態控制。

---

## 0.3 `Teacher`、`Student`、`Distillation` 是什麼

### `Teacher`

`Teacher` 是「訓練時比較強、可以看到更多資訊的老師模型」。

它通常可以看到一些真機部署時不一定能直接拿到，或不想依賴的資訊，例如：

- 更完整的 privileged observation
- 更方便訓練的額外狀態資訊

因為它看到的資訊更多，所以通常比較容易學得好。

但是代價是：

- 它不一定適合直接拿去真機部署

### `Student`

`Student` 是「最後想部署的學生模型」。

它通常只能看部署時真的拿得到的資料，例如：

- `policy`
- `history`

所以它比較符合真機條件。

### `Distillation`

`Distillation` 是「讓 student 去模仿 teacher」。

概念是：

1. 先訓練一個很強的 teacher
2. 再讓 student 用較少的資訊去學 teacher 的行為
3. 這樣 student 雖然看得比 teacher 少，但仍有機會學到 teacher 的好動作

你可以把它理解成：

- `Teacher` = 看答案教學的老師
- `Student` = 考試時只能自己作答的學生
- `Distillation` = 老師先做示範，學生再模仿

---

## 0.4 這個專案裡 observation group 的意思

目前你最常看到四組 observation：

### `policy`

給 deployable actor 使用的基本當前觀測。

### `history`

給 actor 的觀測歷史，讓 actor 不只看單一瞬間。

### `critic`

只給 critic 使用的 privileged observation。

重點是：

- 這些資料幫助 critic 在訓練時更會評估
- 但部署時通常不會給 actor

### `teacher`

給 teacher actor / critic 使用的更完整 privileged observation。

所以這四組在概念上可以這樣記：

- `policy` = 目前這一刻的基本資料
- `history` = 過去一小段資料
- `critic` = 訓練時只給 critic 的額外資訊
- `teacher` = 訓練 teacher 時可看的更完整資訊

---

## 0.5 `Parser`、`CLI 參數`、`Hydra 覆寫` 到底差在哪

這個專案有兩種常見的改參數方式。

### 第一種：`--開頭` 的參數

這些是由 `argparse` parser 處理的，也就是你說的 `Parser`。

例如：

- `--task`
- `--agent`
- `--num_envs`
- `--max_iterations`
- `--resume`
- `--load_run`
- `--checkpoint`
- `--headless`
- `--disable_keyboard_control`

這些參數定義在：

- `scripts/rsl_rl/train.py`
- `scripts/rsl_rl/play.py`

裡面的 `parser.add_argument(...)`。

你可以把它理解成：

- `Parser` 管的是「命令列上標準參數」

### 第二種：`name=value` 形式的覆寫

例如：

- `env.stage=1`
- `env.draw_debug_vis=False`

這種不是 `argparse` parser 處理的，而是留給 Hydra 去覆寫設定物件。

在 `train.py` / `play.py` 裡，你會看到：

```python
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
```

意思就是：

1. parser 先吃掉它認得的 `--xxx`
2. 剩下不認得的東西，例如 `env.stage=1`
3. 再交給 Hydra

所以這個專案的實際規則是：

- `--xxx` = parser 參數
- `env.xxx=...` = Hydra 覆寫

### 最常見的錯誤

很多人會把：

```bash
stage=1
```

誤以為可以直接改 stage。

但正確寫法是：

```bash
env.stage=1
```

因為 stage 是 environment config 裡的一個欄位，不是 parser 直接定義的獨立參數。

---

## 0.6 如果你以後要自己新增模式，要改哪裡

這一段是給你之後自己維護專案時看的。

### 如果你只是想「多一個命令列參數」

例如你想新增：

- `--my_debug_flag`

那你要去改：

- `scripts/rsl_rl/train.py`
- 或 `scripts/rsl_rl/play.py`

裡面的：

```python
parser.add_argument(...)
```

### 如果你想「多一個環境模式」

例如你想加一個新 task，像：

- `Template-Redrhex-Something-Direct-v0`

那通常要改：

1. `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`
   - 新增或修改 env cfg 類別
2. `source/RedRhex/RedRhex/tasks/direct/redrhex/__init__.py`
   - 把新 task 註冊進 gym

### 如果你想「多一個 agent 模式」

例如你想新增：

- 新的 PPO cfg
- 新的 teacher cfg
- 新的 distillation cfg

那通常要改：

1. `source/RedRhex/RedRhex/tasks/direct/redrhex/agents/`
   - 新增或修改 agent config class
2. `source/RedRhex/RedRhex/tasks/direct/redrhex/__init__.py`
   - 把新的 `*_cfg_entry_point` 名稱註冊進 task

然後之後你才能在終端機這樣用：

```bash
--agent 你新註冊的名字
```

### 如果你只是想臨時改某個 env 參數

例如只想把 stage 改成 4，或關掉 debug vis，

那不用改 parser，只要在命令最後加 Hydra override：

```bash
env.stage=4 env.draw_debug_vis=False
```

---

## 0.7 完全照抄版：從零開始跑一次 Train + Play

下面是一套最保守、最適合第一次操作的流程。

### Terminal 1：進入專案

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

### Terminal 1：先做 smoke test

```bash
python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 32
```

如果這一步出錯，先不要急著 train。

### Terminal 1：開始訓練完整主 task

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name my_first_train
```

### Terminal 2：開 TensorBoard

另外開一個新終端機：

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
tensorboard --logdir logs/rsl_rl --port 6006 --bind_all
```

瀏覽器打開：

- 本機：`http://localhost:6006`
- 遠端：`http://<你的主機IP>:6006`

### Terminal 3：找最新 checkpoint

再開一個新終端機：

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab

RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/* | head -1)")
CKPT=$(ls -v logs/rsl_rl/redrhex_wheg/$RUN/model_*.pt | tail -1)
echo "RUN=$RUN"
echo "CKPT=$CKPT"
```

### Terminal 3：播放模型

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --load_run "$RUN" \
  --checkpoint "$CKPT"
```

這一段只適用於：

- 一般 PPO run
- 也就是 `logs/rsl_rl/redrhex_wheg/...` 下面的 checkpoint
- 對應 agent 是 `rsl_rl_cfg_entry_point`

**重要：`play.py` 的 `--agent` 必須和 checkpoint 類型對上。**

- 如果 checkpoint 來自一般 PPO：
  - 用 `--agent rsl_rl_cfg_entry_point`
- 如果 checkpoint 來自 teacher PPO：
  - 用 `--agent rsl_rl_teacher_cfg_entry_point`
- 如果 checkpoint 來自 distillation：
  - 用 `--agent rsl_rl_distillation_cfg_entry_point`

如果你拿：

- `redrhex_wheg_distill/.../model_*.pt`

卻配：

- `--agent rsl_rl_cfg_entry_point`

那就會在載入時看到：

- `Missing key(s): actor.* / critic.*`
- `Unexpected key(s): student.* / teacher.*`

因為一般 PPO runner 期待的是 `actor/critic` 權重，但 distillation checkpoint 裡存的是 `student/teacher` 權重。

### Terminal 3：只匯出 policy，不播放

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 1 \
  --disable_keyboard_control \
  --load_run "$RUN" \
  --checkpoint "$CKPT" \
  --export_policy
```

匯出後的檔案位置通常會在：

```text
logs/rsl_rl/redrhex_wheg/<你的 run>/exported/
```

---

## 0.8 最常見的「我要改模式」到底怎麼改

很多時候你不需要改程式，只需要改命令裡的一個參數。

### 例 1：我要從完整主任務改成快速直走

把：

```bash
--task Template-Redrhex-Direct-v0
```

改成：

```bash
--task Template-Redrhex-ForwardFast-Direct-v0
```

### 例 2：我要從一般 PPO 改成 teacher PPO

把：

```bash
--agent rsl_rl_cfg_entry_point
```

改成：

```bash
--agent rsl_rl_teacher_cfg_entry_point
```

### 例 3：我要從一般 PPO 改成 distillation

把：

```bash
--agent rsl_rl_cfg_entry_point
```

改成：

```bash
--agent rsl_rl_distillation_cfg_entry_point
```

### 例 4：我要讓主 task 固定在 stage 3

在命令最後面加：

```bash
env.stage=3
```

例如：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2000 \
  --run_name stage3_debug \
  env.stage=3
```

### 例 5：我要關掉 GUI

加上：

```bash
--headless
```

### 例 6：我要在 play 時不要鍵盤控制

加上：

```bash
--disable_keyboard_control
```

### 例 7：我要只匯出模型，不播放

加上：

```bash
--export_policy
```

### 例 8：我要從舊 checkpoint 接著訓練

加上：

```bash
--resume --load_run <run 名稱> --checkpoint model_xxxxx.pt
```

### 例 9：我要改平行環境數量

改：

```bash
--num_envs 2048
```

例如改成：

```bash
--num_envs 4096
```

### 例 10：我要縮短或拉長訓練時間

改：

```bash
--max_iterations 2500
```

例如改成：

```bash
--max_iterations 800
```

或：

```bash
--max_iterations 5000
```

你可以把這一節記成一個最簡單規則：

- 要換大方向，通常改 `--task` 或 `--agent`
- 要改環境細節，通常加 `env.xxx=...`
- 要改訓練長度或規模，通常改 `--num_envs`、`--max_iterations`

---

## 0.9 如果你現在就要開始 Train，請直接照這條流程做

這一節就是回答你現在最在意的問題：

- 我現在到底要跑 `train.py` 還是五階段 pipeline？
- `Task` 要選哪一個？
- `Agent` 要選哪一個？
- 要不要一開始就 train teacher / student？
- 最後哪條路最適合正式訓練？

先講最短結論：

1. 如果你現在只是想把訓練先跑起來，直接跑 `train.py` 就可以。
2. 但是如果你要的是「完整 locomotion 能力最正式、最穩的主線訓練」，五階段 curriculum 仍然存在，而且仍然推薦使用。
3. `Teacher` / `Student` 是進階路線，不是新手第一步。

### 0.9.1 最重要的觀念

目前主 task 的預設設定是：

- `Template-Redrhex-Direct-v0`
- `stage = 5`
- `curriculum_auto_progress = False`

這代表：

- 你如果直接跑 `train.py --task Template-Redrhex-Direct-v0`
- 它不會自動幫你跑 Stage1 -> Stage2 -> Stage3 -> Stage4 -> Stage5
- 它現在跑的是「單段的 Stage5 mixed-skills 訓練」

所以一定要記住：

- `train.py` 直接跑主 task = 單段 mixed training
- `train_stage_pipeline.sh` = 真正的五階段 curriculum training

### 0.9.2 你現在該選哪一條路

先看這張一頁式決策表：

| 你的目標 | `Task` | `Agent` | 建議腳本 | 說明 |
|---|---|---|---|---|
| 第一次上手，把整個訓練跑通 | `Template-Redrhex-Direct-v0` | `rsl_rl_cfg_entry_point` | `train.py` | 先把 baseline 跑起來 |
| 最快看直走結果 | `Template-Redrhex-ForwardFast-Direct-v0` | `rsl_rl_cfg_entry_point` | `train.py` | 快速驗證路線 |
| 只想穩定往前走，不想訓練其他技能 | `Template-Redrhex-Direct-v0` + `env.stage=1` | `rsl_rl_cfg_entry_point` | `train.py` | 非 ForwardFast 的正式 forward-only 路線 |
| 做最正式的完整 locomotion 主線 | `Template-Redrhex-Direct-v0` | `rsl_rl_cfg_entry_point` | `train_stage_pipeline.sh` | 真正五階段 curriculum |
| 做 forward-only teacher | `Template-Redrhex-Direct-v0` + `env.stage=1` | `rsl_rl_teacher_cfg_entry_point` | `train.py` | 只訓 forward 的 teacher |
| 做 forward-only student distillation | `Template-Redrhex-Direct-v0` + `env.stage=1` | `rsl_rl_distillation_cfg_entry_point` | `train.py` | 要先有 forward-only teacher checkpoint |
| 做 teacher 上界 | `Template-Redrhex-Direct-v0` | `rsl_rl_teacher_cfg_entry_point` | `train.py` | 進階研究路線 |
| 做 student distillation | `Template-Redrhex-Direct-v0` | `rsl_rl_distillation_cfg_entry_point` | `train.py` | 要先有 teacher checkpoint |

#### 路線 A：我只是第一次要把訓練跑起來

這是最推薦的新手起點。

目標：

- 先確認環境、訓練腳本、checkpoint、play、export 都正常

做法：

1. 啟動環境
2. 跑 smoke test
3. 直接跑一般 PPO
4. 用 play 看模型

這條路建議：

- `Task`：`Template-Redrhex-Direct-v0`
- `Agent`：`rsl_rl_cfg_entry_point`

也就是：

- 不先做 teacher
- 不先做 distillation
- 先把最基本可部署主線跑通

#### 路線 B：我想最快看到直走結果

這條路最適合：

- 現場快速迭代
- reward / 物理參數剛改完
- 想快速看 robot 有沒有往前動

這條路建議：

- `Task`：`Template-Redrhex-ForwardFast-Direct-v0`
- `Agent`：`rsl_rl_cfg_entry_point`

也就是：

- 用 ForwardFast
- 但仍然先用一般 PPO
- 不需要一開始就 teacher / student

這條路是：

- 快速驗證路線
- 不是最正式的完整多技能主線

#### 路線 B-2：我只想穩定往前走，但不要 ForwardFast

這條路就是你現在問的模式，而且它其實已經存在。

這條路的定位是：

- 不追求 ForwardFast 那種快速收斂
- 不訓練 lateral / diagonal / yaw
- 只做穩定的 forward-only 訓練
- 可以搭配一般 PPO、teacher、student distillation
- 更接近「正式 forward-only 主線」，而不是快速驗證特化分支

這條路建議：

- `Task`：`Template-Redrhex-Direct-v0`
- `env.stage=1`
- `Agent`：
  - 一般 PPO：`rsl_rl_cfg_entry_point`
  - teacher：`rsl_rl_teacher_cfg_entry_point`
  - student distillation：`rsl_rl_distillation_cfg_entry_point`

這條路和 ForwardFast 的差別是：

- **ForwardFast**
  - 專門為了快
  - 迭代數較短
  - DR 較窄
  - power / torque 相關 reward 權重較低
  - 比較像 fast lane

- **主 task + `env.stage=1`**
  - 仍然使用完整主 task 架構
  - 只是把技能固定在 Stage1 forward-only
  - reward 裡的 `power_efficiency`、`torque_penalty` 權重也已經存在
  - 比較適合你現在說的「穩定往前直走就好，不求快」

如果你要的是：

- 穩定前進
- 不需要 lateral / diagonal / yaw
- 可以用 teacher / student
- 不想走 ForwardFast 快速收斂路線

那我最推薦的就是這條。

#### 路線 C：我要完整 locomotion 能力，而且希望最正式

這條路就是目前最推薦的正式主線。

這條路建議：

- `Task`：`Template-Redrhex-Direct-v0`
- `Agent`：`rsl_rl_cfg_entry_point`
- 訓練方式：`train_stage_pipeline.sh`

也就是：

- 仍然用一般 PPO 作為主線
- 但不是直接跑單段 `train.py`
- 而是用五階段 curriculum：
  - Stage1 forward
  - Stage2 lateral
  - Stage3 diagonal
  - Stage4 yaw
  - Stage5 mixed integration

如果你現在問我：

- 「我最後想要完整能力最好，最應該跑哪條？」

那我會回答：

- 優先跑這條五階段主線

#### 路線 D：我想做進階研究，追更高上限

這才是 teacher / student 的位置。

這條路建議順序是：

1. 先有穩定 baseline
   - 最好先完成一般 PPO 主線
   - 最好先有一個你滿意的五階段或單段 mixed baseline
2. 再 train teacher
3. 最後再 distill student

這條路的定位是：

- 研究型、進階型
- 不是第一次上手就先做的事

### 0.9.3 我現在最推薦你的實際順序

如果你現在整個流程很混亂，我最推薦你照這個順序：

#### Step 0：啟動虛擬環境

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

#### Step 1：先確認整個 stack 沒壞

```bash
python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 32
```

#### Step 2：如果你只是第一次要成功跑起來

直接跑：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name my_first_train
```

這一步的意義是：

- 先把主 task 的單段 mixed 訓練跑通
- 先確認 checkpoint / play / export 全部正常

#### Step 3：如果你想先快速看直走

改跑：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-ForwardFast-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 1500 \
  --run_name forward_fast_trial_a
```

#### Step 3-B：如果你想只做穩定 forward-only，而且不要 ForwardFast

先做一般 PPO 的 forward-only 版本：

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

這條路的特點是：

- 還是主 task
- 但固定在 Stage1
- 所以不會去訓 lateral / diagonal / yaw
- 又不像 ForwardFast 那樣特別追求快速收斂

如果你要 forward-only teacher：

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

如果你要 forward-only student distillation，先找 teacher：

```bash
TEACHER_RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg_teacher/* | head -1)")
TEACHER_CKPT=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg_teacher/$TEACHER_RUN/model_*.pt | tail -1)")
echo "TEACHER_RUN=$TEACHER_RUN"
echo "TEACHER_CKPT=$TEACHER_CKPT"
```

因為 distillation 會從 `redrhex_wheg_distill` experiment root 找 resume 路徑，所以先建立連結：

```bash
mkdir -p logs/rsl_rl/redrhex_wheg_distill
ln -s ../redrhex_wheg_teacher/$TEACHER_RUN logs/rsl_rl/redrhex_wheg_distill/$TEACHER_RUN
```

再啟動 forward-only student distillation：

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

這一條路最推薦的實際順序是：

1. 先做 forward-only 一般 PPO baseline
2. 如果你想要更高上限，再做 forward-only teacher
3. 最後再做 forward-only student distillation

#### Step 3-B-1：完整照抄版

如果你現在的目標很明確，就是：

- 只做正式 forward-only
- 保留主 task 的 reward / observation / teacher-student 架構
- 不走 ForwardFast

那你可以直接照下面這一整段做。

**A. 先進專案並啟動 Isaac Lab 環境**

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

**B. train forward-only baseline student**

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

**C. train forward-only privileged teacher**

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

**D. 找 teacher checkpoint**

```bash
TEACHER_RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg_teacher/* | head -1)")
TEACHER_CKPT=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg_teacher/$TEACHER_RUN/model_*.pt | tail -1)")
echo "TEACHER_RUN=$TEACHER_RUN"
echo "TEACHER_CKPT=$TEACHER_CKPT"
```

**E. 讓 distillation 可以讀到 teacher run**

```bash
mkdir -p logs/rsl_rl/redrhex_wheg_distill
ln -s ../redrhex_wheg_teacher/$TEACHER_RUN logs/rsl_rl/redrhex_wheg_distill/$TEACHER_RUN
```

如果你之前已經建過同名連結，`ln -s` 可能會顯示檔案已存在。  
這時先確認它指向的是正確的 teacher run，再決定要不要手動刪掉舊連結重建。

**F. 啟動 forward-only student distillation**

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

**G. 為什麼這裡只用 teacher checkpoint，沒有用舊 student checkpoint？**

這是目前這套 distillation 流程的**刻意設計**，不是漏掉。

原因是底層 `DistillationRunner` / `StudentTeacher` 的載入規則本來就是：

- 如果載入的是一般 PPO checkpoint（裡面主要是 `actor.*` / `actor_obs_normalizer.*`）
  - 這些權重會被轉成 **teacher network**
  - student 會重新隨機初始化
  - 這代表「開始一個新的 distillation 訓練」
- 如果載入的是**舊的 distillation checkpoint**（裡面有 `student.*` / `teacher.*`）
  - 才會同時載入 student 與 teacher
  - 這代表「續跑既有 distillation 訓練」

所以：

- 用 `forward_stage1_teacher` 的 checkpoint 來開 `forward_stage1_distill`
  - 是合理的
  - 代表「拿 teacher 當 supervision，重新訓一個 student」
- 你之前訓好的一般 PPO student checkpoint
  - **不會**在這條流程裡自動當成 student warm-start
  - 如果你直接把一般 PPO student checkpoint 丟給 distillation loader，它會把那個 `actor` 當成 teacher 來讀，不是你想要的「student 初始化」

實務上要這樣分：

1. 你要**開始新的 distillation**
- 用 teacher PPO checkpoint

2. 你要**續跑舊的 distillation**
- 用 distillation run 自己產生的 `model_*.pt`

3. 你要「teacher 來自 teacher PPO，但 student 也想從舊 PPO baseline warm-start」
- 這是**另外一種雙來源初始化需求**
- 目前這個專案還沒有內建這個功能
- 如果要做，必須另外改 loader，把 teacher 權重和 student 初始權重分開指定

這整套流程的順序不要顛倒：

1. 先 train baseline
2. 再 train teacher
3. 最後才做 distillation

因為 distillation 需要先有 teacher checkpoint 可以模仿。

#### Step 4：如果你要正式完整訓練

不要只停在單段 `train.py`，而是改跑：

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --num_envs 4096 \
  --s1 8000 \
  --s2 8000 \
  --s3 9000 \
  --s4 10000 \
  --s5 12000
```

這一步才是：

- 真正的五階段 curriculum
- 目前最推薦的完整 locomotion 主線

#### Step 5：如果主線已經穩了，才考慮 teacher / student

先 train teacher：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name wheg_privileged_teacher_v1
```

再做 distillation：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_distillation_cfg_entry_point \
  --resume \
  --load_run <teacher_run> \
  --checkpoint <teacher_model>.pt \
  --headless \
  --num_envs 2048 \
  --max_iterations 800 \
  --run_name wheg_student_distill_v1
```

### 0.9.4 如果你只要我給你一句最簡單的建議

如果你現在只問：

- 「我今天到底應該先訓什麼？」

我的建議是：

1. 第一次上手：先跑 `Template-Redrhex-Direct-v0 + rsl_rl_cfg_entry_point`
2. 想快速看直走：跑 `Template-Redrhex-ForwardFast-Direct-v0 + rsl_rl_cfg_entry_point`
3. 想只做穩定 forward-only：跑 `Template-Redrhex-Direct-v0 + env.stage=1`
4. 想做正式完整主線：跑 `train_stage_pipeline.sh`
5. 想追更高上限：等主線穩了再做 teacher -> student

也就是說，對大多數情況：

- `rsl_rl_cfg_entry_point` 才是你的第一選擇
- `train_stage_pipeline.sh` 才是完整正式路線
- `teacher/student` 是進階路線，不是第一步
- 如果你只想穩定往前走，不想混其他技能，也不用 ForwardFast，就用 `Template-Redrhex-Direct-v0 + env.stage=1`

## 1. 你先要知道的事

目前專案有兩個主要 task：

| 用途 | Task ID | 說明 |
|---|---|---|
| 完整能力訓練 | `Template-Redrhex-Direct-v0` | 主任務，包含 rough terrain、history actor、asymmetric critic、teacher obs、symmetry augmentation、actuator randomization、fault injection 等改革 |
| 快速直走收斂 | `Template-Redrhex-ForwardFast-Direct-v0` | forward-only 快速版，適合現場快速迭代與 sim2real 直走調參 |

目前 `train.py` / `play.py` 都支援三種 agent 入口：

| `--agent` 值 | 用途 | 可否直接當部署學生策略 |
|---|---|---|
| `rsl_rl_cfg_entry_point` | 一般 PPO，actor 吃 `policy + history`，critic 吃 privileged obs | 可以，這是最常用的部署路線 |
| `rsl_rl_teacher_cfg_entry_point` | privileged teacher PPO，actor/critic 都吃 `teacher` obs | 不建議直接當真機部署策略，主要用來做 teacher 上界或蒸餾 |
| `rsl_rl_distillation_cfg_entry_point` | teacher-student distillation | 可以，這是把 teacher 壓縮成可部署 student 的路線 |

最重要的實務觀念：

- `rsl_rl_cfg_entry_point` 已經是改革後的主力設定，不是舊版 baseline。
- 現在的主 task 內建 history、critic privileged obs、teacher obs、對稱增強、rough terrain、故障注入與 actuator randomization，不需要另外開很多旗標才會生效。
- `play.py` 現在每次載入 checkpoint 都會自動匯出 `policy.pt` 和 `policy.onnx` 到該 run 的 `exported/` 目錄。
- `play.py` 和 `eval_command_sweep.py` 會自動幫你把錯誤的 checkpoint 參數修正成真正的 `model_*.pt`。就算你誤傳資料夾、`events.out.tfevents...` 或 `exported/policy.pt`，它也會盡量 fallback 到同 run 裡最新的訓練 checkpoint。

---

## 2. 先做環境準備

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

建議第一次操作前先做一次最小檢查：

```bash
bash -n scripts/rsl_rl/train_stage_pipeline.sh

python -m py_compile \
  scripts/rsl_rl/train.py \
  scripts/rsl_rl/play.py \
  scripts/rsl_rl/eval_command_sweep.py \
  scripts/rsl_rl/validate_reform_stack.py \
  scripts/rsl_rl/validate_distillation_stack.py
```

如果你只是想先確認新改革的訓練堆疊能不能跑，再加做這個 smoke test：

```bash
python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 32
```

這個腳本會檢查：

- 環境能否成功 reset / rollout
- `policy/history/critic/teacher` observation 維度是否一致
- reward 和 observation 是否有 NaN / Inf
- rough terrain / fault injection / actuator scaling 是否真的在跑

---

## 3. 最短成功路徑

如果你完全是第一次使用這個專案，推薦照這個順序：

1. 跑 `validate_reform_stack.py`，確認環境與 observation stack 正常。
2. 選一條訓練路線：
   - 想先快速拿到直走模型：用 `Template-Redrhex-ForwardFast-Direct-v0`
   - 想直接做完整能力模型：用 `Template-Redrhex-Direct-v0`
3. 開 TensorBoard 看訓練。
4. 用 `play.py` 載入最新 checkpoint。
5. 從 `exported/policy.onnx` 或 `exported/policy.pt` 拿部署檔。

---

## 4. 日誌、checkpoint、輸出檔會放在哪裡

所有 RSL-RL 訓練都會寫到：

```text
logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
```

常見 experiment root：

| 模式 | experiment root |
|---|---|
| 主 task PPO | `logs/rsl_rl/redrhex_wheg/` |
| ForwardFast PPO | `logs/rsl_rl/redrhex_forward_fast/` |
| 主 task teacher PPO | `logs/rsl_rl/redrhex_wheg_teacher/` |
| ForwardFast teacher PPO | `logs/rsl_rl/redrhex_forward_fast_teacher/` |
| 主 task distillation | `logs/rsl_rl/redrhex_wheg_distill/` |
| ForwardFast distillation | `logs/rsl_rl/redrhex_forward_fast_distill/` |

每個 run 內通常有：

- `model_*.pt`：真正可拿來 resume / play / eval 的訓練 checkpoint
- `events.out.tfevents...`：TensorBoard 日誌，不是模型
- `params/env.yaml`、`params/agent.yaml`：該次訓練實際用到的設定
- `exported/policy.pt`：JIT 匯出
- `exported/policy.onnx`：ONNX 匯出

實務上要記住：

- 能拿來 `--checkpoint` 的主角永遠是 `model_*.pt`
- `events.out.tfevents...` 不能拿去 `play.py`
- `exported/policy.pt` 不是訓練 checkpoint，不能拿來 resume 訓練

---

## 5. 最常用的訓練指令

## 5.1 主 task 一般 PPO

這是目前最推薦的主線訓練入口。它用 deployable actor 設計：

- actor：`policy + history`
- critic：`policy + history + critic privileged obs`

也就是說，訓練時 critic 會看到比較多資訊，但最後部署時 actor 不需要 teacher-only 的特權資訊。

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name wheg_locomotion_reform_v1
```

如果你要把 env 數量拉高：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 4096 \
  --max_iterations 2500 \
  --run_name wheg_locomotion_reform_v1_big
```

## 5.2 ForwardFast 快速直走 PPO

這條路線適合：

- 現場想快速得到可上機的直走 policy
- reward / 物理參數剛調完，想快點看到行為變化
- 不想一開始就訓完整 mixed locomotion

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-ForwardFast-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 1500 \
  --run_name forward_fast_reform_v1
```

如果 GPU 夠，可以加到 4096 env：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-ForwardFast-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 4096 \
  --max_iterations 1500 \
  --run_name forward_fast_reform_v1_big
```

## 5.3 主 task 五階段 curriculum

如果你要完整能力整合，推薦直接使用五階段 pipeline，而不是手動一段一段拚。

這裡有兩個原本指南很重要、現在也仍然成立的觀念：

- 五階段不是五個互不相干的模型，而是同一個 policy 連續微調
- 目的不是把所有技能一次混著硬學，而是降低 forward / lateral / diagonal / yaw 彼此干擾

你可以把它理解成：

1. Stage1 先把直走打穩
2. Stage2 再把側移獨立拉起來
3. Stage3 再學 `vx + vy` 的耦合控制
4. Stage4 單獨練 yaw，避免旋轉技能干擾其他技能
5. Stage5 最後再整合前四段

五個 stage 的意義：

| Stage | 重點 |
|---|---|
| `env.stage=1` | forward-only，先把直走穩定性打底 |
| `env.stage=2` | lateral-only，讓側移獨立成型 |
| `env.stage=3` | diagonal-only，讓 `vx + vy` 組合動作成型 |
| `env.stage=4` | yaw-only，先把原地旋轉單獨練穩 |
| `env.stage=5` | mixed，整合前四段 |

補充一個很重要的訓練邏輯：

- Stage5 不是暴力覆蓋前四段
- 目前環境裡有前進防遺忘機制，會盡量保護 Stage1 已建立的穩定直走能力，再慢慢把 lateral / diagonal / yaw 拉上來

### 一鍵跑完整 pipeline

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --num_envs 4096 \
  --s1 8000 \
  --s2 8000 \
  --s3 9000 \
  --s4 10000 \
  --s5 12000
```

### 先做 GUI 預檢再放整晚

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --precheck_gui 1 \
  --precheck_stage 1 \
  --precheck_envs 64 \
  --precheck_iters 120 \
  --num_envs 4096 \
  --s1 8000 \
  --s2 8000 \
  --s3 9000 \
  --s4 10000 \
  --s5 12000
```

### pipeline 會幫你做什麼

- 每個 stage 自動接前一段 checkpoint
- 預設使用 full resume，也就是 policy、optimizer、iteration continuity 都會接續
- 自動做 stage health gate
- 寫 pipeline log 到 `logs/rsl_rl/pipeline/<run_tag>.log`
- 最後會印出 `FINAL_CKPT=...`

### 中途斷掉後從某 stage 接著跑

注意：如果你要從 stage 3 或 stage 4 接著跑，`--run_tag` 必須和前一次一致，因為 pipeline 會用它去找前一段 run。

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_a \
  --start_stage 3 \
  --num_envs 4096 \
  --s1 8000 \
  --s2 8000 \
  --s3 9000 \
  --s4 10000 \
  --s5 12000
```

### 若你真的要用 policy-only handoff

這通常只適合做實驗，不建議當正式 curriculum。

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag overnight_policy_only \
  --resume_policy_only 1 \
  --reset_action_std 0.8
```

它的效果是：

- 只載入 policy 權重
- optimizer state 不接續
- iteration 會重置，所以 `model_*.pt` 編號會重新開始

## 5.4 手動指定 stage 的一般訓練

如果你不要 pipeline，也可以直接在 `train.py` 後面加 Hydra override：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 8000 \
  --run_name stage1_manual \
  env.stage=1 \
  env.draw_debug_vis=False
```

如果你要手動接上一段：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 8000 \
  --run_name stage2_manual \
  --resume \
  --load_run <上一段 run 名稱> \
  --checkpoint model_xxxxx.pt \
  env.stage=2
```

如果你是在做手動逐段 debug，原本文件裡常用的兩個額外 override 仍然有用：

```bash
env.draw_debug_vis=False \
env.dr_try_physical_material_randomization=False
```

用途是：

- `env.draw_debug_vis=False`：減少不必要視覺除錯負擔
- `env.dr_try_physical_material_randomization=False`：在某些手動除錯情境下，先把物理材質隨機化關掉，讓問題更容易重現與定位

---

## 6. Teacher 與 Distillation 操作

## 6.1 Teacher PPO

teacher 版本是為了訓練上界與之後做蒸餾。它的 actor/critic 都吃 `teacher` privileged obs。

重要提醒：

- teacher 很適合在模擬裡追求更高學習效率或當蒸餾來源
- teacher 不等於真機可直接部署策略
- 真機或 sim2real 部署仍應優先使用一般 PPO student 或 distillation student

### 主 task teacher

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name wheg_privileged_teacher_v1
```

### ForwardFast teacher

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-ForwardFast-Direct-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 1500 \
  --run_name forward_fast_privileged_teacher_v1
```

## 6.2 Distillation 先做 smoke test

現在專案已經把 distillation runner 接好了，但蒸餾流程比一般 PPO 更容易因 checkpoint 路徑規則而卡住，所以推薦先做 smoke test。

### 一次檢查 reform stack + teacher + distillation

```bash
python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 32 \
  --runner_smoke \
  --distill_smoke \
  --runner_steps 8 \
  --distill_steps 8 \
  --log_dir /tmp/redrhex_reform_smoke
```

這會做三件事：

- random rollout smoke
- 1 iteration PPO smoke
- teacher PPO + distillation smoke

輸出檔常見位置：

- `/tmp/redrhex_validate_stats.json`
- `/tmp/redrhex_reform_smoke/teacher/teacher_smoke.pt`
- `/tmp/redrhex_reform_smoke/distill/distill_smoke.pt`

### 已有 teacher checkpoint，單獨檢查 distillation

```bash
python scripts/rsl_rl/validate_distillation_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 8 \
  --teacher_ckpt /absolute/path/to/model_xxxxx.pt \
  --log_dir /tmp/redrhex_reform_distill
```

## 6.3 正式做 distillation

### Step 1：先訓練 teacher

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500 \
  --run_name wheg_privileged_teacher_v1
```

### Step 2：確認 teacher checkpoint

```bash
TEACHER_RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg_teacher/* | head -1)")
TEACHER_CKPT=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg_teacher/$TEACHER_RUN/model_*.pt | tail -1)")
echo "TEACHER_RUN=$TEACHER_RUN"
echo "TEACHER_CKPT=$TEACHER_CKPT"
```

### Step 3：讓 distillation runner 找得到 teacher run

這一步很重要。現在 `train.py` 在 distillation 模式下，會用 distill config 的 `experiment_name` 當 resume 根目錄：

- 主 task distill：`logs/rsl_rl/redrhex_wheg_distill/`
- ForwardFast distill：`logs/rsl_rl/redrhex_forward_fast_distill/`

也就是說，你不能假設 teacher run 放在 `redrhex_wheg_teacher/` 時，`--agent rsl_rl_distillation_cfg_entry_point --resume --load_run <teacher_run>` 一定會直接找到。

最簡單的做法是建立軟連結，讓 teacher run 同時出現在 distill experiment root 下面：

```bash
mkdir -p logs/rsl_rl/redrhex_wheg_distill
ln -s ../redrhex_wheg_teacher/$TEACHER_RUN logs/rsl_rl/redrhex_wheg_distill/$TEACHER_RUN
```

如果是 ForwardFast distill，對應改成：

```bash
mkdir -p logs/rsl_rl/redrhex_forward_fast_distill
ln -s ../redrhex_forward_fast_teacher/$TEACHER_RUN logs/rsl_rl/redrhex_forward_fast_distill/$TEACHER_RUN
```

### Step 4：開始 distillation

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
  --run_name wheg_student_distill_v1
```

ForwardFast 版本只要把 task 換成 `Template-Redrhex-ForwardFast-Direct-v0` 即可。

### Step 5：播放 distillation student

```bash
STUDENT_RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg_distill/* | head -1)")
STUDENT_CKPT=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg_distill/$STUDENT_RUN/model_*.pt | tail -1)")

python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_distillation_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --load_run "$STUDENT_RUN" \
  --checkpoint "$STUDENT_CKPT"
```

`play.py` 已經有處理 distillation runner，會自動用 `student_obs_normalizer` 匯出 student policy。

---

## 7. TensorBoard 怎麼看

最簡單的開法：

```bash
tensorboard --logdir logs/rsl_rl --port 6006 --bind_all
```

常先看這幾類：

- `Train/mean_reward`
- `Train/mean_episode_length`
- `Episode_Termination/terminated`
- `Episode_Reward/rew_fall`
- 各種 `diag_*` 診斷指標

對主 task 而言，現在常見的診斷重點是：

- episode length 有沒有穩定上升
- terminated 有沒有下降
- base height / tilt 相關診斷有沒有長期惡化
- forward / lateral / diagonal / yaw 是否都真的有學起來，而不是只剩一種技能

如果你跑的是五階段 curriculum，原本指南裡每個 stage 的觀察重點也值得保留：

### 五階段各 stage 建議觀察

全階段共通：

- `Train/mean_episode_length`：要上升且穩定
- `Train/mean_reward`：整體往上
- `Episode_Termination/terminated`：下降
- `Episode_Reward/rew_fall`：接近 0
- `Episode_Reward/diag_base_height`：不要長期崩掉

Stage1：

- `diag_cmd_vx` 應大於 0，`diag_cmd_vy/wz` 應接近 0
- `rew_tracking`、`rew_forward_gait` 應上升
- `diag_forward_duty_ema` 建議往穩定 tripod 節奏收斂

Stage2：

- `diag_lateral_fsm_state` 最好能進到 `2`，也就是 `LATERAL_STEP`
- `rew_lateral_speed_deficit` 往 0 靠近
- `diag_lateral_vel` 絕對值上升且方向正確

Stage3：

- `rew_diag_sign` 轉正且穩定
- `diag_vel_error` 下降

Stage4：

- `rew_yaw_track` 上升
- `diag_wz_error` 下降
- `diag_roll_rms`、`diag_pitch_rms` 不要爆掉

Stage5：

- forward / lateral / diagonal / yaw 四類指標都要有反應
- 不能只剩其中一種技能表現好

---

## 8. Play、匯出、鍵盤控制

## 8.1 播放最新 checkpoint

### 主 task PPO

```bash
RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/* | head -1)")
CKPT=$(ls -v logs/rsl_rl/redrhex_wheg/$RUN/model_*.pt | tail -1)

python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --load_run "$RUN" \
  --checkpoint "$CKPT"
```

### ForwardFast PPO

```bash
RUN=$(basename "$(ls -td logs/rsl_rl/redrhex_forward_fast/* | head -1)")
CKPT=$(ls -v logs/rsl_rl/redrhex_forward_fast/$RUN/model_*.pt | tail -1)

python scripts/rsl_rl/play.py \
  --task Template-Redrhex-ForwardFast-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --load_run "$RUN" \
  --checkpoint "$CKPT"
```

你也可以直接傳絕對路徑：

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --checkpoint /absolute/path/to/model_xxxxx.pt
```

如果你是從五階段 pipeline 完成後要抓最終模型，原本手冊這段仍然很好用：

```bash
PIPE_LOG=$(ls -t logs/rsl_rl/pipeline/*.log | head -1)
FINAL_CKPT=$(grep -F "[DONE] FINAL_CKPT=" "$PIPE_LOG" | tail -1 | sed 's/.*FINAL_CKPT=//')
echo "PIPE_LOG=$PIPE_LOG"
echo "FINAL_CKPT=$FINAL_CKPT"
basename "$FINAL_CKPT"
```

如果 `basename "$FINAL_CKPT"` 不是 `model_xxxxx.pt`，就不要直接拿去 play。

## 8.2 Play 的實用行為

- 若 `--checkpoint` 是資料夾，會自動抓其中最新 `model_*.pt`
- 若 `--checkpoint` 指到 `events.out.tfevents...` 或 `exported/policy.pt`，會自動嘗試 fallback 到相鄰的訓練 checkpoint
- 若 run 名稱像 `..._stage4`，`play.py` 會自動把 `env.stage` 推成 4
- 若你不想讓它自動推 stage，可以加 `--disable_auto_stage_from_checkpoint`
- 每次 `play.py` 成功載入模型後，都會自動匯出：
  - `exported/policy.pt`
  - `exported/policy.onnx`

## 8.3 只匯出，不進入 play 迴圈

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 1 \
  --disable_keyboard_control \
  --checkpoint /absolute/path/to/model_xxxxx.pt \
  --export_policy
```

## 8.4 關閉鍵盤控制

如果你想看模型在環境命令採樣下自然表現，不要手動切指令：

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 64 \
  --disable_keyboard_control \
  --checkpoint /absolute/path/to/model_xxxxx.pt
```

## 8.5 開啟鍵盤控制時可用的初始命令

`--initial_command` 可選：

- `forward`
- `backward`
- `left`
- `right`
- `diag_left`
- `diag_right`
- `yaw_ccw`
- `yaw_cw`
- `stop`

初次使用建議：

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --checkpoint /absolute/path/to/model_xxxxx.pt
```

## 8.6 鍵盤對應

鍵盤控制是切換模式，不是按住才有作用。按一次就會維持該命令，直到你按下一個按鍵。

| 按鍵 | 命令 |
|---|---|
| `W` | 前進 |
| `S` | 後退 |
| `A` | 左移 |
| `D` | 右移 |
| `T` | 左前 |
| `R` | 右前 |
| `F` | 左後 |
| `G` | 右後 |
| `Q` | 逆時針旋轉 |
| `E` | 順時針旋轉 |
| `Space` | 停止 |

注意：

- Isaac Sim 視窗必須是焦點視窗
- 如果你發現按鍵沒反應，先點一下 Isaac Sim 視窗再試

## 8.7 Headless 錄影

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --video \
  --video_length 1000 \
  --num_envs 64 \
  --disable_keyboard_control \
  --checkpoint /absolute/path/to/model_xxxxx.pt
```

---

## 9. 評估與驗收

## 9.1 `validate_reform_stack.py`

這是改革後最基礎的健康檢查。

### 只做 random rollout smoke

```bash
python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 32
```

### 加做 1 iteration PPO smoke

```bash
python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 32 \
  --runner_smoke \
  --runner_steps 8
```

### 加做 teacher + distillation smoke

```bash
python scripts/rsl_rl/validate_reform_stack.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 32 \
  --steps 32 \
  --runner_smoke \
  --distill_smoke \
  --runner_steps 8 \
  --distill_steps 8 \
  --log_dir /tmp/redrhex_reform_smoke
```

## 9.2 `eval_command_sweep.py`

這個腳本適合做量化驗收，不是只靠肉眼看 play。

### 主 task 最終 mixed 驗收

```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 256 \
  --checkpoint /absolute/path/to/model_xxxxx.pt \
  --eval_profile stage5
```

如果你要沿用原本那套比較嚴格的 Stage5 驗收門檻，可以直接用這個版本：

```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 256 \
  --checkpoint /absolute/path/to/model_xxxxx.pt \
  --eval_profile stage5 \
  --warmup_steps 120 \
  --sweep_steps 600 \
  --accept_duration_s 2.0 \
  --accept_vx_abs 0.15 \
  --accept_vy_abs 0.15 \
  --accept_wz_abs 0.40 \
  --accept_lin_ratio 0.55 \
  --accept_wz_ratio 0.55 \
  --accept_yaw_tilt_bound 0.60 \
  --accept_yaw_tilt_ratio 0.70 \
  --accept_diag_sign_ratio 0.70 \
  --accept_diag_component_ratio 0.50 \
  --accept_max_fall_rate 0.20 \
  --accept_skill_pass_ratio 0.60 \
  --accept_overall_pass_ratio 0.70
```

### 輸出 CSV

```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs 256 \
  --checkpoint /absolute/path/to/model_xxxxx.pt \
  --eval_profile stage5 \
  --csv logs/rsl_rl/redrhex_wheg/eval_command_sweep.csv
```

`--eval_profile` 常用值：

- `stage1`
- `stage2`
- `stage3`
- `stage4`
- `stage5`
- `full`

---

## 10. 恢復訓練與常用參數

## 10.1 一般 resume

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --resume \
  --load_run <run 名稱> \
  --checkpoint model_xxxxx.pt
```

## 10.2 policy-only resume

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --resume \
  --resume_policy_only \
  --reset_action_std 0.8 \
  --load_run <run 名稱> \
  --checkpoint model_xxxxx.pt
```

## 10.3 常用 train 參數

| 參數 | 用途 |
|---|---|
| `--task` | 選 task |
| `--agent` | 選 PPO / teacher / distill 入口 |
| `--num_envs` | 環境數量 |
| `--max_iterations` | 訓練迭代數 |
| `--resume` | 從既有 checkpoint 繼續 |
| `--load_run` | 要從哪個 run 繼續 |
| `--checkpoint` | 指定 `model_*.pt` |
| `--resume_policy_only` | 只載 policy，不載 optimizer |
| `--reset_action_std` | 搭配 policy-only 時重新設 action std |
| `env.stage=<1..5>` | 指定主 task 的 curriculum stage |

---

## 11. 這次改革後，操作上有什麼實際變化

這一段專門講「你現在操作這個專案時，和舊版相比，哪些地方不同」。

### 11.1 現在不是只有單一 observation

環境現在會同時提供：

- `policy`
- `history`
- `critic`
- `teacher`

這帶來的操作差異是：

- 一般 PPO 直接用 `rsl_rl_cfg_entry_point`
- teacher PPO 要改成 `rsl_rl_teacher_cfg_entry_point`
- distillation 要改成 `rsl_rl_distillation_cfg_entry_point`

### 11.2 現在有 asymmetric critic

所以你不需要另外手改 train 腳本來分 actor/critic input，配置已經在 agent cfg 中接好。

### 11.3 現在有 symmetry augmentation

這已經在 PPO cfg 中啟用，不需要你額外傳旗標。

### 11.4 現在有 rough terrain、actuator randomization、fault injection

這些已經在主 task 環境裡。代表：

- 主 task 的訓練難度比舊版平地 baseline 更高
- 但理論上更接近真機部署需求
- 如果你只是想先快速看到「會往前走」，請先用 ForwardFast

### 11.5 `play.py` 現在會自動匯出 policy

所以現在一般流程可以簡化成：

1. train
2. play 一次
3. 直接去該 run 的 `exported/` 取 `policy.onnx` 或 `policy.pt`

不需要再另外寫一支 export script。

---

## 12. 最容易踩的坑

## 12.1 把 TensorBoard 檔案當 checkpoint

錯誤示範：

- `events.out.tfevents...`
- `params/agent.yaml`
- `exported/policy.pt`

正確做法：

- 傳 `model_*.pt`

## 12.2 `--checkpoint` 只給檔名，但沒給 `--load_run`

如果你只傳：

```bash
--checkpoint model_2500.pt
```

那通常還要搭配：

```bash
--load_run <run 名稱>
```

否則就直接傳完整絕對路徑。

## 12.3 pipeline 續跑時換了 `--run_tag`

如果你要 `--start_stage 3` 或 `--start_stage 4`，`--run_tag` 必須和前一次一致，否則 pipeline 找不到前一段 run。

## 12.4 `env.stage=1` 寫法錯了

Hydra 覆寫一定要寫：

```bash
env.stage=1
```

不是：

```bash
stage=1
```

## 12.5 鍵盤沒反應

先確認：

- 你沒有加 `--disable_keyboard_control`
- Isaac Sim 視窗有被點到
- 你不是在 headless 模式

## 12.6 `omni.platforminfo` 或 CPU 告警很多

這類訊息很多時候只是平台偵測或環境告警，不一定是主因。先優先檢查：

- 是否一出生就 `terminated`
- `Episode_Reward/diag_base_height` 是否崩掉
- 是否是 checkpoint 路徑傳錯

## 12.7 teacher 和 student 混用

記住這個原則：

- 要一般部署：優先用 `rsl_rl_cfg_entry_point`
- 要做蒸餾上界：先訓 `rsl_rl_teacher_cfg_entry_point`
- 要可部署的蒸餾學生：用 `rsl_rl_distillation_cfg_entry_point`

## 12.8 distillation 找不到 teacher checkpoint

先檢查三件事：

1. `TEACHER_RUN` 是否真的存在
2. `TEACHER_CKPT` 是否真的是 `model_*.pt`
3. 該 run 是否對 distill experiment root 可見

最穩的做法通常是先建立軟連結，再啟動 distillation。

## 12.9 系統沒有 `rg`

原本這曾經是 pipeline 會中斷的原因之一。現在 `train_stage_pipeline.sh` 已經有 `grep` fallback，所以不會因為少了 `rg` 就直接掛掉，但若你自己寫外部腳本，還是建議優先確認 `rg` 是否可用。

## 12.10 訓練時看起來像被彈飛，不像在走路

這是目前最容易讓人誤判的現象之一。

如果你直接跑：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  --max_iterations 2500
```

你現在跑到的不是「單純 forward-only 慢慢學走路」。

你實際上跑到的是：

- 主 task
- 單段 Stage5 mixed training
- rough terrain
- domain randomization
- terrain curriculum
- fault injection
- push randomization

所以早期畫面常常會出現：

- 往不同方向亂衝
- 看起來像被彈出去
- 很多機器人翻滾、滑移、跳起來

這不一定表示 PhysX 壞掉，也不一定表示 reward 失效。  
很多時候只是因為你現在的預設模式本來就很難，而且它不是只訓 forward。

另外，這個環境的 actor 也不是「完全從零動作開始學」。

- policy 會輸出 residual action
- 但環境本身還會把 command-conditioned gait / drive bias 疊上去

所以就算 policy 還很爛，機器人也已經會被餵一些驅動目標。  
這也是為什麼你會看到它不是安靜站著，而是很早就開始很兇地亂動。

### 你第一個應該做的排查

先不要直接看 Stage5 mixed task。

先改成主 task 的 forward-only 正式模式：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 256 \
  --max_iterations 100 \
  --run_name debug_stage1_forward \
  env.stage=1
```

這一步的用途是：

- 保留主 task 的正式 reward / observation stack
- 但只訓 Stage1 forward-only
- 不再混 lateral / diagonal / yaw

如果你切到這裡之後，畫面就明顯比較像「往前掙扎學走」，那代表你原本看到的大多不是 bug，而是 mixed Stage5 本來就很兇。

### 如果 Stage1 還是很像被炸出去

再做第二層排查，把訓練難度先降到最乾淨：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 256 \
  --max_iterations 100 \
  --run_name debug_stage1_clean \
  env.stage=1 \
  env.domain_randomization_enable=False \
  env.dr_push_enable=False \
  env.dr_fault_enable=False \
  env.terrain_curriculum_enable=False \
  env.terrain.terrain_type=plane
```

如果這個版本明顯正常很多，通常代表主因不是程式爆炸，而是：

- mixed command 太難
- rough terrain 太難
- push / fault / randomization 疊太多

### 你該怎麼解讀結果

有三種常見情況：

1. `Template-Redrhex-Direct-v0` 預設 Stage5 很亂，但 `env.stage=1` 明顯正常很多
- 這通常代表主因是預設 mixed-skill 訓練太激進
- 不是單純物理錯誤

2. `env.stage=1` 還是很亂，但關掉 randomization / push / fault / terrain 後正常很多
- 這通常代表主因是訓練難度疊太高
- 不是核心控制器完全壞掉

3. 即使在 `env.stage=1 + plane + no DR` 還是高速亂飛
- 這時才要高度懷疑：
  - actuator / joint target 過大
  - asset 碰撞或初始姿態有問題
  - 接觸設定或重心設定不合理

### 目前最實用的建議

- 如果你只是想確認「它有沒有在學往前走」，不要直接盯 Stage5 mixed task。
- 先用 `Template-Redrhex-Direct-v0 + env.stage=1`。
- 如果你想要更乾淨的可視化起步，再先用 `ForwardFast`。
- 等 forward-only 看起來合理後，再回去 mixed task 或五階段 curriculum。

## 12.11 `play.py` 的 `--agent` 和 checkpoint 類型不一致

這也是很常見的坑。

例如你拿：

- `logs/rsl_rl/redrhex_wheg_distill/.../model_799.pt`

去跑：

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_cfg_entry_point \
  --checkpoint /abs/path/to/model_799.pt
```

這通常會報：

- `Missing key(s): actor.* / critic.*`
- `Unexpected key(s): student.* / teacher.*`

原因不是 checkpoint 壞掉，而是：

- `rsl_rl_cfg_entry_point` 會建立一般 PPO 的 `ActorCritic`
- 但 distillation checkpoint 裡面存的是 `student` / `teacher` 權重

正確對應如下：

1. 一般 PPO checkpoint
- `--agent rsl_rl_cfg_entry_point`

2. teacher PPO checkpoint
- `--agent rsl_rl_teacher_cfg_entry_point`

3. distillation checkpoint
- `--agent rsl_rl_distillation_cfg_entry_point`

所以你如果要播放 distillation student，應該改成：

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_distillation_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --checkpoint /home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/redrhex_wheg_distill/2026-04-16_18-58-00_forward_stage1_distill/model_799.pt
```

如果你已經知道 `load_run` 和 `checkpoint` 檔名，也可以寫成：

```bash
python scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --agent rsl_rl_distillation_cfg_entry_point \
  --num_envs 64 \
  --initial_command stop \
  --load_run 2026-04-16_18-58-00_forward_stage1_distill \
  --checkpoint model_799.pt
```

另外像你 log 裡前面那串：

- `omni.platforminfo.plugin failed to retrieve CPU information`

通常不是這次失敗的主因。  
你這次真正的主因是 checkpoint 結構和 `--agent` 類型不一致。

---

## 13. 建議工作流

### 情境 A：第一次接觸專案

```text
validate_reform_stack.py
-> ForwardFast train
-> TensorBoard
-> play.py
-> exported/policy.onnx
```

### 情境 B：要做完整 locomotion 能力

```text
GUI precheck
-> 5-stage pipeline
-> eval_command_sweep.py
-> play.py
-> exported/policy.onnx
```

### 情境 C：要研究 teacher-student

```text
teacher PPO
-> validate_distillation_stack.py
-> distillation train
-> play distill student
-> exported/policy.onnx
```

---

## 14. 最後的實務建議

- 先確定 `validate_reform_stack.py` 能過，再開長訓練。
- 要快速直走，就先用 ForwardFast，不要一開始就硬打完整 mixed task。
- 要正式完整能力，優先跑五階段 pipeline，不要只靠單段 mixed 硬練。
- 要做 sim2real 部署，優先使用一般 PPO student 或 distillation student。
- 每次找到可用模型後，記得去對應 run 目錄確認 `exported/policy.onnx` 是否已經生成。
- 如果你看到行為很好但數值驗收不清楚，補跑 `eval_command_sweep.py`，不要只靠目視判斷。
- 一個很實用的節奏是：先 GUI precheck 2 到 5 分鐘，再做短迭代快篩，確認不是出生即死後，再放長訓練或整晚 pipeline。

這份文件未來若再有 train / play / export / validation 流程變更，應優先更新這裡。
