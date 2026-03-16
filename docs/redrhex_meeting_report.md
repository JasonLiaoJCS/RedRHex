# RedRhex 專案變更會議報告（完整說明版）

## 1. 這份文件的目的

這份文件是給還沒有完整參與過前期開發的組員，用來快速理解我們這段時間在 RedRhex 專案上到底做了哪些事情、為什麼要做、改了哪些程式、遇到了哪些問題、以及這些修改帶來了什麼效果與限制。

這份文件不是操作手冊。

如果要照著實際指令去訓練、驗收、播放模型，請看：

- `docs/redrhex_train_play_guide.md`

這份文件的定位是「報告用的邏輯整理」。

## 2. 專案背景：我們原本在做什麼

我們的目標是讓 RedRhex 這個六足 wheg 機器人，在同一套 policy 下具備完整的多技能 locomotion 能力，包含：

- 直走（forward）
- 側走（lateral）
- 斜走（diagonal）
- 原地旋轉（yaw turn）

原本的做法是把多種技能混在同一次訓練裡一起學，也就是單段式 multi-skill training。

這個做法理論上很直接，但在實際訓練上很容易出現以下問題：

- 不同技能的控制需求互相衝突
- 不同技能的獎勵項彼此拉扯
- policy 容易學到折衷但不夠好的動作
- 比較難診斷「到底是哪一項技能拖垮整體」
- 當某個技能特別難（尤其 lateral / yaw）時，會把整體訓練穩定性一起拉低

所以後來我們把訓練流程改成 curriculum learning，也就是分階段訓練。

## 3. 為什麼要從單段式改成分階段訓練

### 3.1 單段式的核心問題

單段式 multi-skill 訓練有一個很典型的問題：同一個 actor 在同時面對多種任務時，梯度來源是混雜的。

對 RedRhex 這類 locomotion 任務來說，這會放大幾個問題：

- 直走偏好的腿部相位規律，和側走偏好的 ABAD / 支撐邏輯不同
- 側走需要較強的側向推進與姿態穩定，但直走通常會傾向壓低橫向動作
- 原地旋轉會引入較大的角速度要求，容易讓 base height / body contact / roll 穩定性惡化
- 當所有獎勵一起生效時，policy 容易選擇「保守但不完成任務」的解

最後就會出現你實際觀察到的結果：

- 直走可能還可以
- 側走有方向但速度不夠積極
- 斜走只有一點點效果
- 旋轉很差，甚至一開始就 body contact 然後死亡

### 3.2 分階段的設計目標

分階段訓練的目標不是把技能拆散，而是先把每個子能力各自練穩，再做整合。

我們想達成的效果是：

- Stage1 先把最基礎、最重要的穩定直走打穩
- Stage2 單獨處理側走，避免被直走邏輯壓制
- Stage3 再把前向與側向組合成斜走
- Stage4 單獨處理原地旋轉，避免一開始就被混合技能干擾
- Stage5 再把前四階段的能力整合成完整技能組

這樣做的核心理由，是把「技能獨立收斂」和「最終整合」分開處理。

## 4. 為什麼最後定案成五階段，而不是四階段

我們中途曾經採用過四階段思路，但後來改成五階段，原因很明確。

原本四階段的問題在於：

- 直走、側走、旋轉雖然有分開，但斜走沒有被單獨訓練
- 斜走如果只靠前向與側向「自然外推」，通常不夠穩，也不夠強
- 真正最難的不是「會不會斜走」，而是「在不破壞穩定性的前提下，同時保留前向與側向分量」

因此我們後來改成五階段：

- Stage1：Forward-only
- Stage2：Lateral-only
- Stage3：Diagonal-only
- Stage4：Yaw-only
- Stage5：Mixed integration

這樣的好處是：

- 斜走被明確視為一個技能，而不是附帶結果
- 原地旋轉被單獨隔離出來處理
- 最後的 Stage5 才是「總整合」，而不是把某個尚未成熟的技能硬塞進中間階段

## 5. 五個 Stage 各自在負責什麼

## Stage1：Forward-only（穩定直走基礎）

Stage1 的任務是把「穩定直走」做到夠可靠，這是整個課程式訓練的基礎。

這一階段的重點：

- 只強化前向線速度
- 優先建立穩定支撐、步態規律、base height、避免 body contact
- 讓 policy 先學會「不趴地、能連續推進」

這一階段為什麼重要：

- 後面所有技能都建立在基本穩定前進能力之上
- 如果連直走都不穩，後續 lateral / diagonal / yaw 會更容易崩潰
- 直走是最容易觀察是否「控制邏輯正確」的技能

這一階段我們特別做的事：

- 讓 forward 的步態 prior 更明確
- 強調 tripod 交替與 duty factor
- 限制不必要的橫向或旋轉干擾

## Stage2：Lateral-only（純側向能力）

Stage2 的任務是處理你明確指出的問題：雖然有側走效果，但速度太慢、不夠積極。

這一階段的重點：

- 單獨強化側向速度與方向正確性
- 壓低直走邏輯對 policy 的主導
- 讓機器人真的學會把力用在橫向移動，而不是只做很保守的小幅偏移

為什麼不能讓 lateral 只當作 mixed training 的一部分：

- 如果直走獎勵同時很強，policy 很容易偏向「少量 lateral + 維持穩定 forward posture」
- 這在 TensorBoard 上會看起來像有一點 lateral，但實際位移和速度都很弱

所以 Stage2 的目的，是把 lateral 從「附帶效果」變成「主任務」。

## Stage3：Diagonal-only（前向 + 側向的組合技能）

Stage3 的任務是讓 policy 學會把 Stage1 的前向能力和 Stage2 的側向能力組合起來，形成真正可用的斜向行走。

這一階段的重點：

- 同時保留前向分量與側向分量
- 不只是方向對，還要兩個速度分量都不能太弱
- 避免 policy 退化成只剩 forward 或只剩 lateral

為什麼 Stage3 必須獨立存在：

- 斜走不是單純把 forward 與 lateral 各取一半
- 對 wheg 機構來說，斜向推進對重心轉移、支撐分佈、地面接觸節奏都更敏感
- 如果不獨立訓練，policy 很容易只會做「偏左的直走」而不是真正 diagonal locomotion

## Stage4：Yaw-only（原地旋轉）

Stage4 是最難的單一技能階段，因為它最容易引發姿態失穩與 body contact。

這一階段的重點：

- 專注在角速度輸出
- 抑制利用錯誤線速度「作弊式」轉向
- 控制旋轉時的 base height、roll、pitch、body contact

這一階段為什麼難：

- yaw command 會直接挑戰 base 姿態穩定性
- 機器人容易一開始就壓低身體、碰地、判定終止
- 如果沒有把旋轉從其他技能中隔離出來，它往往是最先把整體訓練拖垮的任務

所以 Stage4 的定位不是「順便學轉彎」，而是「把最容易導致死亡的旋轉控制單獨訓練穩定」。

## Stage5：Mixed（完整技能整合）

Stage5 是整套 curriculum 的總整合階段。

這一階段不是從零開始教技能，而是要做三件事：

- 把前四階段各自收斂的子能力整合成同一個 command-conditioned policy
- 讓 policy 學會根據不同指令切換對應控制模式
- 在整合後重新潤化，降低技能切換時的衝突和遺忘

這一階段的真正價值：

- 不是「再學一遍全部技能」
- 而是「讓已經存在的技能共存，而且切換時不要互相污染」

對外報告時，可以把 Stage5 描述成：

「前四個 stage 是把各個 locomotion primitive 單獨做穩，Stage5 則是把這些 primitive 放回同一套 policy 裡重新整合，讓模型學會依照 command 做模式切換，同時透過額外訓練把技能切換造成的干擾再抹平。」

這就是你想強調的「整合前面能力並持續潤化」的核心意義。

## 6. 我們做了哪些主要程式修改

這一段是整場報告最重要的技術內容，因為它對應到「我們不是只改訓練步驟，而是有實際改底層程式」。

## 6.1 `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py`

這個檔案是整個 locomotion 邏輯的核心，包含：

- 動作如何轉成關節控制
- 各種 command mode 如何切換
- 各種 reward 與診斷值如何計算
- 訓練和 play/eval 的外部控制如何進入環境

我們對這個檔案的修改，核心方向有四個。

### A. 把控制邏輯改成「技能模式感知」

舊思路比較接近「同一套控制直接吃所有情境」。

新思路則是：

- 依據目前 command 判斷是 forward / lateral / diagonal / yaw
- 不同模式下，允許不同的輸出結構
- 某些模式會抑制特定動作通道，避免互相干擾

這樣做的原因是：

- 不同技能本來就不該完全共用同一種動作分配
- 如果不做模式感知，policy 會傾向學出中庸但不夠好的解

### B. 加入更明確的 forward gait prior

直走是基礎能力，所以我們在 forward 模式裡強化了步態先驗：

- Tripod 交替
- 特定相位關係
- 著地較慢、擺動較快的非對稱 duty-cycle
- 期望 stance fraction 接近設計值

這樣做的原因是：

- 純讓 RL 自己亂搜，常常會學到高頻抽搐或六腿同轉
- 先給合理的 locomotion prior，可以降低探索空間
- 對 wheg 機器人而言，步態先驗能顯著提升起步穩定性

### C. 讓 reward 與診斷值分開處理

你之前注意到 TensorBoard 裡很多 `rew_*` 和 `diag_*` 值，這件事本身就是刻意設計出來的。

我們的方向是：

- reward 項負責真的影響學習
- diagnostic 項負責讓人看懂模型在做什麼

也就是說：

- 某些 reward 會在特定 stage 被抑制或設為零
- 但診斷值仍盡量保留輸出，方便觀察

這樣做的原因是：

- 訓練需要聚焦，不然不同 reward 會互相打架
- 但如果把所有非主技能訊號都拿掉，就很難 debug

這也是你在 TensorBoard 看到「很多欄位存在，但某些 stage 會是 0」的主要原因。

### D. 加入 play/eval 的 forward 相容保護

這是針對你後來遇到的一個很關鍵問題：

- 舊版穩定的直走模型，理論上以前可以正常走
- 但在新版控制邏輯下，用同樣 checkpoint 去 play，卻出現「六支腳一起轉、機身趴地、body contact、死亡」

我們最後確認，這不一定代表舊模型壞掉，而是可能有兩個系統層面的原因：

- 新版控制器的輸出解讀方式和舊版訓練時不同
- 一進 play 就直接給 forward 指令，起步太激進

因此在 `redrhex_env.py` 中，我們加了「forward 相容保護」：

- 只在 pure forward 模式下觸發
- 預設只在 `external_control=True` 時生效
- 也就是只影響 play / eval 的外部命令覆寫，不影響正常訓練
- 它會把 pure forward 的輸出拉回比較保守的 forward bias，再只允許小幅 residual 修正

這個機制的目的不是讓模型作弊，而是：

- 讓舊 checkpoint 在新版控制器下仍然能以更接近舊行為的方式起步
- 把「控制器更新造成的回放退化」和「模型本身學壞」分開

這是保留「舊穩定直走性能」最重要的一個相容性修正。

## 6.2 `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`

這個檔案負責環境設定值與超參數。

我們在這裡做的關鍵修改，是把上面提到的相容保護做成可配置參數，而不是寫死。

目前有幾個重要設定：

- `play_forward_compat_enable`
- `play_forward_compat_only_external`
- `play_forward_compat_bias_scale`
- `play_forward_compat_residual_scale`
- `play_forward_compat_residual_clip`

這樣做的好處是：

- 可以快速開關相容模式
- 可以微調「舊模型保守起步」的強度
- 不需要每次都直接改主邏輯

從工程角度看，這比硬編碼在 `redrhex_env.py` 裡更可維護，也更安全。

## 6.3 `scripts/rsl_rl/train_stage_pipeline.sh`

這個檔案是把五階段訓練串起來的自動化 pipeline。

這個腳本的重要性非常高，因為它直接解決了你之前的一個核心痛點：

- 不想每跑完一個 stage 就半夜起來手動接下一個 checkpoint

### 這個腳本做了什麼

- 自動依序執行 Stage1 到 Stage5
- 自動找到前一個 stage 的 checkpoint
- 自動把下一個 stage 接在上一個 stage 後面
- 最後輸出 `FINAL_RUN` 和 `FINAL_CKPT`

也就是說，它把「概念上分階段」和「操作上可整夜連跑」這兩件事同時成立。

### 為什麼這樣設計

你曾經問過：

「五個 stage 一定要分開嗎？可不可以一次跑完？」

我們的答案是：

- 學習邏輯上，要分階段
- 執行流程上，可以自動串起來

這就是這支腳本存在的價值。

### 這支腳本的關鍵修正

#### A. 預設使用 full resume，而不是只載入 policy

這一點非常重要。

在 `train_stage_pipeline.sh` 裡，現在預設：

- `RESUME_POLICY_ONLY=0`

這代表：

- 不只是接續權重
- 也接續 optimizer state
- 也保留 iteration / runner 狀態的連續性

這樣做的意義是：

- 真正比較接近 curriculum learning 的「接續學習」
- 不只是把上一階段當成 pretrained initialization

如果每一個 stage 都重新從 `model_0.pt` 開始編號，通常代表至少 runner 層級是重新起算的，這會讓「延續性」在實務上變弱，也更容易產生遺忘。

你之前發現舊版五個資料夾的 `model_*.pt` 編號是接續的，這正是比較合理的接續訓練型態。

所以這次的 pipeline 調整，核心就是把這件事拉回來。

#### B. 加入 GUI 預檢

這是直接對應你「整晚白跑」的痛點。

之前你已經明確遇到過：

- headless 放整晚
- 結果一出生就死、在空中抽搐
- 隔天才知道整段根本沒有效學習

所以我們在 pipeline 裡加入了：

- `--precheck_gui`
- `--precheck_stage`

這個流程的意義是：

- 在正式 headless 長訓練前，先用 GUI 肉眼確認起步正常
- 先確認模型至少「有在訓練」，而不是一出生就死

這不是花俏功能，而是避免浪費 GPU 夜間時數的必要工程控制。

#### C. 加入 stage health gate

這個設計是為了自動檢查某個 stage 是否已經失控。

它會從 log 中抓幾個基本健康指標，例如：

- `Mean episode length:`
- `Episode_Termination/terminated:`

如果這些指標明顯不合理，就會直接在 pipeline 中止，而不是盲目繼續後續 stage。

這樣做的原因是：

- 如果 Stage1 已經爛掉，繼續跑 Stage2~5 只是在把錯誤放大
- curriculum 的前一階段如果沒站穩，後面很難補救

#### D. `rg` 缺失時的 fallback

你曾經遇到過：

- `rg: 指令找不到`
- 導致 health gate 無法從 log 取值

所以我們在腳本裡做了 fallback：

- 若 `rg` 存在，優先用 `rg`
- 若 `rg` 不存在，改用 `grep`

這是一個小修改，但很重要，因為它避免 pipeline 因為系統工具缺失而中斷。

## 6.4 `scripts/rsl_rl/play.py`

這個檔案負責載入訓練好的模型做播放與肉眼檢查。

它是你這段時間出現問題最多、也最容易誤判「模型壞掉」的地方之一。

### A. 修正 checkpoint 路徑解析

你反覆遇到的一個錯誤是：

- `--checkpoint="$FINAL_CKPT"` 實際上指到的是 `events.out.tfevents...`
- `runner.load()` 嘗試把 TensorBoard event file 當成 PyTorch checkpoint 載入
- 最後噴出：
  - `UnpicklingError: invalid load key, 'H'`

這個問題的本質不是模型壞掉，而是「讀錯檔案」。

所以 `play.py` 現在做了更嚴格的 checkpoint 防呆：

- 只把 `model_*.pt` 當作合法的訓練 checkpoint
- 如果你傳進來的是 `events.out.tfevents...`
- 它會先找同資料夾最近的 `model_*.pt`
- 找不到再往 `logs/rsl_rl/redrhex_wheg` 底下遞迴找最新的 `model_*.pt`

這個修改的好處是：

- 降低人為路徑失誤
- 減少誤把 event file、`policy.pt`、匯出檔當成訓練 checkpoint 的機率

### B. 把實際啟動的預設命令改成 `stop`

這也是一個非常關鍵的穩定性修正。

你觀察到的症狀之一是：

- 一進 play 就六腳同轉
- 然後趴地
- 然後 body contact
- 然後死亡

這種情況如果一開始就直接給 `forward`，問題會被放大。

所以現在 CLI 的預設：

- `--initial_command` 預設是 `stop`

也就是說，一打開 play 時，先讓模型站穩，而不是立刻推進。

這樣做的目的很直接：

- 先把「起步穩定性」和「命令追蹤能力」拆開檢查
- 避免一啟動就用最敏感的 forward 命令把問題掩蓋掉

要注意一點：

- `KeyboardController` 類別內部建構子的預設參數仍保留 `forward`
- 但 `play.py` 現在會把 CLI 的 `stop` 明確傳進去
- 所以實際運行時的預設行為已經是 `stop`

也就是說，對使用者來講，現在的預設起步是安全得多的。

### C. 加入從 checkpoint 路徑自動推斷 `env.stage`

在 `play.py` 裡，現在會根據 checkpoint 路徑中的 `stage1` 到 `stage5` 自動推斷目前應該用哪個 stage 的環境設定。

這個設計的意義是：

- 避免拿 Stage4 模型卻用 Stage1 的環境模式去播放
- 減少「同一個模型在錯誤 stage 設定下被誤判」的情況

如果你不想自動推斷，也可以關閉：

- `--disable_auto_stage_from_checkpoint`

這樣做的理由，是把「方便性」和「可控性」同時保留。

## 6.5 `scripts/rsl_rl/eval_command_sweep.py`

這個檔案是我們新增的標準化驗收工具。

它的目標是把「肉眼看起來好像有在動」轉成一組可量化的驗收標準。

### 這支程式在做什麼

它會載入一個指定 checkpoint，然後在固定的指令集合下做掃描，例如：

- forward
- left
- right
- diag_left
- diag_right
- yaw_ccw
- yaw_cw

再針對每種 command 計算：

- 速度誤差
- 角速度誤差
- fall rate
- stance / swing 相關指標
- contact histogram
- roll / pitch RMS
- 技能是否通過 acceptance gate

### 為什麼要多做這支程式

因為你已經實際遇到過：

- 肉眼覺得 forward 還不錯
- 但 lateral、yaw 幾乎不行
- 如果沒有統一驗收，容易因為單一技能看起來還可以，就誤以為整體模型已經可用

這支程式的價值是：

- 把多技能能力拆開驗收
- 能清楚說出哪個技能 PASS、哪個 FAIL
- 能看出失敗是因為追蹤不夠、姿態不穩，還是 fall rate 太高

### 它會不會「自動切換 stage 去評估」？

這個問題你有特別問過，答案要說清楚：

- 它不會自己去輪流載入五個 stage 的 checkpoint
- 它一次只評估「你現在指定載入的那一個模型」
- 但它可以根據 checkpoint 路徑自動推斷 `env.stage`

也就是說：

- 它會自動套對應的環境 stage 設定
- 但不會自動替你把 Stage1 到 Stage5 全部跑一輪

如果你要評估五個 stage，就要各自指定對應 checkpoint 執行五次。

### 為什麼它對我們很重要

因為它提供了你後來一直在講的那些結論，例如：

- `PASS: forward, diag_left`
- `FAIL: left, right, diag_right, yaw_ccw, yaw_cw`
- `stability.fall_rate = 1.0`
- `roll_rms = 1.58`

這些數字讓我們能從「感覺不好」進一步變成「知道是哪裡不好」。

### 這支程式也做了 checkpoint 防呆

和 `play.py` 一樣，`eval_command_sweep.py` 也補上了：

- 只接受 `model_*.pt`
- 若傳錯 event file，會嘗試自動 fallback 到同資料夾的 `model_*.pt`

目的是避免驗收程式和播放程式出現相同的載入錯誤。

## 6.6 `docs/redrhex_train_play_guide.md`

這份文件現在已經被整理成主要操作指南，並且合併了原本 explainer 的操作性內容。

它現在包含：

- 為什麼改成五階段
- 每個 stage 的功能
- 一致性檢查
- GUI 預檢
- 一鍵 pipeline
- 手動逐段訓練
- TensorBoard 觀察重點
- eval 指令
- play 指令
- 常見錯誤與排除

這樣做的理由是：

- 把散落在不同文件、不同對話裡的流程整合成單一可信來源
- 降低組員照著舊版說明操作而踩坑的機率

## 6.7 `docs/redrhex_stage_training_explainer.md`

這份文件現在的角色已經改成導向頁。

它的用途是：

- 保留舊檔名，避免連結斷掉
- 告訴使用者主要內容已整併到 `docs/redrhex_train_play_guide.md`

這是一個文件管理層面的整理，不是功能修改，但對團隊協作很重要。

## 7. 舊版穩定版本和新版到底差在哪裡

你特別要求我們參考的舊版穩定節點是：

- `ca4b1ab`：`橫走直走穩定版本`

目前分支上的新版本是：

- `eca15ea`：`5stage-new-stable`

這兩者的根本差異，不只是「參數不同」，而是開發方向不同。

舊版穩定版本的特性：

- 直走、橫走已有一定穩定性
- 控制邏輯相對比較單純
- 在特定版本的 play / env 組合下，直走表現良好

新版的目標：

- 擴展到完整五階段 curriculum
- 納入 diagonal 與 yaw 的顯式訓練
- 增加更完整的評估、驗收與自動化流程

因此新版的挑戰不只是「把舊功能保留住」，還要：

- 不破壞原本穩定的 forward
- 同時讓其他技能可被獨立訓練與最終整合

這也是為什麼你會看到某些新版本一度出現：

- forward 被弄壞
- 舊模型回放也變差

因為我們不是只換訓練長度，而是連控制邏輯、播放路徑、stage handoff 都一起在變。

所以後來我們做的修正策略，不是單純「回退」，而是：

- 把舊穩定 forward 的價值保留下來
- 再用相容保護和 pipeline 設計，讓新技能能加上去

## 8. 我們如何回應你提出過的每一個核心問題

這一段是會議中特別好用的，因為它直接把你一路提出的痛點對應到工程修正。

## 問題 1：側走有效果，但速度太慢，不夠積極

### 成因

- lateral 在 mixed setting 中容易被 forward 邏輯壓制
- policy 可能選擇「保守偏移」而不是「積極側推」
- 如果穩定性懲罰太強，也會讓 lateral 動作變得畏縮

### 我們怎麼做

- 把 lateral 從混合技能中拆成 Stage2 單獨訓練
- 在 stage-specific 邏輯中，讓 lateral 有獨立主目標
- 透過 TensorBoard 與 eval 指標單獨觀察側向能力

### 好處

- 可以明確判定 lateral 是不是自己學不起來
- 可以避免「整體 reward 看起來不差，但 lateral 實際很弱」的錯覺

## 問題 2：斜走只有一點點效果，而且前面階段似乎沒有真的練斜走

### 成因

- 如果 diagonal 沒被單獨當作一個技能，通常只會變成 forward / lateral 的副產品
- 這種副產品通常方向似乎對，但速度分量不完整、穩定性也不夠

### 我們怎麼做

- 把 diagonal 從「混合裡自然出現」改成 Stage3 單獨訓練
- 在 eval 中為 diagonal 加入單獨的 sign / component acceptance

### 好處

- 斜走不再是模糊的附帶能力，而是可驗收的技能
- 可以知道 diagonal 失敗是方向錯，還是某個分量太弱

## 問題 3：原地旋轉很差，一開始就 body contact 然後死掉

### 成因

- yaw 是最容易導致姿態失穩的技能
- 若與其他技能混訓，policy 很容易學不到正確旋轉，而只會用錯誤線速度或直接失穩
- 一旦 base height、roll、body contact 沒壓住，就會快速觸發 termination

### 我們怎麼做

- 把 yaw 單獨拆成 Stage4
- 在 eval 中加入角速度與穩定性判斷
- 在流程上加入 GUI 預檢與 stage health gate，避免一整晚都在死

### 好處

- 旋轉問題不會再被混在其他技能裡看不清楚
- 可以獨立判斷 yaw 失敗是控制不對，還是穩定性不夠

## 問題 4：五個 stage 資料夾裡的 `model_*.pt` 編號都重置，這樣真的有接續學習嗎？

### 成因

- 如果是新建 runner 或只載入 policy，不保留 optimizer / state，形式上會像是「接著訓練」，但延續性比較弱
- 這容易造成 curriculum 的接續效果不足，甚至較容易遺忘前一階段技能

### 我們怎麼做

- 在 `train_stage_pipeline.sh` 預設採用 full resume
- 不是只接權重，而是盡量延續整個訓練狀態

### 好處

- 更符合真正的 curriculum learning
- 降低每一階段都像重新開一個訓練的問題

## 問題 5：`play.py` 可能有問題，因為以前正常的舊模型現在也會壞

### 成因

這個判斷是合理的，而且後來證明確實有兩個「不是模型本身」的可能來源：

- `play.py` 可能載錯檔案（把 `events.out.tfevents...` 當 checkpoint）
- `play.py` 一進來就給 forward，放大起步不穩
- 新版 env 控制邏輯和舊版模型的輸出假設不完全一致

### 我們怎麼做

- `play.py` 加入 checkpoint 路徑防呆
- CLI 預設起步改成 `stop`
- env 加入 pure forward 的 play/eval 相容保護

### 好處

- 把「回放問題」和「訓練問題」分開
- 降低誤把工具鏈問題誤判為模型壞掉的機率

## 問題 6：`eval_command_sweep.py` 會不會自動切換各種 stage？

### 正確答案

- 不會自動輪流跑五個 stage
- 它一次只載入一個指定 checkpoint
- 但會根據 checkpoint 路徑自動推斷 `env.stage`

### 這代表什麼

- 它是「單模型、多命令」驗收工具
- 不是「多 stage 批次驗收」工具

## 問題 7：是不是有程式控制每個 stage 使用哪些獎勵，哪些獎勵被抑制？

### 正確答案

有，而且核心就在環境邏輯中。

概念上是：

- 環境維持一整套完整的 reward / diagnostic 結構
- 依據當前 stage 與 command mode，讓部分 reward 項有效、部分項目被抑制或變成零

這也是為什麼在 TensorBoard 上：

- 你看得到很多 `rew_*`
- 但某些 stage 會有大量 `0.0000`

這不是壞掉，而是因為那些 reward 項在當前 stage 本來就不是主訓練目標。

### 為什麼這樣設計

- 訓練面：聚焦
- 工程面：可觀測

也就是同時滿足：

- 不讓所有 reward 打架
- 仍然保留觀察與 debug 能力

## 9. TensorBoard 上為什麼有些數值看起來「不合理」

這是你後來很常觀察、也很重要的一個面向。

有三種常見原因。

## 原因 A：reward 被刻意抑制

如果目前 stage 不在訓練某個技能，對應的 reward 很可能就是 0。

這是設計結果，不代表失敗。

## 原因 B：診斷值存在，但不直接參與當前學習

例如：

- `diag_*` 可能是我們保留下來做監控用
- 它不是主要 reward，但能幫我們判斷姿態、接觸、速度誤差

所以不能只看「是不是非零」，而要看它是否符合該 stage 的預期趨勢。

## 原因 C：某些曲線不穩，代表策略還沒收斂或根本在崩壞

例如你曾經看到：

- episode length 很低
- termination 很高
- 表示機器人常常很快就死

這種情況就不是單純 reward 0，而是訓練健康度本身有問題。

因此我們後來才會加：

- GUI 預檢
- health gate
- 更清楚的 stage 拆分

## 10. 我們這一輪開發帶來的主要好處

這部分是向會議對象說明「為什麼這些改動值得」。

## 好處 1：把多技能問題拆成可診斷的子問題

以前是整體看起來不好，但不知道是誰拖垮誰。

現在可以分辨：

- forward 穩不穩
- lateral 有沒有真的推進
- diagonal 是不是兩個分量都有
- yaw 是不是一開始就倒

## 好處 2：把課程式訓練做成可自動化的流程

以前要手動接力。

現在可以：

- 分階段學習
- 但用 pipeline 一次整夜跑完

這兼顧了訓練品質與操作效率。

## 好處 3：減少夜間白跑的風險

透過：

- GUI 預檢
- health gate
- 更嚴格的 checkpoint 防呆

我們把「跑了一整晚才發現根本沒在正確訓練」的風險壓低了。

## 好處 4：把回放與驗收工具工程化

以前可能靠肉眼和手動切命令。

現在：

- `play.py` 更安全
- `eval_command_sweep.py` 可以標準化驗收

這讓開發不再只靠直覺，而是有比較客觀的驗收流程。

## 好處 5：在新架構下，仍試圖保留舊版穩定直走能力

這是這輪修改最重要的取捨之一。

我們不是要推翻舊版，而是：

- 保留舊版穩定前進的價值
- 再往上擴展成完整技能組

所以才有：

- play 起步改 `stop`
- forward 相容保護
- full resume handoff

這些都不是額外包袱，而是為了讓新舊能力能平順銜接。

## 11. 目前仍存在的限制與風險

這一段在會議上也必須講，因為這代表我們是誠實評估，不是只報喜。

目前還不能過度樂觀的點包括：

- lateral 雖然被獨立拉出訓練，但是否已達到你要的積極速度，還要靠最新一輪訓練與 eval 再確認
- diagonal 雖然已被獨立建模，但 Stage3 與 Stage5 是否真的保住兩個分量，仍需持續看 command sweep 結果
- yaw 仍是最脆弱技能，最容易因姿態失穩造成終止
- curriculum 即使做了 full resume，也不代表就完全不會遺忘，Stage5 的整合品質仍取決於訓練時間、reward 配比、stage 難度設計
- play/eval 相容保護是工程緩衝，不是根本上替代真正好的控制策略

這些點都代表：系統更完整了，但不代表所有技能已經完全達標。

## 12. 我們目前最合理的總結

如果要在會議上用一句話總結，可以這樣說：

「我們把原本一次混訓多技能的 RedRhex locomotion 流程，重構成五階段的 curriculum learning，並補上自動接續訓練、標準化驗收、播放相容保護與文件整併，目標是在保留舊版穩定直走能力的前提下，把側走、斜走、原地旋轉逐一練穩，最後再整合成完整技能組。」

如果要再技術一點，可以這樣說：

「這次的重點不是只拉長訓練時間，而是把訓練結構、控制邏輯、checkpoint handoff、驗收工具與回放工具一起重構，讓問題從『整體看起來不好但不知道為什麼』，變成『每個技能可以被單獨訓練、單獨驗收、再做整合』。」

## 13. 報告時建議的講述順序

如果你要在會議上口頭講，可以照這個順序講。

### 第一段：先講原始問題

- 原本是單段式 multi-skill 訓練
- 直走勉強可用，但 lateral、diagonal、yaw 都不理想
- 而且很難知道是哪個技能把系統拖垮

### 第二段：講我們的設計轉向

- 改成五階段 curriculum
- 把 forward / lateral / diagonal / yaw 拆開
- 最後用 Stage5 做整合

### 第三段：講我們不是只改流程，而是有改程式

- `redrhex_env.py`：控制與相容邏輯
- `redrhex_env_cfg.py`：可調參數
- `train_stage_pipeline.sh`：自動接續訓練
- `play.py`：安全播放與 checkpoint 防呆
- `eval_command_sweep.py`：標準化驗收
- `docs/redrhex_train_play_guide.md`：單一整合指南

### 第四段：講為什麼這些改動是必要的

- 避免技能互相干擾
- 避免整晚白跑
- 避免讀錯 checkpoint
- 避免舊模型因工具鏈改動被誤判壞掉

### 第五段：講現在得到的結果

- 流程變得可分解、可追蹤、可驗收
- forward 的穩定性被當作基礎能力保護
- 其他技能有了獨立訓練空間
- 但 lateral / diagonal / yaw 仍需看最新訓練結果持續調整

## 14. 這次我們實際參考了哪些資料

如果組員問「你們是根據什麼做這些修改」，可以這樣回答。

主要參考不是某一篇單一論文，而是三類資料。

### 第一類：你們自己已有的穩定版本與實驗紀錄

- 舊穩定 commit：`ca4b1ab`（`橫走直走穩定版本`）
- 現在的五階段版本：`eca15ea`（`5stage-new-stable`）
- 既有 TensorBoard 曲線
- 既有 `model_*.pt` 訓練結果

這些是最重要的實證基準，因為它們直接反映你們自己的機器人與環境。

### 第二類：目前程式碼本身

- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py`
- `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py`
- `scripts/rsl_rl/train_stage_pipeline.sh`
- `scripts/rsl_rl/play.py`
- `scripts/rsl_rl/eval_command_sweep.py`

這些檔案是我們實際改動和驗證的基礎。

### 第三類：方法論上的常見做法

雖然這次不是直接照抄某個實驗室的單一實作，但設計方向明顯遵循了幾個成熟方法：

- curriculum learning
- command-conditioned locomotion
- stage-specific reward shaping
- 將 reward 與 diagnostic 分離
- 用標準化 sweep 做多技能驗收

也就是說，我們是把通用方法論套用到你們自己的 RedRhex 環境與控制架構上。

## 15. 你在會議上可以直接用的結論

如果你要一段比較完整、可以直接講的結語，可以用下面這段：

「這一輪我們的重點，不是單純把訓練時間拉長，而是把整個 RedRhex 多技能 locomotion 的開發流程重構。原本一次混訓所有技能，導致直走之外的技能互相干擾、也很難診斷問題。後來我們改成五階段 curriculum：先把直走打穩，再把側走、斜走、原地旋轉分別拆開訓練，最後再用 Stage5 做完整整合。配合這個流程，我們同時修改了環境控制邏輯、stage 接續訓練腳本、播放工具與評估工具，並加入 checkpoint 防呆、GUI 預檢與健康檢查，目標是避免整晚白跑、避免誤讀模型、並盡量保留舊版穩定直走能力。現在最大的進展是，整個問題已經被拆成可追蹤、可驗收、可逐步修正的工程流程；接下來的重點，不是重做架構，而是根據這套架構繼續把 lateral、diagonal、yaw 的品質推上去。」 

