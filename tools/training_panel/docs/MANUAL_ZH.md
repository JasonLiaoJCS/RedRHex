# RedRHex Training Panel 使用手冊

版本線：V3.4 First Release，含 Mother 端訓練佇列支援。

發布：BioRoLa ABAD RHex Team  
致謝：Jason Liao、Jacob Yang

## 1. 這是什麼

RedRHex Training Panel 是 RedRHex RSL-RL 實驗的控制台，可以用來訓練、播放、錄影、匯出、整理歷史、寫筆記、管理遠端團隊使用。

它有兩個入口：

- **Mother**：跑在訓練電腦上的本機管理面板，功能最完整。
- **RedRHex To Go**：給團隊用的 child 手機/網頁版，較簡潔、較安全、適合遠端操作。

Mother 掌控機器；Child 負責優雅地排隊。

## 2. 啟動 Mother

在 repo 根目錄執行：

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

打開：

```text
http://127.0.0.1:8080
```

區網使用：

```bash
python -m tools.training_panel --host 0.0.0.0 --port 8080
```

SSH tunnel：

```bash
ssh -L 8080:127.0.0.1:8080 user@host
```

然後在本機瀏覽器開 `http://127.0.0.1:8080`。

## 3. 一般訓練流程

1. 進入 `Train`。
2. 選 task、env 數量、iteration 數量、device、reward preset、terrain preset。
3. 按 `Train`。
4. 如果 GPU 空閒，Mother 會立刻開始訓練。
5. 如果 Isaac 正忙，Mother 會建立一個 `queued` run。
6. 到 `History` 和 `Process Console` 觀察。
7. 有 checkpoint 後，可以用 `TensorBoard`、`Play`、`Record Video`、`Export ONNX`、`Resume to Train`。

### 佇列規則

Mother 會序列化這些 Isaac/GPU 行為：

- training
- play
- video recording
- ONNX export

只要其中一個正在跑，新訓練就會進入 `queued`。如果不想等了，按 `Cancel Queue`。

這是為了避免「同時開很多 Isaac，然後 GPU 表演記憶體消失術」。

## 4. Rewards

`Rewards` 用來調 reward scale 和管理 preset。

規則：

- 內建 preset 不能直接改。
- 要改請先 duplicate。
- 目前選中的 reward preset 會用於下一次訓練。
- History 可以比較不同 run 的 reward 差異。

建議命名像實驗筆記，不要像午夜即興。`stairs_stable_v2` 比 `final_final_really` 好很多。

## 5. Terrain

`Terrain` 用來調 terrain generator、curriculum、sub-terrain 比例。

規則：

- 內建 terrain profile 不能直接改。
- 要改請先 duplicate。
- 目前選中的 terrain preset 會用於訓練。
- Play 和 Video 會盡量使用 run 當初訓練時保存的 terrain 設定。

如果播放結果不像當初訓練環境，先看該 run 的：

```text
params/env.yaml
```

這個檔案是地形設定的時間膠囊。

## 6. History

History 是實驗資料庫。

每張卡會顯示：

- run 狀態
- task/envs/iterations
- reward/terrain 差異
- checkpoint 狀態
- video / ONNX 狀態
- notes / folder 狀態

常用功能：

- `TensorBoard`：看 metrics。
- `Play`：用 checkpoint 開 Isaac 播放。
- `Record Video`：錄製高品質 MP4。
- `Export ONNX`：匯出 `exported/policy.onnx`。
- `Resume to Train`：從最新 checkpoint 接著訓練。
- `Compact Run`：刪舊的 top-level `model_*.pt`，保留最新 checkpoint。
- `Console`：看完整指令與輸出。
- `Compare`：比較 run 之間的 reward / terrain 差異。

Folder 是人類理智維持器，請早點用。

## 7. Process Console

Process Console 有兩個主要區塊：

- **Launch Command**：Mother 實際送出的指令。
- **Output**：Isaac / RSL-RL 的輸出。

按鈕：

- `Copy Command`
- `Copy Output`
- `Pop Out`
- `Open Log Folder`
- `Stop Process`

如果系統有 `tmux`，Mother 會把 Isaac job 跑在 detached tmux session 裡。SSH 進去後可以 attach，然後直接 `Ctrl+C`，很工程，很清爽。

## 8. Videos

Mother 預設錄製高品質結果影片：

- 1920x1080
- 1200 steps
- 30 FPS
- quality rendering

影片會盡量標示 checkpoint iteration，讓你知道現在看的到底是哪個模型。

如果 video 失敗：

1. 開 `Console`。
2. 看 diagnosis。
3. 檢查 GPU memory。
4. 等其他 Isaac job 停掉後重試。

## 9. ONNX 匯出

有 checkpoint 後，在 History 按 `Export ONNX`。

輸出位置：

```text
<run log dir>/exported/policy.onnx
```

如果 ONNX 匯出失敗，檢查：

- checkpoint 是否存在
- Isaac 是否能正常啟動
- 是否有其他 GPU action 正在跑

## 10. Compact Run

`Compact Run` 只會刪除 top-level 舊 checkpoint：

```text
model_0.pt
model_100.pt
model_200.pt
```

它會保留數字最大的 `model_N.pt`。

它不會刪：

- videos
- TensorBoard logs
- params
- notes
- exported policy files

刪除前會要求確認，因為刪檔不應該是驚喜。

## 11. RedRHex To Go

RedRHex To Go 是 child 網頁，主要給手機和團隊遠端使用。

支援：

- connection health
- training queue
- reward tuning
- terrain tuning
- history
- folders / notes
- team video playback
- safe remote actions

不提供：

- 本機檔案開啟
- raw tmux 控制
- 完整 terminal debug
- worker secrets

網址：

```text
https://popcorn-volcano.github.io/redrhex-training-remote/
```

## 12. Remote Worker

Mother 的 `Control Center` 用來管理 remote worker。

可做：

- start worker
- stop worker
- restart worker
- 選 `tmux` 或 child process mode
- 啟用 auto-start
- accept / pause remote jobs
- 檢查設定狀態

Secrets 放在：

```text
~/.redrhex_remote.env
```

不要把 machine token 貼到 GitHub Pages 或任何 browser code。

## 13. 權限角色

- `viewer`：查看 runs 和 artifacts。
- `operator`：可以 queue training 和安全操作。
- `admin`：可以管理 access 和 destructive actions。

角色在 Supabase 的 `profiles` 表設定。

## 14. 通知

通知可以送到 requester 的 Discord。

常見事件：

- convergence detected
- training completed
- training failed/interrupted
- video ready

如果通知很安靜，檢查：

- child 的 Connection page
- Supabase Edge Function secrets
- `run_events`
- requester profile/settings

## 15. Activity

Mother 的 `Activity` 是團隊 mission control。

它會顯示：

- contribution score
- training 次數
- 成功/失敗比例
- videos / ONNX
- active members
- recent failures
- 依使用者折疊的活動紀錄

這頁是拿來做團隊回顧，不是拿來審判。資料是手電筒，不是槌子。

## 16. 疑難排解

### Port Already In Use

`8080` 已經被占用。

```bash
lsof -i :8080
```

停掉舊 panel，或換 port。

### GPU Out Of Memory

常見訊息：

```text
CUDA error: out of memory
PhysX failed to allocate GPU memory
Failed to get DOF velocities from backend
```

檢查：

```bash
nvidia-smi
```

停掉舊 Isaac job、關掉多餘 play/video，再重試。佇列功能就是為了減少這種撞車。

### No Checkpoint

訓練沒有跑到保存 `model_*.pt`，或在建立 log 前就失敗。

開 `Console` 看輸出。

### Video Missing

可能原因：

- checkpoint 不存在
- GPU action 撞在一起
- movie/encoder 問題
- Isaac playback 啟動失敗

先看 `Console`，等 GPU 空了再重試。

### Child 顯示 Offline

看 Mother 的 `Control Center`：

- env file 是否存在
- worker 是否 running
- accepting jobs 是否開啟
- Supabase heartbeat 是否新
- 如果要外網 console/TensorBoard，Cloudflare tunnel 是否正常

## 17. 維護清單

Demo 前：

- pull 最新 repo
- restart Mother
- 從 `Control Center` 啟動 remote worker
- 確認 Child 顯示 machine online
- 跑一次小 smoke training
- 確認 History 更新
- 確認影片能播放

長訓練日開始前：

- 看 `nvidia-smi`
- 磁碟壓力大就 compact 舊 runs
- 把新實驗放進 folder
- 趁腦袋還記得，先寫 notes
