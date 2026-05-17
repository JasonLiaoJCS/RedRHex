# RedRHex To Go 中文使用手冊

`RedRHex To Go` 是 RedRHex Training Panel 的 child 網頁版：給團隊在手機或電腦上遠端查看狀態、調 reward/terrain、排訓練、看影片、寫註記。

一句話版本：

> Child 負責讓團隊到處都能下指令；Mother 負責在訓練電腦上真正開 Isaac、跑模型、管 worker、查 terminal。

入口：

```text
https://popcorn-volcano.github.io/redrhex-training-remote/
```

---

## 1. 使用前檢查

你需要：

- 一個已加入 Supabase 專案的帳號。
- `profiles` 裡有你的權限角色：`viewer`、`operator` 或 `admin`。
- Mother 電腦上的 remote worker 正在跑。
- Mother Control Center 顯示 worker online，且 `Accept Jobs` 是開啟的。

打開 child 後，先去 `Connection` 看健康狀態。理想畫面大概是：

- Auth：已登入。
- Profile：有角色。
- Database：正常。
- Machine：online。
- Worker：accepting jobs。
- GPU：free 或 busy，但不能是未知。
- Storage：ready，影片才會比較開心地出現。

如果 `Machine` 超過 90 秒沒 heartbeat，通常代表 Mother 或 remote worker 沒在正常工作。這時候手機不用生氣，去 Mother 的 Control Center 看 worker。

---

## 2. 角色權限

`viewer`

- 可以看 dashboard、history、影片、notes、connection。
- 不能排訓練，也不能改 preset。

`operator`

- 可以排訓練。
- 可以錄影片、匯出 ONNX、停止 active process。
- 可以改 run name、folder、notes。
- 可以新增或編輯非內建 reward/terrain preset。

`admin`

- 擁有 operator 權限。
- 可以做更多管理類操作，例如成員/清理/高風險設定，視目前 UI 開放功能而定。

如果按鈕是灰的，先看角色，不要先懷疑宇宙。

---

## 3. 選單導覽

`Train`

- 開新訓練用。
- 設定 task、num envs、iterations、device、headless。
- 會帶入目前選到的 reward / terrain snapshot。

`Rewards`

- 團隊共享 reward preset。
- 可以 duplicate 內建 preset 後再編輯。
- 手機版會用較大的輸入列，適合邊看邊調，不適合邊走路邊亂調。

`Terrain`

- 團隊共享 terrain preset。
- 訓練、播放、錄影都應該盡量沿用該 run 訓練時的 terrain 設定。

`History`

- 先看 folders。
- 點 folder 進入 runs。
- 手機上點 run 會直接在該 run 卡片下方展開詳情。
- 可以看 checkpoint、影片、notes、job 狀態和安全遠端操作。

`Connection`

- 看 Supabase、帳號、Mother worker、GPU、storage 是否正常。
- 有問題時先看這頁，它是比較禮貌的儀表板。

`Dashboard`

- 快速摘要：Mother 狀態、queue、recent runs。
- 如果你只想知道「現在能不能丟下一個訓練」，看這裡。

---

## 4. 登入

1. 打開 `RedRHex To Go`。
2. 用團隊提供的登入方式登入。
3. 登入後去 `Connection`。
4. 確認 profile role 正常。

常見問題：

- 看不到資料：可能 profile 沒建好。
- 可以登入但不能按訓練：你可能是 `viewer`。
- 顯示 schema/cache 錯誤：Supabase SQL schema 還沒套最新版，或 PostgREST schema cache 還沒 reload。

---

## 5. 開始訓練

到 `Train`：

1. 選 task。
2. 設定 `num_envs`。
3. 設定 `max_iterations`。
4. 選 device，通常是 `cuda:0`。
5. 保持 headless，除非你真的需要圖形視窗。
6. 確認 reward preset 和 terrain preset。
7. 按 queue training。

Child 會建立一個 job，Mother worker 會認領它，然後在訓練電腦上真正啟動 Isaac。

狀態意思：

- `queued`：已排隊，等 worker。
- `claimed`：worker 已拿到工作。
- `running`：正在跑。
- `completed`：完成。
- `failed`：失敗。
- `cancelled`：取消。

如果 queued 很久：

- Mother worker 可能沒開。
- Mother 可能暫停接受 jobs。
- GPU 可能正在跑其他 play/video/training/export。
- 前面還有 job。
- Connection 頁會告訴你大部分答案。

---

## 6. Reward 使用方式

Reward preset 是團隊共享的設定。

建議流程：

1. 從 `Baseline` 或現有 preset 開始。
2. 如果是內建 preset，先 duplicate。
3. 改名稱，例如：
   - `speed_tracking_jason_v2`
   - `stable_flat_jy_05017`
   - `stairs_safe_test_01`
4. 調整數值。
5. 儲存。
6. 到 Train 使用該 preset。

重點：

- 訓練送出時會帶一份 reward snapshot。
- 後來 preset 再被改，不應該改掉已經送出的訓練設定。
- History 裡可以看 reward comparison，方便知道這次到底和 baseline 差在哪裡。

小提醒：reward 調參很像煮湯，鹽多一點可能剛好，鹽多十倍通常就不是料理了。

---

## 7. Terrain 使用方式

Terrain preset 是訓練地形設定。

建議：

- 平地 policy 就用 flat 類 preset。
- 樓梯/障礙 policy 就清楚命名。
- 不要把 terrain 改成「看起來很酷但沒人知道在測什麼」。

History 會顯示該 run 有多少 terrain overrides。播放與錄影應該盡量用該 run 訓練時的 terrain，而不是目前 UI 選到的 terrain。

---

## 8. History：資料夾與 Run

進入 `History` 時，child 會先顯示 folder layer。

使用方式：

1. 點一個 folder。
2. 看裡面的 runs。
3. 需要回去時按 `Folders` 返回。
4. 手機版點 run 會在原地展開詳情，不需要滑到頁面底部。

Run 卡片會顯示：

- 狀態：running、completed、failed 等。
- checkpoint 狀態。
- video 狀態。
- reward / terrain 摘要。
- 更新時間。

Run details 裡可以：

- 改名稱。
- 改 folder。
- 寫 notes。
- 看 video。
- queue record video。
- queue export ONNX。
- 看 TensorBoard snapshot 或相關連結。

註記請認真寫。未來回頭看 run 的時候，你會感謝過去那個有寫 notes 的自己。

---

## 9. 影片播放

Child 的影片來自 Supabase private storage。Mother worker 會把本機 MP4 上傳，child 再用短效 signed URL 播放。

你可能看到：

`Team Video ready`

- 影片已上傳，可以直接播放。

`uploading to team storage`

- Mother 本機可能已有影片，但還沒同步到 Supabase。
- 等 worker sync，或去 Mother 看 worker 狀態。

`Record Video`

- 有 checkpoint，但還沒有影片。
- 按下後會排一個 video job。

`No checkpoint`

- 還沒有模型 checkpoint，錄影沒有意義。

要注意：錄影也是 Isaac/GPU job，會和 training/play/export 互斥。這不是 child 慢，是我們不想同時開一堆 Isaac 讓電腦進入哲學狀態。

---

## 10. ONNX 匯出

如果 run 有 checkpoint，可以 queue `Export ONNX`。

完成後 History 會顯示 ONNX ready。這通常用於後續部署、推論或交給其他工具。

如果匯出失敗：

- 看 Mother 的 Process Console。
- 檢查 checkpoint 是否存在。
- 檢查環境是否缺套件。

Child 不提供 raw terminal，因為 child 是給團隊安全操作的，不是給大家遠端玩火的。

---

## 11. Notes、名稱與同步

Child 上改 run name、folder、notes 後，會同步到 Supabase，再由 Mother worker 和本機 history 合併。

如果你剛改完但 Mother 還沒立刻看到：

- 等幾秒。
- 確認 worker online。
- 確認 Connection 沒有 stale。
- 避免多個人同時改同一個 run 的同一欄位。

命名建議：

```text
terrain_goal_owner_shortnote
flat_speed_jason_v3
stairs_stable_jy_20env
```

不要只命名 `test`。三天後所有 `test` 都會長得一樣。

---

## 12. Connection 頁怎麼看

`Connection` 是排錯第一站。

重要指標：

`Auth`

- 你有沒有登入。

`Profile`

- 你的 role 是否存在。

`Database`

- Supabase 表格和權限是否可讀。

`Machine`

- Mother 訓練電腦是否在線。

`Worker`

- remote worker 是否接受 jobs。

`GPU`

- 是否有 Isaac/GPU action 正在跑。

`Storage`

- 影片 signed URL 和 artifact 是否能用。

如果 Connection 頁一片紅，先不要連按 Train。那不是訓練，那是祈禱。

---

## 13. 常見問題

### 我按 Train，為什麼一直 queued？

先看 `Connection`：

- Worker stopped：去 Mother Control Center 開 worker。
- Accept jobs paused：Mother 端打開 accepting。
- GPU busy：等目前 job 結束。
- Queue depth 很高：前面有人排了工作。

### 我看不到影片

可能原因：

- run 還沒有 checkpoint。
- 還沒錄影片。
- 影片正在錄。
- 本機影片還沒上傳到 Supabase。
- Supabase storage bucket 或 policy 沒設好。

### 我可以登入，但不能改 reward

你可能是 `viewer`。需要 `operator` 或 `admin`。

### History 沒同步

檢查：

- Worker heartbeat 是否 stale。
- Supabase schema 是否最新版。
- Mother 是否有開。
- 你是否正在編輯欄位，child 會保護未儲存輸入，不會硬刷新蓋掉。

### 我需要看 terminal log

去 Mother。Child 不做 raw terminal，這是故意的安全邊界。

---

## 14. 團隊使用禮儀

- 開訓練前先看 queue 和 GPU 狀態。
- 不要連點同一個 job。
- run name 寫清楚。
- notes 寫出目的、改了什麼、看到什麼。
- 重要 run 放進正確 folder。
- 失敗 run 不一定是壞 run，失敗資訊也很值錢。
- 深度 debug 回 Mother，不要在 child 上硬猜。

---

## 15. 最短操作流程

如果你只想快速完成一次遠端訓練：

1. 打開 RedRHex To Go。
2. 登入。
3. 到 `Connection` 確認 Mother online 且 accepting jobs。
4. 到 `Rewards` 選或建立 reward preset。
5. 到 `Terrain` 選 terrain preset。
6. 到 `Train` 設定 envs / iterations。
7. Queue training。
8. 到 `History` 看 run 狀態。
9. 完成後 Record Video。
10. 看 Team Video，寫 notes，收工。

收工之前記得寫 notes。未來的你會比較少皺眉。
