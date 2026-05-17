# RedRHex Training Panel 更新紀錄

這份 changelog 是給人看的。因為未來的我們應該去訓練機器人，不應該在 commit 裡考古。

格式：

- `Added`：新增功能。
- `Changed`：行為改變。
- `Fixed`：修 bug。
- `Notes`：操作提醒。

## Unreleased / Next

目前沒有。安靜得很可疑，但先享受一下。

## V3.4 First Release

### Added

- Mother 端訓練佇列：
  - Isaac/GPU 正忙時，新訓練會變成 `queued`。
  - GPU 空閒後自動開始下一個 queued run。
  - History card 顯示 `queued`，並提供 `Cancel Queue`。

### Changed

- Isaac/GPU lock 現在包含 training、play、video recording、ONNX export。
- training 正在跑時，Play / Video / ONNX 按鈕會被停用。

### Fixed

- 避免快速連按 Train 時開出重疊 Isaac session，導致 GPU memory 爆掉。
- 修正 failed run 借用旁邊成功 run checkpoint 的誤導顯示。
- 加速 active training 時 `panel_...` 與 `wheg...` 的合併。

## V3.2.1 Terrain Stack

### Added

- Mother 和 Child 的 terrain preset workflow。
- History 的 terrain diff 顯示。
- Play 和 Record Video 會盡量使用 run 當初保存的 terrain 設定。
- Panel 啟動的 playback/video 支援 robot-follow camera。

### Changed

- Reward / terrain preset 選中後直接用於訓練。
- 移除額外的「Use for training」確認流程。
- Child History 改為 folder-first 瀏覽。

### Fixed

- 減少 Mother 啟動的 `panel_...` run 和 RSL-RL 掃描出的 `wheg...` folder 重複顯示。
- Console 標示更清楚：launch command 與 output 分開。
- Console 更重視 Copy Command，而不是只強調 attach command。

## V3.2.0 Remote Refinement

### Added

- Child 端 requester-scoped Discord notification settings。
- Mother 的 Team Activity mission-control 頁面。
- Activity leaderboard、action mix、outcome mix、team pulse、依成員折疊 logs。
- Child dark mode 和 welcome message。

### Changed

- Child menu 改成 phone-first：Train 第一，Dashboard 最後。
- Child History 可以看 reward comparison。
- Child video 顯示更清楚地標示 checkpoint iteration。

### Fixed

- Dark mode menu selection 對比不足。
- Child History 把不同類型資訊塞太近的問題。

## V3.1.x Child Phone UX

### Added

- 手機版 History inline details。
- Child History folder-first navigation。
- 手機和桌面都有返回 folders 的控制。
- 手機版 Rewards 編輯元件放大。
- History card 展開/收合動畫。

### Changed

- Dashboard 簡化成操作狀態，不再塞滿診斷資訊。
- 詳細 health checks 移到 Connection 頁。

### Fixed

- Auto-refresh / 按鈕操作造成 History scroll 跳回頂部。
- Rewards tab 手機寬度異常。
- Refresh 導致 video playback 重置。

## V3.0 Remote Team System

### Added

- `RedRHex To Go` GitHub Pages 靜態 child app。
- Supabase login、roles、jobs、runs、artifacts、events、machine heartbeat。
- Remote worker 指令：

```bash
python -m tools.training_panel.remote_worker
```

- Control Center worker 管理：
  - start
  - stop
  - restart
  - tmux/child mode
  - auto-start
  - accept/pause remote jobs
- Private Supabase Storage + signed URL team video playback。
- Discord/email notification 架構。
- Cloudflare Tunnel live service 支援。

### Changed

- Mother 成為權威 admin/debug/control surface。
- Child 成為簡化、團隊友善的遠端介面。

### Notes

- Machine secrets 只能放訓練電腦。
- Public child page 只能放 Supabase URL 與 anon/publishable key。

## V2.x Remote Foundations

### Added

- Supabase schema：profiles、machines、runs、jobs、events、artifacts、proxy sessions、notifications。
- Worker heartbeat 和 job claim flow。
- Remote role model：
  - viewer
  - operator
  - admin
- Artifact sync 基礎架構。

### Fixed

- Supabase setup diagnostics 更清楚。
- 缺少 environment variables 時錯誤訊息更明確。
- 修正 Supabase `/rest/v1` URL 重複造成 404 的問題。

## V1.1 Local Power Tools

### Added

- History 的 `Export ONNX`。
- `Compact Run`：保留最新 checkpoint，刪除舊 top-level `model_*.pt`。
- History 顯示 ONNX metadata。
- Compact preview 和 exact run-id confirmation。

### Changed

- History Actions panel 排版更清楚。

## V1.0 Local Training Panel

### Added

- 本機 Mother panel。
- Train form。
- History list。
- Process Console。
- TensorBoard launch。
- Play checkpoint。
- Record Video。
- Notes / folders。
- Reward preset workflow。
- Version 和團隊標示。

### Notes

- 從這版開始，panel 已經不只是 terminal commands 的外殼，而是真的可以日常使用。

## 未來更新紀錄規則

新增功能時：

1. 在這裡加簡短 entry。
2. 描述使用者會感覺到的行為，不要只寫檔名。
3. 操作上的坑放在 `Notes`。
4. 保持精簡。未來的我們還要顧訓練，不要把 changelog 寫成論文。
