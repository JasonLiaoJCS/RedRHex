# RedRhex 四階段訓練（Stage1~Stage4）完整說明

> 這份文件是「對外說明版」：
> 1) 你從單階段混合訓練改成四階段做了哪些改變
> 2) 每個階段負責什麼（特別是 Stage4 整合）
> 3) 所有 terminal 指令怎麼一條條打，如何把 checkpoint 串起來
> 4) 評估程式怎麼用、怎麼看結果
> 5) 最後如何播放模型

---

## 1. 從「一次訓練多技能」改成「四階段訓練」，你做了哪些關鍵改變

### 1.1 訓練流程改成 Curriculum（課程式）
原本：一開始就把 forward/lateral/diagonal/yaw 全混在一起訓練。  
現在：分 4 個階段，先拆技能再整合。

- Stage1：只學 forward
- Stage2：只學 lateral
- Stage3：只學 yaw
- Stage4：把前 3 個技能合併進同一個 policy 再潤化

### 1.2 不是 4 個互不相干模型，而是「接續訓練」
每個 stage 都用上一個 stage 的 checkpoint 當起點（`--resume --load_run --checkpoint`），不是重頭開始。

### 1.3 命令條件化 + 模式硬限制（action gating）
你在環境裡做了 mode-conditioned 控制規則，不是只靠 reward 軟引導。

- Forward mode：ABAD 強制鎖住
- Lateral mode：主驅動鎖住或 soft-lock
- Diagonal/Yaw：主驅動 + ABAD 都可動
- Lateral 進入前要經過 GO_TO_STAND（站姿就位）再進 LATERAL_STEP

### 1.4 加了 multi-skill reward 結構（含 forward gait prior / yaw 專項）
你目前簡化 reward 模式是「核心項 + 模式專項」，並含：

- 命令追蹤（vx/vy/wz）
- mode specialization（不同 mode 有不同 shaping）
- forward gait prior（tripod coherence、anti-phase、duty、slow/fast ratio、transition overlap）
- yaw 穩定專項（yaw tracking、roll/pitch penalty、height penalty、slip penalty、cheat penalty）
- lateral soft-lock penalty

### 1.5 加了驗收評估程式（command sweep）
你不是只看 reward 曲線，而是固定命令測試 + pass/fail 指標，能直接看技能是否真的可用。

---

## 2. 四個階段各自負責什麼

## Stage1：Forward-only（打底）
目標：先把「可穩定前進」學紮實。

- 命令只給前進（vx>0, vy=0, wz=0）
- ABAD 在 forward mode 被鎖住
- 讓主驅動先學出穩定前進與基本 gait 節奏

## Stage2：Lateral-only（學側移）
目標：獨立學會側移，而不是被 forward 習慣壓掉。

- 命令只給 lateral（vy!=0）
- 進 lateral 前先 GO_TO_STAND
- 主驅動鎖住或 soft-lock，重點放在 ABAD 側移能力

## Stage3：Yaw-only（學原地旋轉）
目標：獨立學出 yaw，不靠「亂晃」假裝有轉。

- 命令只給 yaw（wz!=0）
- 允許主驅動反轉（differential-style）
- 透過 yaw 專項 reward 壓制翻滾/抬機身作弊

## Stage4：Mixed-skills（整合 + 潤化）
這一段最重要，重點不是「暴力拼接」，而是「從 Stage3 權重起步，在混合命令下繼續 fine-tune」。

Stage4 真正在做的事：

1. 載入 Stage3 checkpoint 當初始 policy  
2. 命令分佈切成 mixed（forward/lateral/diagonal/yaw）  
3. 仍維持 mode gating 硬限制（不是全部全開無規則）  
4. 在同一個 reward 框架下，跨模式共同優化（整合與抑制衝突）  
5. 透過持續 PPO 更新，把「單技能能力」磨成「可切換多技能能力」

所以 Stage4 是「整合與潤化訓練」，不是把前三階段模型直接拿來不再學。

---

## 3. 從零到完成的 terminal 指令（完整串接）

## 3.1 前置

```bash
cd ~/RedRhex/RedRhex
conda activate env_isaaclab
```

---

## 3.2 Stage1

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage1 \
  env.stage=1 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

抓 Stage1 最新 run/checkpoint：

```bash
RUN1=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage1* | head -1)")
CKPT1=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN1/model_*.pt | tail -1)")

echo "RUN1=$RUN1"
echo "CKPT1=$CKPT1"
```

---

## 3.3 Stage2（接續 Stage1）

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage2 \
  --resume \
  --load_run="$RUN1" \
  --checkpoint="$CKPT1" \
  env.stage=2 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

```bash
RUN2=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage2* | head -1)")
CKPT2=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN2/model_*.pt | tail -1)")

echo "RUN2=$RUN2"
echo "CKPT2=$CKPT2"
```

---

## 3.4 Stage3（接續 Stage2）

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage3 \
  --resume \
  --load_run="$RUN2" \
  --checkpoint="$CKPT2" \
  env.stage=3 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

```bash
RUN3=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage3* | head -1)")
CKPT3=$(basename "$(ls -v logs/rsl_rl/redrhex_wheg/$RUN3/model_*.pt | tail -1)")

echo "RUN3=$RUN3"
echo "CKPT3=$CKPT3"
```

---

## 3.5 Stage4（接續 Stage3，做整合）

```bash
python scripts/rsl_rl/train.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=stage4 \
  --resume \
  --load_run="$RUN3" \
  --checkpoint="$CKPT3" \
  env.stage=4 \
  env.draw_debug_vis=False \
  env.dr_try_physical_material_randomization=False
```

抓 Stage4 最終 checkpoint：

```bash
RUN4=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage4* | head -1)")
FINAL_CKPT=$(ls -v logs/rsl_rl/redrhex_wheg/$RUN4/model_*.pt | tail -1)

echo "RUN4=$RUN4"
echo "FINAL_CKPT=$FINAL_CKPT"
```

### 補充：`max_iterations` 在 resume 時的意義
如果你載入的是 `model_14997.pt`，再給 `--max_iterations=5000`，通常會繼續跑到約 `19997`，不是只跑 5000 絕對 index。

---

## 4. 如何保證下一階段真的接到上一階段

每次進下一階段前，都先做：

```bash
echo "$RUN3"
echo "$CKPT3"
test -n "$RUN3" && test -n "$CKPT3" || { echo "RUN3/CKPT3 取得失敗"; exit 1; }
```

如果不印 `echo` 也可以訓練，但你看不到是不是抓錯檔案。

---

## 5. 評估程式（eval_command_sweep.py）怎麼用

## 5.1 基本用法

```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=256 \
  --checkpoint="$FINAL_CKPT" \
  --warmup_steps=120 \
  --sweep_steps=600 \
  --accept_duration_s=2.0 \
  --accept_vy_abs=0.15 \
  --accept_wz_abs=0.40 \
  --accept_yaw_tilt_bound=0.60 \
  --accept_diag_sign_ratio=0.70
```

建議加 CSV 輸出：

```bash
python scripts/rsl_rl/eval_command_sweep.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --num_envs=256 \
  --checkpoint="$FINAL_CKPT" \
  --warmup_steps=120 \
  --sweep_steps=600 \
  --accept_duration_s=2.0 \
  --accept_vy_abs=0.15 \
  --accept_wz_abs=0.40 \
  --accept_yaw_tilt_bound=0.60 \
  --accept_diag_sign_ratio=0.70 \
  --csv logs/rsl_rl/redrhex_wheg/${RUN4}/eval_command_sweep.csv
```

## 5.2 這支程式在做什麼
它會固定掃 7 種命令（forward/left/right/diag_left/diag_right/yaw_ccw/yaw_cw），逐一測試並輸出：

- `Command Tracking (MAE)`：速度追蹤誤差
- `Skill Acceptance`：每個命令的 PASS/FAIL
- `Acceptance Metrics Summary`：整體追蹤、步態、穩定、接觸、能耗統計

## 5.3 如何判讀

看這三塊就夠：

1. `Skill Acceptance`  
- `left/right` 看 `vy_success_s` 是否 >= `accept_duration_s`  
- `yaw_ccw/yaw_cw` 看 `wz_success_s` 是否 >= `accept_duration_s` 且沒高跌倒率  
- `diag` 看 `diag_sign_match` 是否 >= `accept_diag_sign_ratio`

2. `stability`  
- `fall_rate` 越低越好（理想接近 0）
- `roll_rms`、`pitch_rms` 過高代表在翻滾/抬機身

3. `forward gait`  
- `phase_diff_abs_to_pi` 越小越好
- `stance_fraction_abs_err_to_0.65` 越小越好
- `swing_to_stance_speed_ratio` 應明顯大於 1（slow stance / fast swing）

## 5.4 如果 final checkpoint 不好
不要只看最後一顆。請掃描 stage4 的中後段 checkpoint，挑最好的再 play：

```bash
RUN4=$(basename "$(ls -td logs/rsl_rl/redrhex_wheg/*stage4* | head -1)")
for ck in \
  logs/rsl_rl/redrhex_wheg/$RUN4/model_15000.pt \
  logs/rsl_rl/redrhex_wheg/$RUN4/model_17000.pt \
  logs/rsl_rl/redrhex_wheg/$RUN4/model_19000.pt \
  logs/rsl_rl/redrhex_wheg/$RUN4/model_19996.pt
 do
  echo "===== $ck ====="
  python scripts/rsl_rl/eval_command_sweep.py \
    --task=Template-Redrhex-Direct-v0 \
    --headless \
    --num_envs=256 \
    --checkpoint="$ck" \
    --warmup_steps=120 \
    --sweep_steps=600 \
    --accept_duration_s=2.0 \
    --accept_vy_abs=0.15 \
    --accept_wz_abs=0.40 \
    --accept_yaw_tilt_bound=0.60 \
    --accept_diag_sign_ratio=0.70
 done
```

---

## 6. 最終播放（Play）

## 6.1 GUI 即時觀察

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-Direct-v0 \
  --num_envs=64 \
  --checkpoint="$FINAL_CKPT"
```

## 6.2 Headless + 錄影

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Redrhex-Direct-v0 \
  --headless \
  --video \
  --video_length=1000 \
  --num_envs=64 \
  --checkpoint="$FINAL_CKPT"
```

## 6.3 `model_*.pt` / `policy.pt` 在哪

訓練 checkpoint：

```text
logs/rsl_rl/redrhex_wheg/<run>/model_*.pt
```

Play 匯出（JIT/ONNX）：

```text
logs/rsl_rl/redrhex_wheg/<run>/exported/policy.pt
logs/rsl_rl/redrhex_wheg/<run>/exported/policy.onnx
```

---

## 7. 常見坑位（你已經踩過）

- `stage=1` 會錯，必須是 `env.stage=1`
- 沒有 `--resume --load_run --checkpoint` 就不是接續訓練
- `FINAL_CKPT` 空字串時，eval 可能回退去抓到錯的舊模型
- checkpoint 架構不相容（例如舊 256-128-64 vs 新 128-64）會 `size mismatch`
- `omni.platforminfo` CPU 警告通常不是主因，可先忽略

---

## 8. 一句話對外總結

你把原本「一次混合訓練所有技能」改成「Stage1~4 課程式訓練」：先讓 policy 依序學會 forward、lateral、yaw，再在 Stage4 用混合命令與模式約束持續 fine-tune，把前三階段能力整合到同一個 policy，並用 command sweep 做可量化驗收，最後再用 play 實機觀察。
