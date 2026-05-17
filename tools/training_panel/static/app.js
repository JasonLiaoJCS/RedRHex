const DEBUG_POLL_MS = 1500;
const RUNS_POLL_ACTIVE_MS = 10000;
const RUNS_POLL_IDLE_MS = 30000;

const state = {
  selectedRun: null,
  runs: [],
  activeProcessMap: {},
  activeProcesses: [],
  activeProcessesByRun: {},
  activeProcessByKind: {},
  debugTarget: null,
  lastDebug: null,
  debugTimer: null,
  runsRefreshTimer: null,
  renameDirty: false,
  renameDraftRunId: null,
  // Search / filter / sort (Module 5)
  searchQuery: "",
  statusFilter: "",
  sortKey: "newest",
  // Folders (Module 3)
  activeFolder: "",
  folders: [],
  selectedRunIds: new Set(),
  isBulkDeleting: false,
  pendingDeleteRunIds: new Set(),
  notifications: {
    initialized: false,
    knownRunIds: new Set(),
    unreadRunIds: new Set(),
  },
  // Rewards / presets
  presets: [],
  activePresetId: "baseline",
  activePresetOverrides: {},
  selectedPresetId: null,
  rewardDraftPreset: null,
  rewardDefaults: {},
  rewardCompareMode: "default",
  // Terrain / presets
  terrainPresets: [],
  activeTerrainPresetId: "baseline",
  activeTerrainPresetOverrides: {},
  selectedTerrainPresetId: null,
  terrainDefaults: {},
  terrainSchema: [],
  remoteStatus: null,
  activityEvents: [],
  activityAnalytics: null,
  activityFilters: {
    window: "7d",
    member: "",
    category: "",
  },
  activityCollapsedGroups: new Set(),
  // Comparison (Module 6)
  comparisonRun: null,
  comparisonMode: false,
};

const $ = (selector) => document.querySelector(selector);
const THEME_KEY = "redrhex-training-panel-theme";
const NOTIFICATIONS_KEY = "redrhex-training-panel-notifications-v1";

function preferredTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("#theme-toggle").textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
}

function loadNotificationState() {
  try {
    const raw = localStorage.getItem(NOTIFICATIONS_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    state.notifications.knownRunIds = new Set(parsed.knownRunIds || []);
    state.notifications.unreadRunIds = new Set(parsed.unreadRunIds || []);
  } catch {
    state.notifications.knownRunIds = new Set();
    state.notifications.unreadRunIds = new Set();
  }
}

function saveNotificationState() {
  try {
    localStorage.setItem(
      NOTIFICATIONS_KEY,
      JSON.stringify({
        knownRunIds: [...state.notifications.knownRunIds],
        unreadRunIds: [...state.notifications.unreadRunIds],
      })
    );
  } catch {
    // Notification badges are a convenience; storage failures should not block the panel.
  }
}

function renderNotificationBadges() {
  const count = state.notifications.unreadRunIds.size;
  const historyButton = document.querySelector('.nav-button[data-view="history"]');
  if (!historyButton) return;
  historyButton.classList.toggle("has-notification", count > 0);
  historyButton.dataset.notificationCount = String(count);
}

function markHistoryUnread(runId) {
  if (!runId) return;
  state.notifications.knownRunIds.add(runId);
  state.notifications.unreadRunIds.add(runId);
  saveNotificationState();
  renderNotificationBadges();
}

function markHistoryRead(runId) {
  if (!runId || !state.notifications.unreadRunIds.has(runId)) return;
  state.notifications.unreadRunIds.delete(runId);
  saveNotificationState();
  renderNotificationBadges();
}

function reconcileHistoryNotifications(runs) {
  const ids = new Set(runs.map((run) => run.id));
  if (!state.notifications.initialized) {
    if (state.notifications.knownRunIds.size === 0) {
      state.notifications.knownRunIds = new Set(ids);
    } else {
      for (const id of ids) {
        if (!state.notifications.knownRunIds.has(id)) state.notifications.unreadRunIds.add(id);
      }
    }
    state.notifications.initialized = true;
  } else {
    for (const id of ids) {
      if (!state.notifications.knownRunIds.has(id)) state.notifications.unreadRunIds.add(id);
    }
  }
  state.notifications.knownRunIds = new Set([...state.notifications.knownRunIds, ...ids]);
  state.notifications.unreadRunIds = new Set([...state.notifications.unreadRunIds].filter((id) => ids.has(id)));
  saveNotificationState();
  renderNotificationBadges();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || response.statusText);
    error.data = data;
    throw error;
  }
  return data;
}

function setStatus(message, linkUrl = "") {
  const status = $("#notes-status");
  status.textContent = "";
  status.append(document.createTextNode(message));
  if (linkUrl) {
    const link = document.createElement("a");
    link.href = linkUrl;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = linkUrl;
    status.append(document.createTextNode(" "));
    status.append(link);
  }
}

function setTerrainStatus(message) {
  const status = $("#terrain-status");
  if (status) status.textContent = message;
}

function setView(name) {
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === name);
  });
  const titles = {
    train: ["Train", "Start a controlled RSL-RL run with the repo defaults."],
    rewards: ["Rewards", "Tune reward weights with presets and see which settings each run used."],
    terrain: ["Terrain", "Tune terrain generator, curriculum, and sub-terrain mix with presets."],
    history: ["History", "Review runs, notes, checkpoints, TensorBoard, and playbacks."],
    convergence: ["Convergence", "Define reward plateau detection and automatic result-video behavior."],
    activity: ["Activity", "See team run requests, panel actions, and lightweight usage analytics."],
    access: ["Control Center", "Manage local access, V3.0 remote worker status, and remote launch acceptance."],
  };
  $("#view-title").textContent = titles[name][0];
  $("#view-subtitle").textContent = titles[name][1];
}

function rewardPresetsForRender() {
  if (!state.rewardDraftPreset) return state.presets;
  const withoutDraft = state.presets.filter((preset) => preset.id !== state.rewardDraftPreset.id);
  return [state.rewardDraftPreset, ...withoutDraft];
}

function rewardPresetById(presetId) {
  if (state.rewardDraftPreset && state.rewardDraftPreset.id === presetId) return state.rewardDraftPreset;
  return state.presets.find((preset) => preset.id === presetId);
}

function currentRewardEditorValues() {
  const values = {};
  document.querySelectorAll("#reward-categories .reward-row-input").forEach((input) => {
    const key = input.dataset.key;
    const val = parseFloat(input.value);
    if (key && !Number.isNaN(val)) values[key] = val;
  });
  return values;
}

function rewardPresetIdForTraining() {
  if (state.selectedPresetId && rewardPresetById(state.selectedPresetId)) return state.selectedPresetId;
  return state.activePresetId || "baseline";
}

function rewardOverridesForTraining() {
  const presetId = rewardPresetIdForTraining();
  const preset = rewardPresetById(presetId);
  if (state.selectedPresetId === presetId && $("#reward-categories")?.children.length) {
    const values = currentRewardEditorValues();
    if (Object.keys(values).length) {
      if (state.rewardDraftPreset && presetId === state.rewardDraftPreset.id) {
        state.rewardDraftPreset.values = values;
      }
      return values;
    }
  }
  return preset?.values || state.activePresetOverrides || {};
}

function currentTerrainEditorValues() {
  const values = {};
  document.querySelectorAll("#terrain-categories .terrain-row-input").forEach((input) => {
    const key = input.dataset.key;
    if (!key) return;
    values[key] = parseTerrainInput(input, terrainMeta(key));
  });
  return values;
}

function terrainPresetIdForTraining() {
  if (state.selectedTerrainPresetId && state.terrainPresets.some((preset) => preset.id === state.selectedTerrainPresetId)) {
    return state.selectedTerrainPresetId;
  }
  return state.activeTerrainPresetId || "baseline";
}

function terrainOverridesForTraining() {
  const presetId = terrainPresetIdForTraining();
  const preset = state.terrainPresets.find((item) => item.id === presetId);
  if (state.selectedTerrainPresetId === presetId && $("#terrain-categories")?.children.length) {
    const values = currentTerrainEditorValues();
    if (Object.keys(values).length) return values;
  }
  return preset?.values || state.activeTerrainPresetOverrides || {};
}

function updateTrainingPresetIndicators() {
  const rewardId = rewardPresetIdForTraining();
  const rewardPreset = rewardPresetById(rewardId) || { name: rewardId };
  const rewardEl = $("#train-active-preset-name");
  if (rewardEl) rewardEl.textContent = rewardPreset.name || rewardId;

  const terrainId = terrainPresetIdForTraining();
  const terrainPreset = state.terrainPresets.find((preset) => preset.id === terrainId) || { name: terrainId };
  const terrainEl = $("#train-active-terrain-preset-name");
  if (terrainEl) terrainEl.textContent = terrainPreset.name || terrainId;
}

function formData(form) {
  const data = Object.fromEntries(new FormData(form).entries());
  data.display_name = String(data.display_name || "").trim();
  data.headless = form.elements.headless.checked;
  data.resume = Boolean(data.checkpoint);
  data.num_envs = Number(data.num_envs);
  data.max_iterations = Number(data.max_iterations);
  data.reward_preset_id = rewardPresetIdForTraining();
  data.reward_overrides = rewardOverridesForTraining();
  data.terrain_preset_id = terrainPresetIdForTraining();
  data.terrain_overrides = terrainOverridesForTraining();
  if (state.rewardDraftPreset?.source_run_id && data.reward_preset_id === state.rewardDraftPreset.id) {
    data.tweak_source_run_id = state.rewardDraftPreset.source_run_id;
    data.tweak_source_label = state.rewardDraftPreset.source_label || state.rewardDraftPreset.source_run_id;
  }
  return data;
}

function clearTrainingRunName(form = $("#train-form")) {
  const input = form?.querySelector?.('input[name="display_name"]');
  if (input) input.value = "";
}

async function loadSystem() {
  const system = await api("/api/system");
  $("#system-info").textContent = JSON.stringify(system, null, 2);
}

function renderKvGrid(selector, rows) {
  const node = $(selector);
  if (!node) return;
  node.innerHTML = rows
    .map(([key, value]) => `<span class="info-key">${escapeHtml(key)}</span><span class="info-val">${escapeHtml(String(value))}</span>`)
    .join("");
}

function remoteStatusPill(label, value, className, detail = "") {
  return `
    <div class="control-status-card ${className}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      ${detail ? `<small>${escapeHtml(detail)}</small>` : ""}
    </div>
  `;
}

async function loadRemoteStatus() {
  const status = await api("/api/remote/status");
  state.remoteStatus = status;
  const badge = $("#remote-config-badge");
  if (badge) {
    badge.textContent = status.configured ? "Configured" : "Needs Setup";
    badge.className = status.configured ? "status-badge status-completed" : "status-badge status-interrupted";
  }

  const strip = $("#remote-status-strip");
  if (strip) {
    strip.innerHTML = [
      remoteStatusPill("Setup", status.configured ? "Configured" : "Needs Setup", status.configured ? "status-completed" : "status-interrupted", status.env_file_exists ? "env file found" : "env file missing"),
      remoteStatusPill("Worker", status.worker_running ? "Running" : "Stopped", status.worker_running ? "status-running" : "muted-pill", status.worker_runtime_mode || status.worker_mode || "tmux"),
      remoteStatusPill("Remote Control", status.accept_jobs ? "Enabled" : "Paused", status.accept_jobs ? "status-completed" : "status-interrupted", status.remote_web_url || "RedRHex To Go"),
      remoteStatusPill("Isaac/GPU", status.active_isaac_process_count ? "Busy" : "Free", status.active_isaac_process_count ? "status-running" : "status-completed", `${status.active_process_count || 0} active process${Number(status.active_process_count || 0) === 1 ? "" : "es"}`),
    ].join("");
  }

  const workerBadge = $("#remote-worker-badge");
  if (workerBadge) {
    workerBadge.textContent = status.worker_running ? "Running" : "Stopped";
    workerBadge.className = status.worker_running ? "status-badge status-running" : "status-badge muted-pill";
  }
  const workerSummary = $("#remote-worker-summary");
  if (workerSummary) {
    const mode = status.worker_runtime_mode || status.worker_mode || "tmux";
    if (!status.configured) {
      workerSummary.textContent = "Set REDRHEX_SUPABASE_URL and REDRHEX_SUPABASE_MACHINE_TOKEN in your .env to enable remote access.";
    } else if (status.worker_running && status.accept_jobs) {
      workerSummary.textContent = `Connected in ${mode} mode — teammates can control training from RedRHex Go.`;
    } else if (status.worker_running && !status.accept_jobs) {
      workerSummary.textContent = `Connected in ${mode} mode but remote control is paused. Toggle "Allow remote training & control" to resume.`;
    } else {
      workerSummary.textContent = "Not connected. Click Connect to let teammates use RedRHex Go.";
    }
  }
  document.querySelectorAll(".segment-button[data-mode]").forEach((button) => {
    const active = button.dataset.mode === status.worker_mode;
    button.classList.toggle("active", active);
    button.disabled = false;
  });
  const autostart = $("#remote-autostart");
  if (autostart) autostart.checked = Boolean(status.worker_autostart);
  const startButton = $("#remote-worker-start");
  if (startButton) startButton.disabled = status.worker_running || !status.configured;
  const stopButton = $("#remote-worker-stop");
  if (stopButton) stopButton.disabled = !status.worker_running;
  const restartButton = $("#remote-worker-restart");
  if (restartButton) restartButton.disabled = !status.configured;
  const acceptToggle = $("#remote-accept-toggle");
  if (acceptToggle) acceptToggle.checked = Boolean(status.accept_jobs);

  const modeNote = $("#remote-mode-note");
  if (modeNote) {
    const MODE_NOTES = {
      tmux:  "tmux mode: the worker persists in a detached session — it keeps running even if you close this browser tab or restart the panel.",
      child: "Child mode: the worker stops when you close the training panel. Good for quick testing.",
    };
    const currentMode = status.worker_mode || "tmux";
    modeNote.textContent = status.worker_running
      ? `Mode changes are saved but apply on the next restart. ${MODE_NOTES[currentMode] || ""}`
      : MODE_NOTES[currentMode] || "";
  }
  renderKvGrid("#remote-worker-grid", [
    ["Saved Mode", status.worker_mode || "tmux"],
    ["Runtime Mode", status.worker_runtime_mode || "-"],
    ["Auto-start", status.worker_autostart ? "enabled" : "disabled"],
    ["PID", status.worker_pid || "-"],
    ["tmux Session", status.worker_tmux_session || "-"],
    ["Log File", status.worker_log_file || "-"],
    ["Last Error", status.worker_last_error || "-"],
  ]);
  const attachWrap = $("#remote-worker-attach-wrap");
  const attach = $("#remote-worker-attach");
  if (attachWrap && attach) {
    attachWrap.hidden = !status.worker_attach_command;
    attach.textContent = status.worker_attach_command || "";
  }
  const output = $("#remote-worker-output");
  if (output) output.textContent = status.worker_output_tail || "No worker output yet.";

  const setup = $("#remote-setup-list");
  if (setup) {
    setup.innerHTML = (status.setup_checks || [])
      .map(
        (check) => `
          <div class="setup-row ${check.ok ? "ok" : "missing"}">
            <span>${check.ok ? "OK" : "Missing"}</span>
            <strong>${escapeHtml(check.label)}</strong>
            <small>${escapeHtml(check.detail || "")}</small>
          </div>
        `
      )
      .join("");
  }
  const envPath = $("#remote-env-path");
  if (envPath) envPath.textContent = status.env_file_path || "~/.redrhex_remote.env";

  renderKvGrid("#remote-access-grid", [
    ["Phone Page", status.remote_web_url || "-"],
    ["Machine ID", status.machine_id || "-"],
    ["Supabase", status.configured ? status.supabase_url : "not configured"],
    ["Cloudflare", status.cloudflare_tunnel_host || "not configured"],
  ]);
  const phoneUrl = $("#remote-phone-url");
  if (phoneUrl) phoneUrl.textContent = status.remote_web_url || "";
  const tunnelCommand = $("#remote-tunnel-command");
  if (tunnelCommand) tunnelCommand.textContent = status.cloudflare_tunnel_command;
  renderKvGrid("#remote-integrations-grid", [
    ["Panel Version", status.version || "-"],
    ["Active Processes", status.active_process_count || 0],
    ["Isaac/GPU Lock", status.active_isaac_process_count ? "busy" : "free"],
    ["Discord", status.discord_configured ? "configured" : "not configured"],
  ]);
  const raw = $("#remote-status-raw");
  if (raw) raw.textContent = JSON.stringify(status, null, 2);
}

async function saveRemoteAcceptance(acceptJobs) {
  const data = await api("/api/remote/settings", {
    method: "POST",
    body: JSON.stringify({ accept_jobs: acceptJobs }),
  });
  await loadRemoteStatus();
  setStatus(data.status.accept_jobs ? "Remote queued jobs enabled." : "Remote queued jobs disabled.");
}

async function saveRemoteSettings(updates) {
  const data = await api("/api/remote/settings", {
    method: "POST",
    body: JSON.stringify(updates),
  });
  await loadRemoteStatus();
  return data;
}

async function setRemoteWorkerMode(mode) {
  await saveRemoteSettings({ worker_mode: mode });
  setStatus(`Remote worker mode saved: ${mode}. Restart worker to apply if it is running.`);
}

async function setRemoteAutostart(enabled) {
  await saveRemoteSettings({ worker_autostart: enabled });
  setStatus(enabled ? "Remote worker auto-start enabled." : "Remote worker auto-start disabled.");
}

async function startRemoteWorker() {
  const mode = state.remoteStatus?.worker_mode || "tmux";
  await api("/api/remote/worker/start", {
    method: "POST",
    body: JSON.stringify({ mode }),
  });
  await loadRemoteStatus();
  setStatus("Remote worker started.");
}

async function stopRemoteWorker() {
  await api("/api/remote/worker/stop", { method: "POST", body: JSON.stringify({}) });
  await loadRemoteStatus();
  setStatus("Remote worker stop requested.");
}

async function restartRemoteWorker() {
  const mode = state.remoteStatus?.worker_mode || "tmux";
  await api("/api/remote/worker/restart", {
    method: "POST",
    body: JSON.stringify({ mode }),
  });
  await loadRemoteStatus();
  setStatus("Remote worker restarted.");
}

async function copyWorkerAttach() {
  const command = state.remoteStatus?.worker_attach_command || "";
  await copyText(command);
  setStatus("Worker attach command copied.");
}

async function copyWorkerOutput() {
  const output = state.remoteStatus?.worker_output_tail || "";
  await copyText(output);
  setStatus("Worker output copied.");
}

async function copyRemoteEnvPath() {
  await copyText(state.remoteStatus?.env_file_path || "~/.redrhex_remote.env");
  setStatus("Remote env file path copied.");
}

async function copyRemotePhoneUrl() {
  await copyText(state.remoteStatus?.remote_web_url || "");
  setStatus("Phone page URL copied.");
}

async function loadTweaks() {
  const data = await api("/api/tweakables");
  $("#tweak-files").innerHTML = data.files
    .map(
      (file) => `
        <article class="card">
          <strong>${escapeHtml(file.title)}</strong>
          <small>${escapeHtml(file.why)}</small>
          <small>${escapeHtml(file.absolute_path)}</small>
          <span class="pill">${file.exists ? "found" : "missing"}</span>
        </article>
      `
    )
    .join("");
  $("#reward-scales").innerHTML = data.reward_scales
    .map(
      (scale) => `
        <div class="scale-row">
          <div><strong>${escapeHtml(scale.name)}</strong><small>${escapeHtml(scale.relative_path)}:${escapeHtml(scale.line)}</small></div>
          <code>${escapeHtml(scale.value)}</code>
          <small>${escapeHtml(scale.comment || "No inline note yet.")}</small>
        </div>
      `
    )
    .join("");
}

function findRun(runId) {
  return state.runs.find((run) => run.id === runId);
}

function runButtonDisabled(disabled) {
  return disabled ? "disabled" : "";
}

function formatRelativeTime(iso) {
  if (!iso) return "";
  const timestamp = Date.parse(iso);
  if (Number.isNaN(timestamp)) return iso;
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function checkpointIteration(path) {
  const match = String(path || "").match(/model_(\d+)\.pt$/);
  return match ? Number(match[1]) : null;
}

function formatDuration(createdAt, updatedAt) {
  const start = Date.parse(createdAt || "");
  const end = Date.parse(updatedAt || "");
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) return "";
  const seconds = Math.floor((end - start) / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainderSeconds = seconds % 60;
  if (minutes < 60) return `${minutes}m ${remainderSeconds}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

function statusClass(status) {
  if (window.RedRhexStatus?.className) {
    return window.RedRhexStatus.className("run", status);
  }
  const normalized = String(status || "unknown").toLowerCase();
  if (normalized === "completed") return "status-completed";
  if (normalized === "queued") return "status-queued";
  if (normalized === "running" || normalized === "stopping") return "status-running";
  if (normalized === "failed") return "status-failed";
  if (normalized === "interrupted" || normalized === "cancelled") return "status-interrupted";
  return "status-unknown";
}

function statusLabel(kind, status, context) {
  if (window.RedRhexStatus?.label) {
    return window.RedRhexStatus.label(kind, status, context);
  }
  return String(status || "unknown").toLowerCase() || "unknown";
}

function runParamSummary(run) {
  if (!run.params) return "";
  const parts = [];
  if (run.params.task) parts.push(`task: ${run.params.task}`);
  if (run.params.num_envs !== undefined) parts.push(`envs: ${run.params.num_envs}`);
  if (run.params.max_iterations !== undefined) parts.push(`iters: ${run.params.max_iterations}`);
  return parts.join(" · ");
}

function runTimeSummary(run) {
  const relative = formatRelativeTime(run.created_at);
  const duration = formatDuration(run.created_at, run.updated_at);
  if (relative && duration) return `${relative} · duration ${duration}`;
  return relative || duration || "";
}

function checkpointSummary(run) {
  if (!run.latest_checkpoint) return "no checkpoint";
  const iteration = checkpointIteration(run.latest_checkpoint);
  return iteration === null ? "checkpoint" : `checkpoint at iter ${iteration}`;
}

function videoSummary(run) {
  if (run.latest_video) return "video ready";
  if (run.video_status === "recording") return "recording video";
  if (run.video_status === "failed") return "video failed";
  if (run.video_status === "missing_checkpoint") return "video waiting for checkpoint";
  return "";
}

function onnxSummary(run) {
  if (run.onnx_path) return "ONNX ready";
  if (run.onnx_status === "exporting") return "exporting ONNX";
  if (run.onnx_status === "failed") return "ONNX failed";
  return "";
}

function runLogSummary(run) {
  return run.log_dir ? "training log saved" : "no training log";
}

function runStatusDetail(run) {
  if (run.status === "failed" && run.returncode !== undefined && run.returncode !== null) {
    return ` · exit ${run.returncode}`;
  }
  return "";
}

function rememberActiveProcess(key, process) {
  if (!key) return;
  if (!state.activeProcessesByRun[key]) state.activeProcessesByRun[key] = [];
  state.activeProcessesByRun[key].push(process);
  if (!state.activeProcessByKind[key]) state.activeProcessByKind[key] = {};
  if (process.kind && !state.activeProcessByKind[key][process.kind]) {
    state.activeProcessByKind[key][process.kind] = process.run_id;
  }
  if (!state.activeProcessMap[key]) {
    state.activeProcessMap[key] = process.run_id;
  }
}

function activeProcessIdForRun(runId, kind = "") {
  if (!runId) return "";
  if (kind) return state.activeProcessByKind[runId]?.[kind] || "";
  return state.activeProcessMap[runId] || "";
}

function activeProcessForRun(runId, kind = "") {
  if (!runId) return null;
  const processes = state.activeProcessesByRun[runId] || [];
  return processes.find((process) => !kind || process.kind === kind) || null;
}

function activeMediaProcess() {
  return state.activeProcesses.find((process) => ["play", "video", "onnx"].includes(process.kind)) || null;
}

function activeGpuProcess() {
  return state.activeProcesses.find((process) => ["training", "play", "video", "onnx"].includes(process.kind)) || null;
}

function mediaLockMessage(process) {
  if (!process) return "";
  if (process.kind === "training") return "Training is running. New training requests will be queued until the GPU is free.";
  if (process.kind === "video") return "Video recording is running. Stop recording before starting another Isaac action.";
  if (process.kind === "onnx") return "ONNX export is running. Stop it before starting playback or recording.";
  return "Playback is running. Stop Play before starting another Isaac action.";
}

function consoleTargetForRun(runId) {
  const processId =
    activeProcessIdForRun(runId, "play") ||
    activeProcessIdForRun(runId, "video") ||
    activeProcessIdForRun(runId, "onnx") ||
    activeProcessIdForRun(runId, "training") ||
    activeProcessIdForRun(runId, "tensorboard") ||
    activeProcessIdForRun(runId);
  return processId ? { type: "process", id: processId } : { type: "run", id: runId };
}

function scrollConsoleIntoView() {
  const heading = document.querySelector(".debug-heading");
  if (heading) heading.scrollIntoView({ behavior: "smooth", block: "start" });
}

function visibleRunIds() {
  return filteredRuns().map((run) => run.id);
}

function filteredRuns() {
  let runs = [...state.runs];
  // Folder filter
  if (state.activeFolder === "") {
    runs = runs.filter((r) => !r.folder);
  } else if (state.activeFolder) {
    runs = runs.filter((r) => r.folder === state.activeFolder);
  }
  // Search
  if (state.searchQuery) {
    const q = state.searchQuery.toLowerCase();
    runs = runs.filter(
      (r) =>
        (r.display_name || r.id).toLowerCase().includes(q) ||
        r.id.toLowerCase().includes(q) ||
        (r.params?.task || "").toLowerCase().includes(q) ||
        (r.status || "").toLowerCase().includes(q)
    );
  }
  // Status filter
  if (state.statusFilter) {
    runs = runs.filter((r) => r.status === state.statusFilter);
  }
  // Sort
  runs.sort((a, b) => {
    switch (state.sortKey) {
      case "oldest":
        return (a.created_at || "").localeCompare(b.created_at || "");
      case "status":
        return (a.status || "").localeCompare(b.status || "");
      case "iters-desc": {
        const ai = a.params?.max_iterations ?? 0;
        const bi = b.params?.max_iterations ?? 0;
        return bi - ai;
      }
      case "duration-asc": {
        const dur = (r) => {
          const s = Date.parse(r.created_at || "");
          const e = Date.parse(r.updated_at || "");
          return Number.isNaN(s) || Number.isNaN(e) ? Infinity : e - s;
        };
        return dur(a) - dur(b);
      }
      default: // "newest"
        return (b.created_at || "").localeCompare(a.created_at || "");
    }
  });
  return runs;
}

function renderRuns() {
  const runs = filteredRuns();
  const badge = $("#run-count-badge");
  if (badge) badge.textContent = `${runs.length} run${runs.length !== 1 ? "s" : ""}`;
  if (!runs.length) {
    $("#runs").innerHTML = state.runs.length
      ? `<article class="empty-panel">No runs match your search or filter.</article>`
      : `<article class="empty-panel">No training history found yet.</article>`;
    updateBulkToolbar();
    return;
  }
  const gpuProcess = activeGpuProcess();
  $("#runs").innerHTML = runs
    .map((run) => {
      const active = state.selectedRun && state.selectedRun.id === run.id ? "active" : "";
      const title = run.display_name || run.id;
      const deleting = state.pendingDeleteRunIds.has(run.id);
      const queued = String(run.status || "").toLowerCase() === "queued";
      const canTensorboard = Boolean(run.log_dir);
      const canCheckpoint = Boolean(run.latest_checkpoint);
      const playProcessId = activeProcessIdForRun(run.id, "play");
      const videoProcessId = activeProcessIdForRun(run.id, "video");
      const onnxProcessId = activeProcessIdForRun(run.id, "onnx");
      const paramSummary = runParamSummary(run);
      const timeSummary = runTimeSummary(run);
      const videoText = videoProcessId ? "recording video" : videoSummary(run);
      const onnxText = onnxProcessId ? "exporting ONNX" : onnxSummary(run);
      const selected = state.selectedRunIds.has(run.id) || deleting ? "checked" : "";
      const playAction = playProcessId ? "stop-play" : "play";
      const playLabel = playProcessId ? "Stop Play" : "Play";
      const playProcessAttr = playProcessId ? `data-process-id="${escapeHtml(playProcessId)}"` : "";
      const playDisabled = queued || (!canCheckpoint && !playProcessId) || Boolean(gpuProcess && !playProcessId);
      const canTweak = !["running", "stopping"].includes(String(run.status || "").toLowerCase());
      const unread = state.notifications.unreadRunIds.has(run.id);
      return `
        <article class="run-card ${active} ${unread ? "unread" : ""} ${deleting ? "deleting" : ""}" data-run-id="${escapeHtml(run.id)}">
          <input class="run-select-checkbox" type="checkbox" data-run-id="${escapeHtml(run.id)}" ${selected} ${deleting ? "disabled" : ""} aria-label="Select ${escapeHtml(title)} for folder move" data-tooltip="Select for folder move">
          <div class="run-top">
            <div class="run-title">
              ${unread ? `<span class="unread-dot" data-tooltip="Unread history update"></span>` : ""}
              <strong>${escapeHtml(title)}</strong>
            </div>
            <span class="pill status-pill ${deleting ? statusClass("deleting") : statusClass(run.status)}">${deleting ? "deleting" : escapeHtml(statusLabel("run", run.status))}</span>
          </div>
          ${paramSummary ? `<small>${escapeHtml(paramSummary)}</small>` : ""}
          ${timeSummary ? `<small>${escapeHtml(timeSummary)}</small>` : ""}
          <small>${escapeHtml(runLogSummary(run))}</small>
          ${run.reward_preset_id && run.reward_preset_id !== "baseline"
            ? `<small><span class="reward-diff-badge">preset: ${escapeHtml(run.reward_preset_id)}</span></small>`
            : run.reward_diff_count > 0
              ? `<small><span class="reward-diff-badge">${escapeHtml(String(run.reward_diff_count))} reward override${run.reward_diff_count !== 1 ? "s" : ""}</span></small>`
              : ""}
          ${run.terrain_preset_id && run.terrain_preset_id !== "baseline"
            ? `<small><span class="terrain-diff-badge">terrain: ${escapeHtml(run.terrain_preset_id)}</span></small>`
            : run.terrain_diff_count > 0
              ? `<small><span class="terrain-diff-badge">${escapeHtml(String(run.terrain_diff_count))} terrain override${run.terrain_diff_count !== 1 ? "s" : ""}</span></small>`
              : ""}
          ${queued ? `<small>waiting for GPU queue</small>` : ""}
          <small>${escapeHtml(checkpointSummary(run))}${videoText ? ` · ${escapeHtml(videoText)}` : ""}${onnxText ? ` · ${escapeHtml(onnxText)}` : ""}${escapeHtml(runStatusDetail(run))}${run.has_notes ? " <strong>+ notes</strong>" : ""}</small>
          <div class="run-actions">
            <button type="button" data-action="tensorboard" data-run-id="${escapeHtml(run.id)}" ${runButtonDisabled(deleting || !canTensorboard)} data-tooltip="Open metrics">TensorBoard</button>
            <button type="button" data-action="${playAction}" data-run-id="${escapeHtml(run.id)}" ${playProcessAttr} ${runButtonDisabled(deleting || playDisabled)} data-tooltip="${playProcessId ? "Stop Isaac playback" : "Play checkpoint"}">${escapeHtml(playLabel)}</button>
            <button type="button" data-action="resume" data-run-id="${escapeHtml(run.id)}" ${runButtonDisabled(deleting || !canCheckpoint)} data-tooltip="Resume training from checkpoint">Resume to Train</button>
            <button type="button" data-action="tweak" data-run-id="${escapeHtml(run.id)}" ${runButtonDisabled(deleting || queued || !canTweak)} data-tooltip="Copy this run into an editable reward tweak draft">Tweak</button>
            <button type="button" data-action="console" data-run-id="${escapeHtml(run.id)}" ${runButtonDisabled(deleting)} data-tooltip="Show Process Console">Console</button>
            ${queued
              ? `<button type="button" data-action="cancel-queue" data-run-id="${escapeHtml(run.id)}" class="danger-button" ${runButtonDisabled(deleting)} data-tooltip="Cancel this queued training run">Cancel Queue</button>`
              : ""}
            ${videoProcessId
              ? `<button type="button" data-action="stop-video" data-run-id="${escapeHtml(run.id)}" data-process-id="${escapeHtml(videoProcessId)}" ${runButtonDisabled(deleting)} data-tooltip="Stop recording">Stop Recording</button>`
              : ""}
            ${state.selectedRun && state.selectedRun.id !== run.id
              ? `<button type="button" data-action="compare" data-run-id="${escapeHtml(run.id)}" ${runButtonDisabled(deleting)} data-tooltip="Compare with selected">Compare</button>`
              : ""}
          </div>
        </article>
      `;
    })
    .join("");
  document.querySelectorAll(".run-card").forEach((card) => {
    card.addEventListener("click", (event) => {
      const checkbox = event.target.closest(".run-select-checkbox");
      if (checkbox) {
        event.stopPropagation();
        toggleRunSelection(checkbox.dataset.runId, checkbox.checked);
        return;
      }
      const button = event.target.closest("button[data-action]");
      if (button) {
        event.stopPropagation();
        handleRunAction(button.dataset.action, button.dataset.runId, button.dataset.processId);
        return;
      }
      selectRun(card.dataset.runId);
    });
  });
  updateBulkToolbar();
}

function videoUrl(run) {
  return `/api/runs/${encodeURIComponent(run.id)}/video?v=${encodeURIComponent(run.latest_video || run.updated_at || "")}`;
}

function clearVideoPlayer() {
  const video = $("#result-video");
  video.removeAttribute("src");
  video.load();
}

function videoFolder(run) {
  return run && run.latest_video ? String(run.latest_video).replace(/\/[^/]+$/, "") : "";
}

function onnxFolder(run) {
  return run && run.onnx_path ? String(run.onnx_path).replace(/\/[^/]+$/, "") : "";
}

function activeVideoProcessId(run) {
  if (!run) return "";
  const processId = activeProcessIdForRun(run.id, "video");
  if (processId) return processId;
  return run.video_status === "recording" ? run.video_process_id || "" : "";
}

function activeOnnxProcessId(run) {
  if (!run) return "";
  const processId = activeProcessIdForRun(run.id, "onnx");
  if (processId) return processId;
  return run.onnx_status === "exporting" ? run.onnx_process_id || "" : "";
}

function videoPresetLabel(preset) {
  const name = String(preset.preset || "").replace(/^\w/, (char) => char.toUpperCase());
  return `${name} · ${preset.width}x${preset.height} · ${preset.length} steps`;
}

function renderVideoPanel(run) {
  const panel = $("#video-panel");
  const stateBadge = $("#video-state");
  const video = $("#result-video");
  const message = $("#video-message");
  const hasCheckpoint = Boolean(run && run.latest_checkpoint);
  if (!run || (!run.latest_video && !run.video_status && !hasCheckpoint)) {
    panel.hidden = true;
    clearVideoPlayer();
    message.textContent = "";
    return;
  }
  panel.hidden = false;
  const gpuProcess = activeGpuProcess();
  const videoProcessId = activeVideoProcessId(run);
  $("#record-video").disabled = !hasCheckpoint || Boolean(gpuProcess);
  $("#stop-recording").hidden = !videoProcessId;
  $("#open-video-folder").hidden = !run.latest_video;
  $("#copy-video-path").hidden = !run.latest_video;
  if (videoProcessId || run.video_status === "recording") {
    if (run.latest_video) {
      const src = videoUrl(run);
      if (video.getAttribute("src") !== src) video.setAttribute("src", src);
    } else {
      clearVideoPlayer();
    }
    stateBadge.textContent = "Recording";
    stateBadge.className = "status-badge status-running";
    message.textContent = "A headless playback is recording now. The video will appear here when it finishes.";
    return;
  }
  if (run.latest_video) {
    const src = videoUrl(run);
    stateBadge.textContent = "Video Ready";
    stateBadge.className = "status-badge status-completed";
    if (video.getAttribute("src") !== src) video.setAttribute("src", src);
    message.textContent = `Saved from the latest checkpoint. ${run.video_params ? videoPresetLabel(run.video_params) : ""}`;
    return;
  }
  clearVideoPlayer();
  if (run.video_status === "missing_checkpoint") {
    stateBadge.textContent = "Waiting";
    stateBadge.className = "status-badge status-interrupted";
    message.textContent = "Training completed but no checkpoint was found yet, so recording did not start.";
    return;
  }
  if (hasCheckpoint && !run.video_status) {
    stateBadge.textContent = "Ready";
    stateBadge.className = "status-badge muted-pill";
    message.textContent = "No video recorded yet. Record Video uses high quality by default.";
    return;
  }
  stateBadge.textContent = "Video Failed";
  stateBadge.className = "status-badge status-failed";
  message.textContent = "Recording failed. Use the Process Console for the launch command and captured output.";
}

function renderRunDetails() {
  if (state.comparisonMode && state.selectedRun && state.comparisonRun) {
    renderComparisonPanel(state.selectedRun, state.comparisonRun);
    return;
  }
  const run = state.selectedRun;
  const runName = $("#run-name");
  const playProcessId = run ? activeProcessIdForRun(run.id, "play") : "";
  const onnxProcessId = run ? activeOnnxProcessId(run) : "";
  const gpuProcess = activeGpuProcess();
  const queued = run ? String(run.status || "").toLowerCase() === "queued" : false;

  // Header
  $("#details-title").textContent = run ? run.display_name || run.id : "Run Details";
  const subtitle = $("#details-subtitle");
  if (subtitle) {
    subtitle.textContent = run
      ? `${run.status || "unknown"} · ${run.id}`
      : "Select a run from the list to view details.";
  }

  // Run Info block
  const infoBlock = $("#run-info-block");
  const infoGrid = $("#run-info-grid");
  if (infoBlock && infoGrid && run) {
    const rows = [];
    if (run.created_at) rows.push(["Created", formatRelativeTime(run.created_at)]);
    const dur = formatDuration(run.created_at, run.updated_at);
    if (dur) rows.push(["Duration", dur]);
    if (run.params?.task) rows.push(["Task", run.params.task]);
    if (run.params?.num_envs != null) rows.push(["Envs", run.params.num_envs]);
    if (run.params?.max_iterations != null) rows.push(["Iters", run.params.max_iterations]);
    const ckptIter = checkpointIteration(run.latest_checkpoint);
    if (ckptIter !== null) rows.push(["Checkpoint", `iter ${ckptIter}`]);
    const onnxText = onnxProcessId ? "exporting" : (run.onnx_path ? "ready" : (run.onnx_status === "failed" ? "failed" : "missing"));
    rows.push(["ONNX", onnxText]);
    if (run.reward_preset_id && run.reward_preset_id !== "baseline")
      rows.push(["Reward preset", run.reward_preset_id]);
    if (run.terrain_preset_id && run.terrain_preset_id !== "baseline")
      rows.push(["Terrain preset", run.terrain_preset_id]);
    if (run.convergence_detected)
      rows.push(["Converged", `iter ${run.convergence_iteration} (Δ ${run.convergence_improvement_pct?.toFixed(1)}%)`]);
    infoGrid.innerHTML = rows
      .map(([k, v]) => `<span class="info-key">${escapeHtml(k)}</span><span class="info-val">${escapeHtml(String(v))}</span>`)
      .join("");
    infoBlock.style.display = "";
  } else if (infoBlock) {
    infoBlock.style.display = "none";
  }

  // Rename
  if (!run) {
    state.renameDirty = false;
    state.renameDraftRunId = null;
    runName.value = "";
    hideRunConfigPanels();
  } else if (!(state.renameDirty && state.renameDraftRunId === run.id)) {
    runName.value = run.display_name || "";
  }
  runName.disabled = !run;

  // Folder select
  renderFolderSelect(run);

  // Inputs
  const notesEditor = $("#notes-editor");
  notesEditor.disabled = !run;
  if (!run) notesEditor.value = "";
  $("#save-name").disabled = !run;
  $("#save-notes").disabled = !run;

  // Action buttons
  $("#delete-run").disabled = !run;
  $("#compact-run").disabled = !run || !run.log_dir || Boolean(run && activeProcessForRun(run.id));
  $("#open-run-folder").disabled = !run || !run.log_dir;
  $("#tensorboard-run").disabled = !run || !run.log_dir;
  $("#play-run").disabled = !run || queued || (!run.latest_checkpoint && !playProcessId) || Boolean(gpuProcess && !playProcessId);
  $("#play-run").textContent = playProcessId ? "Stop Play" : "Play";
  $("#export-onnx").disabled = !run || queued || !run.latest_checkpoint || Boolean(gpuProcess);
  $("#export-onnx").textContent = onnxProcessId ? "Exporting ONNX" : "Export ONNX";
  $("#copy-onnx-path").hidden = !run || !run.onnx_path;
  $("#copy-onnx-path").disabled = !run || !run.onnx_path;
  $("#open-onnx-folder").hidden = !run || !run.onnx_path;
  $("#open-onnx-folder").disabled = !run || !run.onnx_path;
  $("#resume-run").disabled = !run || !run.latest_checkpoint;
  $("#tweak-run").disabled = !run || ["running", "stopping"].includes(String(run.status || "").toLowerCase());
  $("#stop-process").disabled = !state.debugTarget && !run;

  const hasCommand = Boolean(state.lastDebug && state.lastDebug.command);
  $("#copy-command").hidden = !hasCommand;
  $("#copy-command").disabled = !hasCommand;
  $("#open-process-log-folder").disabled = !state.lastDebug || !(state.lastDebug.process_log || state.lastDebug.log_file);

  renderVideoPanel(run);
}

function hasActiveRun() {
  return (
    Object.keys(state.activeProcessMap).length > 0 ||
    state.runs.some((run) => ["queued", "running", "stopping"].includes(run.status) || run.video_status === "recording")
  );
}

function scheduleRunsRefresh() {
  if (state.runsRefreshTimer) clearTimeout(state.runsRefreshTimer);
  const delay = hasActiveRun() ? RUNS_POLL_ACTIVE_MS : RUNS_POLL_IDLE_MS;
  state.runsRefreshTimer = setTimeout(async () => {
    try {
      await loadRuns();
    } catch {
      scheduleRunsRefresh();
    }
  }, delay);
}

async function loadRuns() {
  const selectedId = state.selectedRun && state.selectedRun.id;
  const [runsData, processesData] = await Promise.all([api("/api/runs"), api("/api/processes")]);
  state.runs = runsData.runs;
  reconcileHistoryNotifications(state.runs);
  state.activeProcessMap = {};
  state.activeProcesses = [];
  state.activeProcessesByRun = {};
  state.activeProcessByKind = {};
  for (const process of processesData.processes) {
    if (process.returncode !== null) continue;
    state.activeProcesses.push(process);
    rememberActiveProcess(process.run_id, process);
    rememberActiveProcess(process.source_run_id, process);
  }
  const validRunIds = new Set(state.runs.map((run) => run.id));
  state.selectedRunIds = new Set([...state.selectedRunIds].filter((runId) => validRunIds.has(runId)));
  if (selectedId) {
    const selected = findRun(selectedId);
    if (selected) {
      state.selectedRun = selected;
    } else {
      clearRunDetailState({ render: false });
    }
  }
  renderRuns();
  renderRunDetails();
  renderFolderSidebar();
  scheduleRunsRefresh();
}

async function selectRun(runId) {
  const run = findRun(runId);
  if (!run) {
    setStatus("Run not found. Refresh history and try again.");
    return;
  }
  if (!state.selectedRun || state.selectedRun.id !== runId) {
    state.renameDirty = false;
    state.renameDraftRunId = null;
  }
  state.selectedRun = run;
  markHistoryRead(runId);
  renderRunDetails();
  renderRuns();
  // Hide reward panel until loaded
  const rewardPanel = $("#reward-config-panel");
  if (rewardPanel) rewardPanel.hidden = true;
  const terrainPanel = $("#terrain-config-panel");
  if (terrainPanel) terrainPanel.hidden = true;
  const [notesData] = await Promise.all([
    api(`/api/runs/${encodeURIComponent(runId)}/notes`),
    run.log_dir ? loadRewardConfigForRun(runId) : Promise.resolve(),
    run.log_dir ? loadTerrainConfigForRun(runId) : Promise.resolve(),
  ]);
  if (!state.selectedRun || state.selectedRun.id !== runId) return;
  $("#notes-editor").value = notesData.notes;
  setStatus(run.latest_checkpoint ? `Latest checkpoint: ${run.latest_checkpoint}` : "No checkpoint available yet.");
  setDebugTarget({ type: "run", id: runId });
}

function debugEndpoint(target) {
  if (target.type === "process") return `/api/processes/${encodeURIComponent(target.id)}/debug`;
  return `/api/runs/${encodeURIComponent(target.id)}/debug`;
}

function terminalUrl(target) {
  return `/static/terminal.html?type=${encodeURIComponent(target.type)}&id=${encodeURIComponent(target.id)}`;
}

function openTerminalView(target = state.debugTarget) {
  if (!target) {
    setStatus("Select a run or start a process first.");
    return null;
  }
  return window.open(terminalUrl(target), "_blank", "noopener");
}

async function copyText(text) {
  if (!text.trim()) {
    setStatus("No console output to copy yet.");
    return;
  }
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

async function copyDebugOutput() {
  const text = [
    $("#debug-status").textContent,
    "",
    "Command",
    $("#debug-command").textContent,
    "",
    "Output",
    $("#debug-log").textContent,
  ].join("\n");
  await copyText(text);
  setStatus("Console output copied.");
}

async function copyLaunchCommand() {
  const command = state.lastDebug && state.lastDebug.command;
  if (!command) {
    setStatus("No launch command is available for this process.");
    return;
  }
  await copyText(command);
  setStatus("Launch command copied.");
}

async function openLocation(path, label = "location") {
  if (!path) {
    setStatus(`No ${label} path is available.`);
    return;
  }
  const data = await api("/api/open-location", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
  await copyText(data.path);
  const suffix = data.opened ? " Host open requested." : "";
  setStatus(`${label} path copied. ${suffix} ${data.command}`);
}

async function openRunFolder() {
  if (!state.selectedRun || !state.selectedRun.log_dir) {
    setStatus("No run folder is linked yet.");
    return;
  }
  await openLocation(state.selectedRun.log_dir, "Run folder");
}

async function openVideoFolder() {
  const folder = videoFolder(state.selectedRun);
  await openLocation(folder, "Video folder");
}

async function openProcessLogFolder() {
  const logPath = state.lastDebug && (state.lastDebug.process_log || state.lastDebug.log_file);
  const folder = logPath ? String(logPath).replace(/\/[^/]+$/, "") : "";
  await openLocation(folder, "Process log folder");
}

async function copyVideoPath() {
  if (!state.selectedRun || !state.selectedRun.latest_video) {
    setStatus("No video path is available yet.");
    return;
  }
  await copyText(state.selectedRun.latest_video);
  setStatus(`Video path copied: ${state.selectedRun.latest_video}`);
}

async function openOnnxFolder() {
  await openLocation(onnxFolder(state.selectedRun), "ONNX export folder");
}

async function copyOnnxPath() {
  if (!state.selectedRun || !state.selectedRun.onnx_path) {
    setStatus("No ONNX path is available yet.");
    return;
  }
  await copyText(state.selectedRun.onnx_path);
  setStatus(`ONNX path copied: ${state.selectedRun.onnx_path}`);
}

function isLiveDebug(debug) {
  if (debug.kind) return debug.returncode === null;
  return debug.status === "running" || debug.status === "stopping" || debug.status === "video recording";
}

function outputDiagnosis(output) {
  if (!output) return "";
  if (/ERROR_OUT_OF_DEVICE_MEMORY|Out of GPU memory|Unable to allocate buffer/.test(output)) {
    return "Diagnosis: GPU memory is exhausted. Stop old Isaac/RedRHex processes, keep Headless checked, then retry.";
  }
  if (/moviepy is not installed|gymnasium\[other\]/i.test(output)) {
    return 'Diagnosis: video encoding dependencies are missing. Run: pip install "gymnasium[other]" moviepy';
  }
  if (/ffmpeg|ImageSequenceClip|encoder/i.test(output) && /error|not found|failed/i.test(output)) {
    return "Diagnosis: video encoding failed. Check that moviepy and ffmpeg are available in the active conda environment.";
  }
  const moduleMatch = output.match(/ModuleNotFoundError: No module named '([^']+)'/);
  if (moduleMatch && moduleMatch[1] !== "pkg_resources") {
    return `Diagnosis: Python module '${moduleMatch[1]}' is missing in the active conda environment. Run: pip install ${moduleMatch[1]}`;
  }
  if (/ModuleNotFoundError: No module named 'pkg_resources'/.test(output)) {
    return "Diagnosis: TensorBoard is missing setuptools/pkg_resources inside the selected Python environment.";
  }
  if (/No checkpoints in the directory: .* match/.test(output)) {
    return "Diagnosis: the resume checkpoint was interpreted relative to the wrong run folder. Use the updated panel and retry.";
  }
  if (/no MP4 was produced|No recorded video found/i.test(output)) {
    return "Diagnosis: the video process ended but no MP4 was found. Open the process log folder and check the play/video output.";
  }
  if (/policy\.onnx was not produced|ONNX export finished/i.test(output)) {
    return "Diagnosis: ONNX export finished without exported/policy.onnx. Check the checkpoint load and exporter output.";
  }
  return "";
}

function renderDebug(debug) {
  state.lastDebug = debug;
  const logTail = debug.log_tail ?? debug.process_log_tail ?? "";
  const live = isLiveDebug(debug);
  const rows = [];
  if (debug.kind) rows.push(["Type", `${debug.kind} process`]);
  if (debug.id) rows.push(["Run", debug.id]);
  if (debug.run_id && !debug.id) rows.push(["Process", debug.run_id]);
  if (debug.pid) rows.push(["PID", debug.pid]);
  if (debug.status) rows.push(["Status", debug.status]);
  if (debug.returncode !== undefined && debug.returncode !== null) rows.push(["Return", debug.returncode]);
  if (debug.process_log || debug.log_file) rows.push(["Log", debug.process_log || debug.log_file]);
  const diagnosis = outputDiagnosis(logTail);
  if (diagnosis) rows.push(["Hint", diagnosis]);

  $("#debug-live").textContent = live ? "Live" : "Snapshot";
  $("#debug-live").className = live ? "status-badge live-pill" : "status-badge muted-pill";
  $("#debug-status").innerHTML = rows.length
    ? rows
        .map(([key, value]) => `<span class="debug-kv"><strong>${escapeHtml(key)}:</strong> ${escapeHtml(String(value))}</span>`)
        .join("")
    : escapeHtml(debug.debug_hint || "No process selected.");
  const commandText = debug.command || "";
  const outputText = logTail || debug.debug_hint || "No terminal output captured yet.";
  $("#debug-command").textContent = commandText;
  $("#debug-command-block").hidden = !commandText;
  $("#debug-log").textContent = outputText;
  $("#debug-log-block").hidden = !outputText;
  $("#debug-log").scrollTop = $("#debug-log").scrollHeight;
  renderRunDetails();
}

async function refreshDebug() {
  if (!state.debugTarget) return;
  const target = { ...state.debugTarget };
  try {
    const debug = await api(debugEndpoint(target));
    if (
      !state.debugTarget ||
      state.debugTarget.type !== target.type ||
      state.debugTarget.id !== target.id ||
      (target.type === "run" && !findRun(target.id))
    ) {
      return;
    }
    renderDebug(debug);
    if (!isLiveDebug(debug) && state.debugTarget.type === "process") stopDebugPolling();
  } catch (error) {
    if (
      !state.debugTarget ||
      state.debugTarget.type !== target.type ||
      state.debugTarget.id !== target.id
    ) {
      return;
    }
    $("#debug-live").textContent = "Error";
    $("#debug-live").className = "status-badge error-pill";
    $("#debug-status").innerHTML = `<span class="debug-kv"><strong>Error:</strong> ${escapeHtml(error.message)}</span>`;
  }
}

function stopDebugPolling() {
  if (state.debugTimer) clearInterval(state.debugTimer);
  state.debugTimer = null;
}

function setDebugTarget(target) {
  state.debugTarget = target;
  renderRunDetails();
  stopDebugPolling();
  refreshDebug();
  state.debugTimer = setInterval(refreshDebug, DEBUG_POLL_MS);
}

function renderDebugPayload(payload) {
  if (!payload) return;
  renderDebug({
    ...payload,
    log_tail: payload.log_tail ?? payload.process_log_tail ?? "",
  });
}

function hideRunConfigPanels() {
  const rewardPanel = $("#reward-config-panel");
  const rewardContent = $("#reward-config-content");
  if (rewardPanel) rewardPanel.hidden = true;
  if (rewardContent) rewardContent.innerHTML = "";
  const terrainPanel = $("#terrain-config-panel");
  const terrainContent = $("#terrain-config-content");
  if (terrainPanel) terrainPanel.hidden = true;
  if (terrainContent) terrainContent.innerHTML = "";
}

function clearRunDetailState({ render = true } = {}) {
  state.selectedRun = null;
  state.comparisonRun = null;
  state.comparisonMode = false;
  state.debugTarget = null;
  state.lastDebug = null;
  state.renameDirty = false;
  state.renameDraftRunId = null;
  stopDebugPolling();
  const notesEditor = $("#notes-editor");
  if (notesEditor) notesEditor.value = "";
  const debugCommand = $("#debug-command");
  if (debugCommand) debugCommand.textContent = "";
  const debugLog = $("#debug-log");
  if (debugLog) debugLog.textContent = "";
  const debugCommandBlock = $("#debug-command-block");
  if (debugCommandBlock) debugCommandBlock.hidden = true;
  const debugLogBlock = $("#debug-log-block");
  if (debugLogBlock) debugLogBlock.hidden = true;
  const debugStatus = $("#debug-status");
  if (debugStatus) debugStatus.textContent = "";
  const debugLive = $("#debug-live");
  if (debugLive) {
    debugLive.textContent = "Idle";
    debugLive.className = "status-badge muted-pill";
  }
  hideRunConfigPanels();
  renderVideoPanel(null);
  if (render) renderRunDetails();
}

async function startTraining(event) {
  event.preventDefault();
  const form = $("#train-form");
  $("#train-status").textContent = "Starting training...";
  try {
    const payload = formData(form);
    clearTrainingRunName(form);
    const run = await api("/api/training/start", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const runLabel = run.display_name ? `${run.display_name} (${run.id})` : run.id;
    $("#train-status").textContent =
      run.status === "queued"
        ? `Queued ${runLabel}. It will start when the GPU is free.`
        : `Started ${runLabel} with pid ${run.pid}`;
    markHistoryUnread(run.id);
    await loadRuns();
    await loadActivity();
  } catch (error) {
    $("#train-status").textContent = error.message;
  }
}

async function saveNotes() {
  if (!state.selectedRun) {
    setStatus("Select a run first.");
    return;
  }
  await api(`/api/runs/${encodeURIComponent(state.selectedRun.id)}/notes`, {
    method: "POST",
    body: JSON.stringify({ notes: $("#notes-editor").value }),
  });
  await loadRuns();
  setStatus("Notes saved.");
}

async function saveName() {
  if (!state.selectedRun) {
    setStatus("Select a run first.");
    return;
  }
  const runId = state.selectedRun.id;
  const displayName = $("#run-name").value;
  const data = await api(`/api/runs/${encodeURIComponent(runId)}/rename`, {
    method: "POST",
    body: JSON.stringify({ display_name: displayName }),
  });
  state.renameDirty = false;
  state.renameDraftRunId = null;
  await loadRuns();
  await loadActivity();
  state.selectedRun = findRun(runId) || data.run || state.selectedRun;
  renderRunDetails();
  renderRuns();
  setStatus("Name saved.");
}

function tensorboardHost() {
  return location.hostname === "127.0.0.1" || location.hostname === "localhost" ? "127.0.0.1" : "0.0.0.0";
}

function displayTensorboardUrl(data, host) {
  return host === "0.0.0.0" ? `http://${location.hostname}:${data.port}` : data.url;
}

function openPendingTensorBoardWindow() {
  const win = window.open("about:blank", "_blank");
  if (!win) return null;
  win.document.write(
    "<!doctype html><title>TensorBoard</title><body style=\"font:14px system-ui;padding:24px;background:#f5f7f8;color:#1f2523\"><h1>Starting TensorBoard...</h1><p>The training panel is launching TensorBoard for this run.</p></body>"
  );
  win.document.close();
  return win;
}

function showTensorBoardWindowError(win, error) {
  if (!win || win.closed) return;
  win.document.open();
  win.document.write(
    `<!doctype html><title>TensorBoard failed</title><body style="font:14px system-ui;padding:24px;background:#f5f7f8;color:#1f2523"><h1>TensorBoard failed to start</h1><p>${escapeHtml(
      error.message
    )}</p><pre style="white-space:pre-wrap;background:#232d30;color:#f4f9fa;padding:12px;border-radius:7px">${escapeHtml(
      error.data?.log_tail || ""
    )}</pre></body>`
  );
  win.document.close();
}

async function startTensorBoardForRun(runId, pendingWindow) {
  const host = tensorboardHost();
  const win = pendingWindow || openPendingTensorBoardWindow();
  const data = await api(`/api/runs/${encodeURIComponent(runId)}/tensorboard`, {
    method: "POST",
    body: JSON.stringify({ host }),
  });
  const url = displayTensorboardUrl(data, host);
  if (win && !win.closed) {
    win.opener = null;
    win.location.href = url;
  }
  setStatus(
    data.already_running ? `TensorBoard is already running on port ${data.port}.` : `Started TensorBoard on port ${data.port}.`,
    url
  );
  setDebugTarget({ type: "process", id: data.id });
}

async function playRun(runId) {
  const gpuProcess = activeGpuProcess();
  if (gpuProcess) {
    setStatus(mediaLockMessage(gpuProcess));
    await loadRuns();
    return;
  }
  const data = await api(`/api/runs/${encodeURIComponent(runId)}/play`, {
    method: "POST",
    body: JSON.stringify({ device: "cuda:0" }),
  });
  const target = { type: "process", id: data.id };
  setStatus(data.attach_command ? `Started play process ${data.pid}. Attach with: ${data.attach_command}` : `Started play process ${data.pid}.`);
  setDebugTarget(target);
  await loadRuns();
}

async function recordVideo() {
  if (!state.selectedRun) {
    setStatus("Select a run first.");
    return;
  }
  const gpuProcess = activeGpuProcess();
  if (gpuProcess) {
    setStatus(mediaLockMessage(gpuProcess));
    await loadRuns();
    return;
  }
  const data = await api(`/api/runs/${encodeURIComponent(state.selectedRun.id)}/record-video`, {
    method: "POST",
    body: JSON.stringify({ device: "cuda:0" }),
  });
  const target = { type: "process", id: data.id };
  setDebugTarget(target);
  setStatus(
    data.attach_command
      ? `Recording high quality video. Attach with: ${data.attach_command}`
      : "Recording high quality video."
  );
  await loadRuns();
}

async function exportOnnx() {
  if (!state.selectedRun) {
    setStatus("Select a run first.");
    return;
  }
  const gpuProcess = activeGpuProcess();
  if (gpuProcess) {
    setStatus(mediaLockMessage(gpuProcess));
    await loadRuns();
    return;
  }
  const data = await api(`/api/runs/${encodeURIComponent(state.selectedRun.id)}/export-onnx`, {
    method: "POST",
    body: JSON.stringify({ device: "cuda:0" }),
  });
  setDebugTarget({ type: "process", id: data.id });
  setStatus(data.attach_command ? `Exporting ONNX. Attach with: ${data.attach_command}` : "Exporting ONNX.");
  await loadRuns();
}

async function stopVideoRecording() {
  const processId = activeVideoProcessId(state.selectedRun);
  if (!processId) {
    setStatus("No active video recording process was found.");
    return;
  }
  await stopVideoProcess(processId);
}

function resumeRun(runId) {
  const run = findRun(runId);
  if (!run || !run.latest_checkpoint) {
    setStatus("No checkpoint available for this run.");
    return;
  }
  const form = $("#train-form");
  form.elements.checkpoint.value = run.latest_checkpoint;
  setView("train");
  $("#train-status").textContent = `Resume selected from ${run.display_name || run.id}. Choose iterations/envs, then start training.`;
}

function handleActionError(error, pendingWindow = null) {
  if (pendingWindow) showTensorBoardWindowError(pendingWindow, error);
  if (error.data) {
    renderDebugPayload(error.data);
    if (error.data.run_id && error.data.kind) setDebugTarget({ type: "process", id: error.data.run_id });
  }
  setStatus(error.message);
}

async function runningProcessForSelectedRun() {
  if (!state.selectedRun) return null;
  const data = await api("/api/processes");
  return data.processes
    .filter(
      (process) =>
        process.returncode === null &&
        (process.run_id === state.selectedRun.id || process.source_run_id === state.selectedRun.id)
    )
    .sort((left, right) => String(right.started_at || "").localeCompare(String(left.started_at || "")))[0];
}

async function stopSelectedProcess() {
  let processId = state.debugTarget && state.debugTarget.type === "process" ? state.debugTarget.id : "";
  if (!processId) {
    const related = await runningProcessForSelectedRun();
    processId = related ? related.run_id : "";
  }
  if (!processId) {
    setStatus("No running training/play/video/TensorBoard process was found for the selected run.");
    return;
  }
  const data = await api("/api/training/stop", {
    method: "POST",
    body: JSON.stringify({ run_id: processId }),
  });
  setDebugTarget({ type: "process", id: processId });
  setStatus(data.stopped ? `Stopping ${processId}...` : "Process is not running.");
  await refreshDebug();
}

async function stopProcessById(processId) {
  if (!processId) {
    setStatus("No active process was found for this run.");
    return;
  }
  const data = await api("/api/training/stop", {
    method: "POST",
    body: JSON.stringify({ run_id: processId }),
  });
  setDebugTarget({ type: "process", id: processId });
  setStatus(data.stopped ? `Stopping ${processId}...` : "Process is not running.");
  await loadRuns();
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForProcessExit(processId) {
  for (let index = 0; index < 24; index += 1) {
    await delay(750);
    await loadRuns();
    const stillLive = state.activeProcesses.some((process) => process.run_id === processId);
    if (!stillLive) return true;
  }
  return false;
}

async function stopPlayProcess(processId) {
  if (!processId) {
    setStatus("No active play process was found for this run.");
    return;
  }
  const data = await api("/api/training/stop", {
    method: "POST",
    body: JSON.stringify({ run_id: processId }),
  });
  setDebugTarget({ type: "process", id: processId });
  setStatus(data.stopped ? "Stopping play..." : "Play process is not running.");
  await waitForProcessExit(processId);
  await refreshDebug();
}

async function stopVideoProcess(processId) {
  if (!processId) {
    setStatus("No active recording process was found for this run.");
    return;
  }
  const data = await api("/api/training/stop", {
    method: "POST",
    body: JSON.stringify({ run_id: processId }),
  });
  setDebugTarget({ type: "process", id: processId });
  setStatus(data.stopped ? "Stopping recording..." : "Recording process is not running.");
  await waitForProcessExit(processId);
  await refreshDebug();
}

function formatDeletePreview(preview) {
  const paths = preview.paths.length
    ? preview.paths.map((item) => `- ${item.kind}: ${item.path}`).join("\n")
    : "- No log/note files were found; only the panel history entry will be removed.";
  return [
    "This permanently deletes the selected training history entry.",
    "",
    "It will remove these repo-owned files/directories:",
    paths,
    "",
    "Click OK to delete. This cannot be undone.",
  ].join("\n");
}

async function deleteSelectedRun() {
  if (!state.selectedRun) {
    setStatus("Select a run first.");
    return;
  }
  const runId = state.selectedRun.id;
  state.pendingDeleteRunIds.add(runId);
  renderRuns();
  try {
    const preview = await api(`/api/runs/${encodeURIComponent(runId)}/delete-preview`);
    if (!window.confirm(formatDeletePreview(preview))) {
      setStatus("Delete cancelled.");
      return;
    }
    const result = await api(`/api/runs/${encodeURIComponent(runId)}/delete`, {
      method: "POST",
      body: JSON.stringify({ confirm: true, delete_logs: true }),
    });
    const deletedRunId = result.run_id || runId;
    state.runs = state.runs.filter((run) => run.id !== deletedRunId);
    clearRunDetailState();
    renderRuns();
    await loadRuns();
    await loadActivity();
    setStatus(`Deleted ${deletedRunId}. Removed ${result.deleted_paths.length} log/note path(s).`);
  } catch (error) {
    throw error;
  } finally {
    state.pendingDeleteRunIds.delete(runId);
    renderRuns();
  }
}

function formatBulkDeletePreview(preview) {
  const lines = [];
  for (const run of preview.runs || []) {
    const label = run.display_name || run.id;
    lines.push(`- ${label}: ${(run.paths || []).length} path(s)`);
  }
  if (preview.missing && preview.missing.length) {
    lines.push(`Missing: ${preview.missing.join(", ")}`);
  }
  return [
    `This permanently deletes ${preview.run_count || 0} selected run(s).`,
    `Repo-owned paths to remove: ${preview.path_count || 0}`,
    "",
    lines.join("\n") || "- No matching runs found.",
    "",
    "Click OK to delete. This cannot be undone.",
  ].join("\n");
}

async function deleteSelectedRuns() {
  const runIds = [...state.selectedRunIds];
  if (!runIds.length) {
    setStatus("Select one or more runs first.");
    return;
  }
  state.isBulkDeleting = true;
  runIds.forEach((runId) => state.pendingDeleteRunIds.add(runId));
  updateBulkToolbar();
  renderRuns();
  try {
    const preview = await api("/api/runs/delete-preview", {
      method: "POST",
      body: JSON.stringify({ run_ids: runIds, delete_logs: true }),
    });
    if (!preview.run_count) {
      setStatus("No selected runs can be deleted.");
      return;
    }
    if (!window.confirm(formatBulkDeletePreview(preview))) {
      setStatus("Bulk delete cancelled.");
      return;
    }
    const result = await api("/api/runs/delete", {
      method: "POST",
      body: JSON.stringify({ run_ids: runIds, delete_logs: true, confirm: true }),
    });
    state.selectedRunIds.clear();
    const affectedRunIds = new Set([
      ...(result.run_ids || []),
      ...(result.deleted_run_ids || []),
      ...runIds,
    ]);
    state.runs = state.runs.filter((run) => !affectedRunIds.has(run.id));
    if (state.selectedRun && affectedRunIds.has(state.selectedRun.id)) {
      clearRunDetailState();
    }
    renderRuns();
    await loadRuns();
    await loadActivity();
    const skipped = (result.skipped_duplicate_ids || []).length;
    const missing = (result.missing || []).length;
    const extras = [
      skipped ? `${skipped} duplicate skipped` : "",
      missing ? `${missing} missing` : "",
    ].filter(Boolean);
    const suffix = extras.length ? ` (${extras.join(", ")})` : "";
    setStatus(`Deleted ${result.deleted_count} run${result.deleted_count === 1 ? "" : "s"}. Removed ${result.deleted_paths.length} log/note path(s).${suffix}`);
  } finally {
    runIds.forEach((runId) => state.pendingDeleteRunIds.delete(runId));
    state.isBulkDeleting = false;
    updateBulkToolbar();
    renderRuns();
  }
}

async function handleRunAction(action, runId, processId = "") {
  const pendingWindow = action === "tensorboard" ? openPendingTensorBoardWindow() : null;
  try {
    if (action === "stop-play") {
      await stopPlayProcess(processId || activeProcessIdForRun(runId, "play"));
      return;
    }
    if (action === "stop-video") {
      await stopVideoProcess(processId || activeProcessIdForRun(runId, "video"));
      return;
    }
    if (action === "stop-process") {
      await stopProcessById(processId || state.activeProcessMap[runId]);
      return;
    }
    if (action === "cancel-queue") {
      await cancelQueuedRun(runId);
      return;
    }
    if (!state.selectedRun || state.selectedRun.id !== runId) {
      await selectRun(runId);
    }
    if (action === "tensorboard") await startTensorBoardForRun(runId, pendingWindow);
    if (action === "play") await playRun(runId);
    if (action === "resume") resumeRun(runId);
    if (action === "tweak") await tweakFromRun(runId);
    if (action === "compare") { startComparison(runId); return; }
    if (action === "console") {
      setDebugTarget(consoleTargetForRun(runId));
      scrollConsoleIntoView();
      setStatus("Process console loaded.");
    }
  } catch (error) {
    handleActionError(error, pendingWindow);
  }
}

async function cancelQueuedRun(runId) {
  const data = await api(`/api/runs/${encodeURIComponent(runId)}/cancel-queue`, { method: "POST" });
  await loadRuns();
  await loadActivity();
  setStatus(data.cancelled ? `Cancelled queued run ${runId}.` : "That run is no longer queued.");
}

function applyPreset(kind) {
  const form = $("#train-form");
  if (kind === "smoke") {
    form.elements.num_envs.value = 4;
    form.elements.max_iterations.value = 1;
  } else {
    form.elements.num_envs.value = 64;
    form.elements.max_iterations.value = 100;
  }
  form.elements.headless.checked = true;
  form.elements.device.value = "cuda:0";
}

function clearResume() {
  $("#train-form").elements.checkpoint.value = "";
  $("#train-status").textContent = "Resume checkpoint cleared.";
}

function applyTrainingParamsToForm(params) {
  const form = $("#train-form");
  if (!form || !params) return;
  form.elements.task.value = params.task || "Template-Redrhex-Direct-v0";
  form.elements.num_envs.value = params.num_envs ?? 4;
  form.elements.max_iterations.value = params.max_iterations ?? 1;
  form.elements.device.value = params.device || "cuda:0";
  form.elements.seed.value = params.seed ?? "";
  form.elements.checkpoint.value = "";
  form.elements.headless.checked = params.headless !== false;
}

async function applyTweakPayload(payload) {
  if (!payload || !payload.training_params || !payload.reward_preset) return;
  if (!state.presets.length) await loadRewardsPage();
  if (!state.terrainPresets.length) await loadTerrainPage();
  const params = payload.training_params;
  applyTrainingParamsToForm(params);
  state.rewardDraftPreset = {
    ...payload.reward_preset,
    draft: true,
    source_run_id: payload.source_run?.id || params.tweak_source_run_id || payload.reward_preset.source_run_id,
    source_label: params.tweak_source_label || payload.source_run?.display_name || payload.source_run?.id || "",
  };
  state.selectedPresetId = state.rewardDraftPreset.id;
  state.activePresetId = state.rewardDraftPreset.id;
  state.activePresetOverrides = state.rewardDraftPreset.values || {};
  state.activeTerrainPresetId = params.terrain_preset_id || "baseline";
  state.selectedTerrainPresetId = state.activeTerrainPresetId;
  state.activeTerrainPresetOverrides = params.terrain_overrides || {};
  renderPresets();
  renderTerrainPresets();
  setView("rewards");
  selectPresetForEdit(state.rewardDraftPreset.id);
  $("#train-status").textContent = payload.message || `Loaded tweak draft from ${state.rewardDraftPreset.source_label || "run"}.`;
  setStatus("Tweak draft is selected for the next training run. Adjust rewards, then start training.");
}

async function tweakFromLastRun() {
  try {
    await applyTweakPayload(await api("/api/tweaks/last-run"));
  } catch (error) {
    $("#train-status").textContent = error.message;
    setStatus(error.message);
  }
}

async function tweakFromRun(runId) {
  try {
    await applyTweakPayload(await api(`/api/runs/${encodeURIComponent(runId)}/tweak`));
  } catch (error) {
    setStatus(error.message);
  }
}

// ============================================================
// Rewards & Presets Page
// ============================================================

const REWARD_MAX_SCALE = 8; // denominator for bar fill percentage

const REWARD_META = {
  rew_scale_forward_vel:       { label: "Forward Velocity",          category: "Locomotion Goals",      sign: "positive", description: "Rewards moving in the commanded direction. Higher = robot pushes harder to move but may sacrifice stability." },
  rew_scale_vel_tracking:      { label: "Velocity Tracking (Linear)", category: "Locomotion Goals",      sign: "positive", description: "Rewards precisely matching the commanded XY speed (exponential loss). Higher = tighter speed following." },
  rew_scale_ang_vel_tracking:  { label: "Velocity Tracking (Turn)",  category: "Locomotion Goals",      sign: "positive", description: "Rewards matching the commanded turn rate. Higher = robot follows rotation commands more closely." },
  rew_scale_vel_tracking2:     { label: "Velocity Tracking (Aux)",   category: "Locomotion Goals",      sign: "positive", description: "Secondary velocity tracking term (L2 error). Works alongside the primary tracking reward." },
  rew_scale_direction_align:   { label: "Direction Alignment",       category: "Locomotion Goals",      sign: "positive", description: "Rewards moving in the same direction as commanded. Helps with diagonal and sideways motion." },
  rew_scale_rotation_direction:{ label: "In-Place Rotation Bonus",   category: "Rotation Mode",         sign: "positive", description: "Extra reward when the robot correctly rotates on the spot. Higher = stronger incentive for tight in-place turns." },
  rew_scale_smooth_rotation:   { label: "Smooth Rotation",           category: "Rotation Mode",         sign: "positive", description: "Rewards smooth rotation without abrupt speed changes. Currently 0 (disabled). Increase to penalise jerky turning." },
  rew_scale_rotation_dir:      { label: "Leg Rotation Direction",    category: "Leg Motion",            sign: "positive", description: "Rewards each leg rotating in the correct direction for the current command. More correct legs = more reward." },
  rew_scale_all_legs:          { label: "All Legs Active",           category: "Leg Motion",            sign: "positive", description: "Rewards having all six legs spinning. Encourages full leg use rather than dragging." },
  rew_scale_min_leg_vel:       { label: "Minimum Leg Speed",         category: "Leg Motion",            sign: "positive", description: "Rewards the slowest leg for moving. Ensures no leg is stalled while others rotate." },
  rew_scale_mean_leg_vel:      { label: "Mean Leg Speed",            category: "Leg Motion",            sign: "positive", description: "Rewards higher average leg rotation speed. Higher = generally faster leg movement." },
  rew_scale_orientation:       { label: "Body Tilt Penalty",         category: "Stability Penalties",   sign: "negative", description: "Penalises the body tilting from upright. More negative = stricter upright requirement. Near 0 = allows more exploration." },
  rew_scale_base_height:       { label: "Height Deviation Penalty",  category: "Stability Penalties",   sign: "negative", description: "Penalises the body being too high or too low (target: 12 cm). More negative = stricter height control." },
  rew_scale_lin_vel_z:         { label: "Vertical Bounce Penalty",   category: "Stability Penalties",   sign: "negative", description: "Penalises up-and-down bouncing. More negative = smoother vertical motion required." },
  rew_scale_ang_vel_xy:        { label: "Roll/Pitch Wobble Penalty", category: "Stability Penalties",   sign: "negative", description: "Penalises rolling and pitching. More negative = stricter anti-wobble requirement." },
  rew_scale_gait_coherence:    { label: "Tripod Phase Coherence",    category: "Gait Coordination",     sign: "positive", description: "Rewards legs within the same tripod group staying in sync (legs 1,3,5 together; 2,4,6 together)." },
  rew_scale_gait_phase_offset: { label: "Tripod Antiphase Reward",   category: "Gait Coordination",     sign: "positive", description: "Rewards the two tripod groups being 180° out of phase — the classic alternating tripod gait." },
  rew_scale_continuous_support:{ label: "Ground Contact Reward",     category: "Gait Coordination",     sign: "positive", description: "Rewards having at least one leg touching the ground at all times. Prevents mid-air hops." },
  rew_scale_abad_action:       { label: "ABAD Motion Reward",        category: "ABAD Control",          sign: "positive", description: "Rewards ABAD joints moving when lateral/rotation commands are given, staying neutral otherwise. Set to 0 to disable." },
  rew_scale_abad_stability:    { label: "ABAD Symmetry Reward",      category: "ABAD Control",          sign: "positive", description: "Rewards left-right ABAD asymmetry when turning (differential steering) and symmetry when walking straight." },
  rew_scale_alive:             { label: "Alive Bonus",               category: "Survival & Smoothness", sign: "positive", description: "Small bonus each step the robot is alive. Encourages longer episodes. Very large values may teach the robot to stand still." },
  rew_scale_action_rate:       { label: "Action Change Penalty",     category: "Survival & Smoothness", sign: "negative", description: "Penalises rapid joint command changes. More negative = smoother, less jerky motion." },
  rew_scale_drive_acc:         { label: "Drive Accel Penalty",       category: "Survival & Smoothness", sign: "negative", description: "Penalises sudden drive motor speed changes. Currently 0 (disabled). Increase to reduce jerky acceleration." },
  rew_scale_collision:         { label: "Body Collision Penalty",    category: "Collision",             sign: "negative", description: "Penalises the body hitting the ground. More negative = harsher punishment for falling flat. (Not yet active in code.)" },
};

const REWARD_CATEGORY_ORDER = [
  "Locomotion Goals", "Rotation Mode", "Leg Motion",
  "Stability Penalties", "Gait Coordination", "ABAD Control",
  "Survival & Smoothness", "Collision",
];

function rewardBarHtml(value, sign) {
  const absVal = Math.abs(value);
  const pct = Math.min(100, (absVal / REWARD_MAX_SCALE) * 100);
  const cls = absVal < 1e-9 ? "zero" : sign;
  const valueClass = absVal < 1e-9 ? "zero" : sign;
  const fillStyle = cls === "zero" ? "width:2px;" : `width:${pct.toFixed(1)}%;`;
  return `
    <div class="reward-bar-wrap">
      <div class="reward-bar">
        <div class="reward-bar-fill ${cls}" style="${fillStyle}"></div>
      </div>
      <span class="reward-bar-value ${valueClass}">${value}</span>
    </div>`;
}

function renderRewardEditor(preset, defaults, isEditable) {
  const categories = {};
  for (const [key, meta] of Object.entries(REWARD_META)) {
    const cat = meta.category;
    if (!categories[cat]) categories[cat] = [];
    const currentValue = (preset.values && preset.values[key] !== undefined)
      ? preset.values[key]
      : (defaults[key] !== undefined ? defaults[key] : 0);
    const isOverridden = preset.values && key in preset.values;
    categories[cat].push({ key, meta, value: currentValue, isOverridden });
  }

  const html = REWARD_CATEGORY_ORDER.map((cat) => {
    const rows = categories[cat] || [];
    if (!rows.length) return "";
    const rowsHtml = rows.map(({ key, meta, value, isOverridden }) => {
      const inputOrValue = isEditable
        ? `<input type="number" class="reward-row-input" data-key="${escapeHtml(key)}" value="${value}" step="0.01" />`
        : `<span class="reward-bar-value ${meta.sign}">${value}</span>`;
      const overrideMark = isOverridden ? ` <span style="color:var(--amber);font-size:11px;">●</span>` : "";
      return `
        <div class="reward-row">
          <div class="reward-row-meta">
            <div class="reward-row-label">${escapeHtml(meta.label)}${overrideMark}</div>
            <div class="reward-row-desc">${escapeHtml(meta.description)}</div>
            <div class="reward-row-varname">${escapeHtml(key)}</div>
          </div>
          ${rewardBarHtml(value, meta.sign)}
          <div>${inputOrValue}</div>
        </div>`;
    }).join("");
    return `
      <div class="reward-category">
        <div class="reward-category-header" onclick="toggleRewardCategory(this)">
          <span class="category-arrow">▼</span> ${escapeHtml(cat)}
        </div>
        <div class="reward-category-body">${rowsHtml}</div>
      </div>`;
  }).join("");

  $("#reward-categories").innerHTML = html;
  updateCategoryToggleButton("#reward-categories", "#preset-collapse-all-btn");
}

function toggleRewardCategory(header) {
  header.classList.toggle("collapsed");
  header.nextElementSibling.classList.toggle("collapsed");
  updateCategoryToggleButton("#reward-categories", "#preset-collapse-all-btn");
}

function allCategoryBodiesCollapsed(containerSelector) {
  const container = $(containerSelector);
  if (!container) return false;
  const bodies = Array.from(container.querySelectorAll(".reward-category-body"));
  return bodies.length > 0 && bodies.every((body) => body.classList.contains("collapsed"));
}

function updateCategoryToggleButton(containerSelector, buttonSelector) {
  const button = $(buttonSelector);
  const container = $(containerSelector);
  if (!button || !container) return;
  const hasCategories = container.querySelector(".reward-category-body") !== null;
  button.disabled = !hasCategories;
  button.textContent = allCategoryBodiesCollapsed(containerSelector) ? "Expand All" : "Collapse All";
}

function setCategoryGroupCollapsed(containerSelector, collapsed) {
  const container = $(containerSelector);
  if (!container) return;
  container.querySelectorAll(".reward-category").forEach((category) => {
    const header = category.querySelector(".reward-category-header");
    const body = category.querySelector(".reward-category-body");
    if (header) header.classList.toggle("collapsed", collapsed);
    if (body) body.classList.toggle("collapsed", collapsed);
  });
}

function toggleCategoryGroupCollapsed(containerSelector, buttonSelector) {
  setCategoryGroupCollapsed(containerSelector, !allCategoryBodiesCollapsed(containerSelector));
  updateCategoryToggleButton(containerSelector, buttonSelector);
}

function toggleRewardCategoriesCollapsed() {
  toggleCategoryGroupCollapsed("#reward-categories", "#preset-collapse-all-btn");
}

function renderPresets() {
  const { selectedPresetId } = state;
  const presets = rewardPresetsForRender();
  $("#preset-list").innerHTML = presets.map((p) => `
    <div class="preset-card ${p.id === selectedPresetId ? "selected" : ""} ${p.draft ? "draft-preset" : ""}"
         data-preset-id="${escapeHtml(p.id)}"
         title="${escapeHtml(p.description)}">
      <div class="preset-card-name">${escapeHtml(p.name)}${p.draft ? ` <span class="draft-badge">Draft</span>` : ""}</div>
      <div class="preset-card-desc">${escapeHtml(p.description)}</div>
    </div>`
  ).join("");
  document.querySelectorAll(".preset-card[data-preset-id]").forEach((card) => {
    card.addEventListener("click", () => selectPresetForEdit(card.dataset.presetId));
  });
  updateTrainingPresetIndicators();
}

function selectPresetForEdit(presetId) {
  const preset = rewardPresetById(presetId);
  if (!preset) return;
  state.selectedPresetId = presetId;
  renderPresets();

  $("#reward-editor-title").textContent = preset.name;
  $("#reward-editor-desc").textContent = preset.description;
  const nameInput = $("#reward-profile-name");
  const descInput = $("#reward-profile-description");
  if (nameInput) {
    nameInput.value = preset.name || "";
    nameInput.disabled = Boolean(preset.built_in);
  }
  if (descInput) {
    descInput.value = preset.description || "";
    descInput.disabled = Boolean(preset.built_in);
  }
  const builtInBadge = $("#preset-builtin-badge");
  builtInBadge.hidden = !preset.built_in && !preset.draft;
  builtInBadge.textContent = preset.draft ? "Unsaved Draft" : "Built-in";

  const activateBtn = $("#preset-activate-btn");
  if (activateBtn) {
    activateBtn.disabled = true;
    activateBtn.hidden = true;
  }

  $("#preset-collapse-all-btn").disabled = false;
  $("#preset-duplicate-btn").disabled = false;
  $("#preset-delete-btn").disabled = preset.built_in && !preset.draft;
  $("#preset-delete-btn").textContent = preset.draft ? "Discard Draft" : "Delete";
  $("#preset-save-btn").disabled = preset.built_in && !preset.draft;
  $("#preset-save-btn").textContent = preset.draft ? "Save as Preset" : "Save Preset";

  renderRewardEditor(preset, state.rewardDefaults, !preset.built_in || Boolean(preset.draft));
  updateTrainingPresetIndicators();
}

async function loadRewardsPage() {
  const [presetsData, tweakData] = await Promise.all([
    api("/api/presets"),
    api("/api/tweakables"),
  ]);
  state.presets = presetsData.presets || [];
  state.activePresetId = presetsData.active_preset_id || "baseline";
  state.rewardDefaults = tweakData.reward_defaults || {};
  // Keep backend active preset as the initial/default selection.
  const active = rewardPresetById(state.activePresetId);
  state.activePresetOverrides = active ? (active.values || {}) : {};

  renderPresets();

  // Render reference files section
  if (tweakData.files) {
    $("#tweak-files").innerHTML = tweakData.files.map((file) => `
      <article class="card">
        <strong>${escapeHtml(file.title)}</strong>
        <small>${escapeHtml(file.why)}</small>
        <small>${escapeHtml(file.absolute_path)}</small>
        <span class="pill">${file.exists ? "found" : "missing"}</span>
      </article>`).join("");
  }
  if (tweakData.reward_scales) {
    $("#reward-scales").innerHTML = tweakData.reward_scales.map((scale) => `
      <div class="scale-row">
        <div><strong>${escapeHtml(scale.name)}</strong><small>${escapeHtml(scale.relative_path)}:${escapeHtml(String(scale.line))}</small></div>
        <code>${escapeHtml(scale.value)}</code>
        <small>${escapeHtml(scale.comment || "No inline note yet.")}</small>
      </div>`).join("");
  }

  // Auto-select the backend active preset on first load; after that, selection drives training.
  if (state.selectedPresetId && rewardPresetById(state.selectedPresetId)) selectPresetForEdit(state.selectedPresetId);
  else if (state.activePresetId) selectPresetForEdit(state.activePresetId);
}

async function activatePreset(presetId) {
  if (state.rewardDraftPreset && presetId === state.rewardDraftPreset.id) {
    state.activePresetId = presetId;
    state.activePresetOverrides = state.rewardDraftPreset.values || {};
    renderPresets();
    selectPresetForEdit(presetId);
    return;
  }
  await api("/api/presets/activate", { method: "POST", body: JSON.stringify({ preset_id: presetId }) });
  state.activePresetId = presetId;
  const active = rewardPresetById(presetId);
  state.activePresetOverrides = active ? (active.values || {}) : {};
  renderPresets();
  if (state.selectedPresetId === presetId) selectPresetForEdit(presetId);
  await loadActivity();
}

async function duplicatePreset(sourcePresetId) {
  const source = rewardPresetById(sourcePresetId);
  if (!source) return;
  const name = window.prompt(`Name for the new preset (copy of ${source.name}):`, `${source.name} (copy)`);
  if (!name) return;
  const newPreset = await api("/api/presets", {
    method: "POST",
    body: JSON.stringify({ name, description: source.description, values: source.values }),
  });
  await loadRewardsPage();
  selectPresetForEdit(newPreset.id);
}

async function deletePreset(presetId) {
  const preset = rewardPresetById(presetId);
  if (!preset || preset.built_in) return;
  if (preset.draft) {
    state.rewardDraftPreset = null;
    state.selectedPresetId = state.activePresetId === presetId ? "baseline" : state.selectedPresetId;
    if (state.activePresetId === presetId) {
      state.activePresetId = "baseline";
      const baseline = rewardPresetById("baseline");
      state.activePresetOverrides = baseline ? (baseline.values || {}) : {};
    }
    renderPresets();
    selectPresetForEdit(state.selectedPresetId || state.activePresetId || "baseline");
    setStatus("Tweak draft discarded.");
    return;
  }
  if (!window.confirm(`Delete preset "${preset.name}"? This cannot be undone.`)) return;
  await api(`/api/presets/${encodeURIComponent(presetId)}/delete`, { method: "POST", body: JSON.stringify({}) });
  await loadRewardsPage();
  await loadActivity();
}

async function savePresetChanges(presetId) {
  const preset = rewardPresetById(presetId);
  if (!preset || preset.built_in) return;
  // Collect values from inputs
  const values = currentRewardEditorValues();
  if (preset.draft) {
    const wasActive = state.activePresetId === preset.id;
    const created = await api("/api/presets", {
      method: "POST",
      body: JSON.stringify({
        name: $("#reward-profile-name")?.value || preset.name,
        description: $("#reward-profile-description")?.value || "",
        values,
      }),
    });
    state.rewardDraftPreset = null;
    if (wasActive) {
      await api("/api/presets/activate", { method: "POST", body: JSON.stringify({ preset_id: created.id }) });
      state.activePresetId = created.id;
      state.activePresetOverrides = created.values || values;
    }
    state.selectedPresetId = created.id;
    await loadRewardsPage();
    selectPresetForEdit(created.id);
    await loadActivity();
    setStatus("Tweak draft saved as a preset.");
    return;
  }
  await api(`/api/presets/${encodeURIComponent(presetId)}/update`, {
    method: "POST",
    body: JSON.stringify({
      name: $("#reward-profile-name")?.value || preset.name,
      description: $("#reward-profile-description")?.value || "",
      values,
    }),
  });
  // Reload and re-select
  const presetData = await api("/api/presets");
  state.presets = presetData.presets;
  state.activePresetId = presetData.active_preset_id || state.activePresetId;
  const active = state.presets.find((p) => p.id === state.activePresetId);
  state.activePresetOverrides = active ? (active.values || {}) : {};
  const updated = state.presets.find((p) => p.id === presetId);
  if (updated) {
    renderPresets();
    selectPresetForEdit(presetId);
    await loadActivity();
  }
}

async function createNewPreset() {
  const name = window.prompt("New preset name:");
  if (!name || !name.trim()) return;
  const preset = await api("/api/presets", {
    method: "POST",
    body: JSON.stringify({ name: name.trim(), description: "", values: {} }),
  });
  await loadRewardsPage();
  selectPresetForEdit(preset.id);
  await loadActivity();
}

// ============================================================
// Terrain & Presets Page
// ============================================================

const TERRAIN_CATEGORY_ORDER = [
  "Importer", "Physics Material", "Curriculum", "Generator",
  "Flat", "Random Rough", "Wave", "Stairs", "Boxes",
];

function terrainMeta(key) {
  return state.terrainSchema.find((item) => item.key === key) || { key, label: key, category: "Other", type: "string" };
}

function terrainValueString(value) {
  if (Array.isArray(value)) return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "";
  return String(value);
}

function parseTerrainInput(input, meta) {
  if (meta.type === "bool") return input.checked;
  if (meta.type === "int") {
    const parsed = parseInt(input.value, 10);
    return Number.isNaN(parsed) ? 0 : parsed;
  }
  if (meta.type === "float") {
    const parsed = parseFloat(input.value);
    return Number.isNaN(parsed) ? 0 : parsed;
  }
  if (meta.type === "range" || meta.type === "list") {
    const raw = input.value.trim();
    if (!raw) return [];
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.map((item) => Number(item)) : parsed;
    } catch {
      return raw.split(",").map((item) => Number(item.trim())).filter((item) => !Number.isNaN(item));
    }
  }
  return input.value;
}

function terrainInputHtml(key, value, meta, isEditable) {
  const disabled = isEditable ? "" : "disabled";
  const valueText = escapeHtml(terrainValueString(value));
  if (!isEditable) return `<code class="terrain-value-code">${valueText}</code>`;
  if (meta.type === "bool") {
    return `<input class="terrain-row-input" data-key="${escapeHtml(key)}" data-type="bool" type="checkbox" ${value ? "checked" : ""} ${disabled} />`;
  }
  if (meta.type === "choice") {
    const choices = meta.choices || [];
    return `<select class="terrain-row-input" data-key="${escapeHtml(key)}" data-type="${escapeHtml(meta.type)}" ${disabled}>
      ${choices.map((choice) => `<option value="${escapeHtml(choice)}" ${String(value) === String(choice) ? "selected" : ""}>${escapeHtml(choice)}</option>`).join("")}
    </select>`;
  }
  if (meta.type === "int" || meta.type === "float") {
    const step = meta.step || (meta.type === "int" ? 1 : 0.01);
    return `<input class="terrain-row-input" data-key="${escapeHtml(key)}" data-type="${escapeHtml(meta.type)}" type="number" step="${escapeHtml(String(step))}" value="${valueText}" ${disabled} />`;
  }
  return `<input class="terrain-row-input terrain-wide-input" data-key="${escapeHtml(key)}" data-type="${escapeHtml(meta.type)}" value="${valueText}" ${disabled} />`;
}

function renderTerrainEditor(preset, defaults, schema, isEditable) {
  const categories = {};
  for (const meta of schema) {
    const cat = meta.category || "Other";
    if (!categories[cat]) categories[cat] = [];
    const key = meta.key;
    const currentValue = (preset.values && preset.values[key] !== undefined)
      ? preset.values[key]
      : (defaults[key] !== undefined ? defaults[key] : "");
    const isOverridden = preset.values && key in preset.values;
    categories[cat].push({ key, meta, value: currentValue, isOverridden });
  }
  const orderedCategories = [
    ...TERRAIN_CATEGORY_ORDER,
    ...Object.keys(categories).filter((cat) => !TERRAIN_CATEGORY_ORDER.includes(cat)).sort(),
  ];
  const html = orderedCategories.map((cat) => {
    const rows = categories[cat] || [];
    if (!rows.length) return "";
    const rowsHtml = rows.map(({ key, meta, value, isOverridden }) => {
      const overrideMark = isOverridden ? ` <span style="color:var(--amber);font-size:11px;">●</span>` : "";
      const valueLabel = isOverridden ? "Preset" : "Default";
      const valueText = terrainValueString(value);
      return `
        <div class="reward-row terrain-row">
          <div class="reward-row-meta">
            <div class="reward-row-label">${escapeHtml(meta.label || key)}${overrideMark}</div>
            <div class="reward-row-desc">${escapeHtml(meta.description || "")}</div>
            <div class="reward-row-varname">${escapeHtml(key)}</div>
          </div>
          <div class="terrain-control-cell">
            ${terrainInputHtml(key, value, meta, isEditable)}
            <div class="terrain-default-line">
              <span>${escapeHtml(valueLabel)}</span>
              <code class="terrain-value-code">${escapeHtml(valueText)}</code>
            </div>
          </div>
        </div>`;
    }).join("");
    return `
      <div class="reward-category">
        <div class="reward-category-header" onclick="toggleTerrainCategory(this)">
          <span class="category-arrow">▼</span> ${escapeHtml(cat)}
        </div>
        <div class="reward-category-body">${rowsHtml}</div>
      </div>`;
  }).join("");
  $("#terrain-categories").innerHTML = html;
  updateCategoryToggleButton("#terrain-categories", "#terrain-preset-collapse-all-btn");
}

function toggleTerrainCategory(header) {
  header.classList.toggle("collapsed");
  header.nextElementSibling.classList.toggle("collapsed");
  updateCategoryToggleButton("#terrain-categories", "#terrain-preset-collapse-all-btn");
}

function toggleTerrainCategoriesCollapsed() {
  toggleCategoryGroupCollapsed("#terrain-categories", "#terrain-preset-collapse-all-btn");
}

function renderTerrainPresets() {
  const { terrainPresets, selectedTerrainPresetId } = state;
  $("#terrain-preset-list").innerHTML = terrainPresets.map((p) => `
    <div class="preset-card ${p.id === selectedTerrainPresetId ? "selected" : ""}"
         data-terrain-preset-id="${escapeHtml(p.id)}"
         title="${escapeHtml(p.description)}">
      <div class="preset-card-name">${escapeHtml(p.name)}</div>
      <div class="preset-card-desc">${escapeHtml(p.description)}</div>
    </div>`
  ).join("");
  document.querySelectorAll(".preset-card[data-terrain-preset-id]").forEach((card) => {
    card.addEventListener("click", () => selectTerrainPresetForEdit(card.dataset.terrainPresetId));
  });
  updateTrainingPresetIndicators();
}

function selectTerrainPresetForEdit(presetId) {
  const preset = state.terrainPresets.find((p) => p.id === presetId);
  if (!preset) return;
  state.selectedTerrainPresetId = presetId;
  renderTerrainPresets();
  $("#terrain-editor-title").textContent = preset.name;
  $("#terrain-editor-desc").textContent = preset.description;
  const nameInput = $("#terrain-profile-name");
  const descInput = $("#terrain-profile-description");
  if (nameInput) {
    nameInput.value = preset.name || "";
    nameInput.disabled = Boolean(preset.built_in);
  }
  if (descInput) {
    descInput.value = preset.description || "";
    descInput.disabled = Boolean(preset.built_in);
  }
  $("#terrain-preset-builtin-badge").hidden = !preset.built_in;
  const activateBtn = $("#terrain-preset-activate-btn");
  if (activateBtn) {
    activateBtn.disabled = true;
    activateBtn.hidden = true;
  }
  $("#terrain-preset-collapse-all-btn").disabled = false;
  $("#terrain-preset-duplicate-btn").disabled = false;
  $("#terrain-preset-delete-btn").disabled = preset.built_in;
  $("#terrain-preset-save-btn").disabled = preset.built_in;
  renderTerrainEditor(preset, state.terrainDefaults, state.terrainSchema, !preset.built_in);
}

async function loadTerrainPage() {
  let presetsData;
  let terrainData;
  try {
    [presetsData, terrainData] = await Promise.all([
      api("/api/terrain/presets"),
      api("/api/terrain"),
    ]);
  } catch (error) {
    state.terrainPresets = [];
    state.terrainDefaults = {};
    state.terrainSchema = [];
    state.activeTerrainPresetOverrides = {};
    $("#terrain-preset-list").innerHTML = `<article class="empty-panel">Terrain API is unavailable. Restart the local panel so the backend reloads the terrain feature.</article>`;
    $("#terrain-categories").innerHTML = "";
    $("#terrain-files").innerHTML = "";
    $("#terrain-values").innerHTML = "";
    const activateBtn = $("#terrain-preset-activate-btn");
    if (activateBtn) activateBtn.disabled = true;
    $("#terrain-preset-collapse-all-btn").disabled = true;
    $("#terrain-preset-duplicate-btn").disabled = true;
    $("#terrain-preset-delete-btn").disabled = true;
    $("#terrain-preset-save-btn").disabled = true;
    setTerrainStatus(error.message);
    return;
  }
  setTerrainStatus("");
  state.terrainPresets = presetsData.presets || [];
  state.activeTerrainPresetId = presetsData.active_preset_id || "baseline";
  state.terrainDefaults = terrainData.terrain_defaults || {};
  state.terrainSchema = terrainData.field_schema || [];
  const active = state.terrainPresets.find((p) => p.id === state.activeTerrainPresetId);
  state.activeTerrainPresetOverrides = active ? (active.values || {}) : {};
  renderTerrainPresets();
  if (terrainData.files) {
    $("#terrain-files").innerHTML = terrainData.files.map((file) => `
      <article class="card">
        <strong>${escapeHtml(file.title)}</strong>
        <small>${escapeHtml(file.why)}</small>
        <small>${escapeHtml(file.absolute_path)}</small>
        <span class="pill">${file.exists ? "found" : "missing"}</span>
      </article>`).join("");
  }
  if (terrainData.terrain_values) {
    $("#terrain-values").innerHTML = terrainData.terrain_values.map((item) => `
      <div class="scale-row">
        <div><strong>${escapeHtml(item.key)}</strong><small>${escapeHtml(item.relative_path)}</small></div>
        <code>${escapeHtml(terrainValueString(item.value))}</code>
        <small>${escapeHtml(terrainMeta(item.key).category || "Terrain")}</small>
      </div>`).join("");
  }
  if (state.activeTerrainPresetId) selectTerrainPresetForEdit(state.activeTerrainPresetId);
}

async function activateTerrainPreset(presetId) {
  await api("/api/terrain/presets/activate", { method: "POST", body: JSON.stringify({ preset_id: presetId }) });
  state.activeTerrainPresetId = presetId;
  const active = state.terrainPresets.find((p) => p.id === presetId);
  state.activeTerrainPresetOverrides = active ? (active.values || {}) : {};
  renderTerrainPresets();
  if (state.selectedTerrainPresetId === presetId) selectTerrainPresetForEdit(presetId);
  await loadActivity();
}

async function duplicateTerrainPreset(sourcePresetId) {
  const source = state.terrainPresets.find((p) => p.id === sourcePresetId);
  if (!source) return;
  const name = window.prompt(`Name for the new terrain preset (copy of ${source.name}):`, `${source.name} (copy)`);
  if (!name) return;
  const newPreset = await api("/api/terrain/presets", {
    method: "POST",
    body: JSON.stringify({ name, description: source.description, values: source.values }),
  });
  await loadTerrainPage();
  selectTerrainPresetForEdit(newPreset.id);
}

async function deleteTerrainPreset(presetId) {
  const preset = state.terrainPresets.find((p) => p.id === presetId);
  if (!preset || preset.built_in) return;
  if (!window.confirm(`Delete terrain preset "${preset.name}"? This cannot be undone.`)) return;
  await api(`/api/terrain/presets/${encodeURIComponent(presetId)}/delete`, { method: "POST", body: JSON.stringify({}) });
  await loadTerrainPage();
  await loadActivity();
}

async function saveTerrainPresetChanges(presetId) {
  const preset = state.terrainPresets.find((p) => p.id === presetId);
  if (!preset || preset.built_in) return;
  const values = {};
  document.querySelectorAll("#terrain-categories .terrain-row-input").forEach((input) => {
    const key = input.dataset.key;
    if (!key) return;
    values[key] = parseTerrainInput(input, terrainMeta(key));
  });
  await api(`/api/terrain/presets/${encodeURIComponent(presetId)}/update`, {
    method: "POST",
    body: JSON.stringify({
      name: $("#terrain-profile-name")?.value || preset.name,
      description: $("#terrain-profile-description")?.value || "",
      values,
    }),
  });
  const presetData = await api("/api/terrain/presets");
  state.terrainPresets = presetData.presets;
  state.activeTerrainPresetId = presetData.active_preset_id || state.activeTerrainPresetId;
  const active = state.terrainPresets.find((p) => p.id === state.activeTerrainPresetId);
  state.activeTerrainPresetOverrides = active ? (active.values || {}) : {};
  const updated = state.terrainPresets.find((p) => p.id === presetId);
  if (updated) {
    renderTerrainPresets();
    selectTerrainPresetForEdit(presetId);
    await loadActivity();
    setTerrainStatus("Terrain preset saved.");
  }
}

async function createNewTerrainPreset() {
  const name = window.prompt("New terrain preset name:");
  if (!name || !name.trim()) return;
  const preset = await api("/api/terrain/presets", {
    method: "POST",
    body: JSON.stringify({ name: name.trim(), description: "", values: {} }),
  });
  await loadTerrainPage();
  selectTerrainPresetForEdit(preset.id);
  await loadActivity();
  setTerrainStatus(`Created terrain preset ${preset.name}.`);
}

// Run detail: reward config panel
function updateRewardCompareToggle() {
  document.querySelectorAll("#reward-compare-mode [data-compare-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.compareMode === state.rewardCompareMode);
  });
}

async function setRewardCompareMode(mode) {
  state.rewardCompareMode = mode === "previous" ? "previous" : "default";
  updateRewardCompareToggle();
  if (state.selectedRun && state.selectedRun.log_dir) {
    await loadRewardConfigForRun(state.selectedRun.id);
  }
}

async function loadRewardConfigForRun(runId) {
  const panel = $("#reward-config-panel");
  const content = $("#reward-config-content");
  if (!panel || !content) return;
  if (!state.selectedRun || state.selectedRun.id !== runId) return;
  updateRewardCompareToggle();
  try {
    const data = await api(`/api/runs/${encodeURIComponent(runId)}/reward-config?compare=${encodeURIComponent(state.rewardCompareMode)}`);
    if (!state.selectedRun || state.selectedRun.id !== runId || !findRun(runId)) {
      panel.hidden = true;
      content.innerHTML = "";
      return;
    }
    const baselineKind = data.baseline_kind || "default";
    const baselineLabel = data.baseline_label || (baselineKind === "previous" ? "last run" : "default");
    const baselineLine = baselineKind === "previous" && data.baseline_run_id
      ? `<p class="muted-copy">Compared with: <strong>${escapeHtml(baselineLabel)}</strong> <code>${escapeHtml(data.baseline_run_id)}</code></p>`
      : "";
    if (data.baseline_missing) {
      content.innerHTML = `<p class="muted-copy">No earlier run with saved reward config was found.</p>`;
      panel.hidden = false;
      return;
    }
    if (!data.changed || data.changed.length === 0) {
      content.innerHTML = `${baselineLine}<p class="muted-copy">All reward values match ${escapeHtml(baselineLabel)} for this run.</p>`;
      panel.hidden = false;
      return;
    }
    const presetLine = data.preset_id && data.preset_id !== "baseline"
      ? `<p class="muted-copy">Preset: <strong>${escapeHtml(data.preset_id)}</strong></p>`
      : "";
    const baselineName = baselineKind === "previous" ? "last run" : "default";
    const rows = data.changed.map((item) => {
      const meta = REWARD_META[item.name] || { label: item.name };
      const delta = item.delta_pct !== null ? item.delta_pct : null;
      const dir = delta !== null ? (delta > 0 ? "up" : "down") : "";
      const deltaHtml = delta !== null
        ? `<span class="diff-delta ${dir}">${delta > 0 ? "+" : ""}${delta}%</span>`
        : "";
      return `<div class="reward-diff-row">
        <span class="diff-name">${escapeHtml(meta.label || item.name)}</span>
        <span class="diff-value">${item.yaml_value}</span>
        <span class="diff-baseline">← ${escapeHtml(baselineName)}: ${item.default_value !== null ? item.default_value : "?"}</span>
        ${deltaHtml}
      </div>`;
    }).join("");
    content.innerHTML = presetLine + baselineLine + rows;
    panel.hidden = false;
  } catch {
    panel.hidden = true;
  }
}

async function loadTerrainConfigForRun(runId) {
  const panel = $("#terrain-config-panel");
  const content = $("#terrain-config-content");
  if (!panel || !content) return;
  if (!state.selectedRun || state.selectedRun.id !== runId) return;
  try {
    const data = await api(`/api/runs/${encodeURIComponent(runId)}/terrain-config`);
    if (!state.selectedRun || state.selectedRun.id !== runId || !findRun(runId)) {
      panel.hidden = true;
      content.innerHTML = "";
      return;
    }
    if (!data.changed || data.changed.length === 0) {
      content.innerHTML = `<p class="muted-copy">All terrain values are at default for this run.</p>`;
      panel.hidden = false;
      return;
    }
    const presetLine = data.preset_id && data.preset_id !== "baseline"
      ? `<p class="muted-copy">Preset: <strong>${escapeHtml(data.preset_id)}</strong></p>`
      : "";
    const rows = data.changed.map((item) => {
      const meta = terrainMeta(item.name);
      const delta = item.delta_pct !== null ? item.delta_pct : null;
      const dir = delta !== null ? (delta > 0 ? "up" : "down") : "";
      const deltaHtml = delta !== null
        ? `<span class="diff-delta ${dir}">${delta > 0 ? "+" : ""}${delta}%</span>`
        : "";
      return `<div class="reward-diff-row">
        <span class="diff-name">${escapeHtml(meta.label || item.name)}</span>
        <span class="diff-value">${escapeHtml(terrainValueString(item.yaml_value))}</span>
        <span class="diff-baseline">← default: ${escapeHtml(terrainValueString(item.default_value !== null ? item.default_value : "?"))}</span>
        ${deltaHtml}
      </div>`;
    }).join("");
    content.innerHTML = presetLine + rows;
    panel.hidden = false;
  } catch {
    panel.hidden = true;
  }
}

// ============================================================
// Run Comparison (Module 6)
// ============================================================

function startComparison(runId) {
  const run = findRun(runId);
  if (!run || !state.selectedRun || run.id === state.selectedRun.id) return;
  state.comparisonRun = run;
  state.comparisonMode = true;
  renderRunDetails();
  renderRuns();
}

function exitComparison() {
  state.comparisonRun = null;
  state.comparisonMode = false;
  renderRunDetails();
  renderRuns();
}

function comparisonRowHtml(label, valA, valB) {
  const same = String(valA ?? "—") === String(valB ?? "—");
  const diffClass = same ? "" : "comparison-diff";
  return `
    <div class="comparison-label">${escapeHtml(label)}</div>
    <div class="comparison-val ${valA !== valB && valA != null ? diffClass : ""}">${escapeHtml(String(valA ?? "—"))}</div>
    <div class="comparison-val ${valA !== valB && valB != null ? diffClass : ""}">${escapeHtml(String(valB ?? "—"))}</div>`;
}

function renderComparisonPanel(runA, runB) {
  const iterA = checkpointIteration(runA.latest_checkpoint);
  const iterB = checkpointIteration(runB.latest_checkpoint);
  const rows = [
    comparisonRowHtml("Status", runA.status, runB.status),
    comparisonRowHtml("Created", formatRelativeTime(runA.created_at), formatRelativeTime(runB.created_at)),
    comparisonRowHtml("Duration", formatDuration(runA.created_at, runA.updated_at), formatDuration(runB.created_at, runB.updated_at)),
    comparisonRowHtml("Task", runA.params?.task, runB.params?.task),
    comparisonRowHtml("Environments", runA.params?.num_envs, runB.params?.num_envs),
    comparisonRowHtml("Max Iterations", runA.params?.max_iterations, runB.params?.max_iterations),
    comparisonRowHtml("Checkpoint iter", iterA !== null ? iterA : "—", iterB !== null ? iterB : "—"),
    comparisonRowHtml("Reward preset", runA.reward_preset_id || "baseline", runB.reward_preset_id || "baseline"),
    comparisonRowHtml("Reward overrides", runA.reward_diff_count || 0, runB.reward_diff_count || 0),
    comparisonRowHtml("Terrain preset", runA.terrain_preset_id || "baseline", runB.terrain_preset_id || "baseline"),
    comparisonRowHtml("Terrain overrides", runA.terrain_diff_count || 0, runB.terrain_diff_count || 0),
    comparisonRowHtml("Return code", runA.returncode, runB.returncode),
    comparisonRowHtml("Has notes", runA.has_notes ? "Yes" : "No", runB.has_notes ? "Yes" : "No"),
    comparisonRowHtml("Has video", runA.has_video ? "Yes" : "No", runB.has_video ? "Yes" : "No"),
  ];
  const panel = document.querySelector(".details-panel");
  if (!panel) return;
  panel.innerHTML = `
    <div class="comparison-header">
      <h2>Comparing Runs</h2>
      <button type="button" id="exit-comparison-btn" class="ghost-button small-button">✕ Close Comparison</button>
    </div>
    <div class="comparison-grid">
      <div class="comparison-label comparison-col-header"></div>
      <div class="comparison-val comparison-col-header"><strong>${escapeHtml(runA.display_name || runA.id)}</strong></div>
      <div class="comparison-val comparison-col-header"><strong>${escapeHtml(runB.display_name || runB.id)}</strong></div>
      ${rows.join("")}
    </div>
    <div id="notes-status" class="status-line"></div>
  `;
  const exitBtn = $("#exit-comparison-btn");
  if (exitBtn) exitBtn.addEventListener("click", exitComparison);
}

// ============================================================
// Folder System (Module 3)
// ============================================================

function folderOptionsHtml() {
  const options = [`<option value="">— Uncategorized —</option>`];
  for (const folder of state.folders) {
    options.push(`<option value="${escapeHtml(folder)}">${escapeHtml(folder)}</option>`);
  }
  return options.join("");
}

function updateBulkToolbar() {
  const count = $("#bulk-selected-count");
  const move = $("#move-selected-runs");
  const clear = $("#clear-selected-runs");
  const deleteButton = $("#delete-selected-runs");
  const selectedCount = state.selectedRunIds.size;
  if (count) count.textContent = `${selectedCount} selected`;
  if (move) move.disabled = selectedCount === 0;
  if (clear) clear.disabled = selectedCount === 0;
  if (deleteButton) deleteButton.disabled = selectedCount === 0 || state.isBulkDeleting;
}

function toggleRunSelection(runId, checked) {
  if (!runId) return;
  if (checked) state.selectedRunIds.add(runId);
  else state.selectedRunIds.delete(runId);
  updateBulkToolbar();
}

function selectVisibleRuns() {
  for (const runId of visibleRunIds()) state.selectedRunIds.add(runId);
  renderRuns();
}

function clearRunSelection() {
  state.selectedRunIds.clear();
  renderRuns();
}

async function assignRunsToFolder(runIds, folderValue, options = {}) {
  const folder = folderValue === "" ? null : folderValue;
  const data = await api("/api/folders/assign", {
    method: "POST",
    body: JSON.stringify({ run_ids: runIds, folder }),
  });
  state.folders = data.folders || state.folders;
  if (options.clearSelection !== false) state.selectedRunIds.clear();
  await loadRuns();
  await loadFolders();
  const label = folder || "Uncategorized";
  setStatus(`Moved ${data.run_ids.length} run${data.run_ids.length !== 1 ? "s" : ""} to ${label}.`);
  return data;
}

async function moveSelectedRunsToFolder() {
  const runIds = [...state.selectedRunIds];
  if (!runIds.length) {
    setStatus("Select one or more runs first.");
    return;
  }
  await assignRunsToFolder(runIds, $("#bulk-folder-select")?.value || "");
}

async function loadFolders() {
  try {
    const data = await api("/api/folders");
    state.folders = data.folders || [];
  } catch {
    state.folders = [];
  }
  renderFolderSidebar();
  renderFolderOptions();
}

function renderFolderSidebar() {
  const sidebar = $("#folder-sidebar");
  if (!sidebar) return;
  const total = state.runs.length;
  const uncategorized = state.runs.filter((r) => !r.folder).length;
  const folderCounts = {};
  for (const folder of state.folders) {
    folderCounts[folder] = state.runs.filter((r) => r.folder === folder).length;
  }
  const uncatActive = state.activeFolder === "" ? "active" : "";
  const allActive = state.activeFolder === null ? "active" : "";
  const folderItems = state.folders
    .map((f) => {
      const active = state.activeFolder === f ? "active" : "";
      return `<div class="folder-item ${active}" data-folder="${escapeHtml(f)}">
        <span class="folder-name">${escapeHtml(f)}</span>
        <span class="folder-count">${folderCounts[f] || 0}</span>
        <button type="button" class="folder-rename-button" data-folder="${escapeHtml(f)}" data-tooltip="Rename folder">Rename</button>
        <button type="button" class="folder-delete-button" data-folder="${escapeHtml(f)}" data-tooltip="Remove folder">×</button>
      </div>`;
    })
    .join("");
  sidebar.innerHTML = `
    <button type="button" id="create-folder-btn" class="folder-create-button" data-tooltip="Create empty folder">
      <span class="folder-create-symbol">+</span>
      <span>New Folder</span>
    </button>
    <div class="folder-item ${uncatActive}" data-folder="__uncategorized__">
      <span class="folder-name">Uncategorized</span>
      <span class="folder-count">${uncategorized}</span>
    </div>
    <div class="folder-item ${allActive}" data-folder="__all__">
      <span class="folder-name">All Runs</span>
      <span class="folder-count">${total}</span>
    </div>
    ${folderItems}
  `;
  const createButton = sidebar.querySelector("#create-folder-btn");
  if (createButton) {
    createButton.addEventListener("click", (event) => {
      event.stopPropagation();
      promptCreateFolder().catch(handleActionError);
    });
  }
  sidebar.querySelectorAll(".folder-delete-button").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteFolder(button.dataset.folder).catch(handleActionError);
    });
  });
  sidebar.querySelectorAll(".folder-rename-button").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      promptRenameFolder(button.dataset.folder).catch(handleActionError);
    });
  });
  sidebar.querySelectorAll(".folder-item").forEach((item) => {
    item.addEventListener("click", () => {
      const raw = item.dataset.folder;
      if (raw === "__all__") state.activeFolder = null;
      else if (raw === "__uncategorized__") state.activeFolder = "";
      else state.activeFolder = raw;
      renderFolderSidebar();
      renderRuns();
    });
  });
}

async function deleteFolder(folderName) {
  const folder = String(folderName || "").trim();
  if (!folder) return;
  const count = state.runs.filter((run) => run.folder === folder).length;
  const message = count
    ? `Remove folder "${folder}"? ${count} run${count !== 1 ? "s" : ""} will move to Uncategorized.`
    : `Remove empty folder "${folder}"?`;
  if (!window.confirm(message)) return;
  const data = await api("/api/folders/delete", {
    method: "POST",
    body: JSON.stringify({ folder }),
  });
  state.folders = data.folders || state.folders.filter((item) => item !== folder);
  if (state.activeFolder === folder) state.activeFolder = "";
  await loadRuns();
  await loadFolders();
  await loadActivity();
  setStatus(`Removed folder ${folder}. Moved ${data.moved_count || 0} run${data.moved_count === 1 ? "" : "s"} to Uncategorized.`);
}

async function renameFolder(oldName, newName) {
  const data = await api("/api/folders/rename", {
    method: "POST",
    body: JSON.stringify({ old_name: oldName, new_name: newName }),
  });
  state.folders = data.folders || state.folders;
  if (state.activeFolder === oldName) state.activeFolder = data.new_folder;
  await loadRuns();
  await loadFolders();
  await loadActivity();
  setStatus(`Renamed folder ${data.old_folder} to ${data.new_folder}.`);
  return data;
}

async function promptRenameFolder(folderName) {
  const oldName = String(folderName || "").trim();
  if (!oldName) return;
  const nextName = window.prompt("Rename folder:", oldName);
  if (!nextName || !nextName.trim() || nextName.trim() === oldName) return;
  await renameFolder(oldName, nextName.trim());
}

function renderFolderOptions() {
  const sel = $("#run-folder-select");
  const bulkSel = $("#bulk-folder-select");
  if (sel) {
    const current = sel.value;
    sel.innerHTML = folderOptionsHtml();
    sel.value = current;
  }
  if (bulkSel) {
    const current = bulkSel.value;
    bulkSel.innerHTML = folderOptionsHtml();
    bulkSel.value = current;
  }
}

function renderFolderSelect(run) {
  const sel = $("#run-folder-select");
  if (!sel) return;
  sel.disabled = !run;
  renderFolderOptions();
  sel.value = run ? (run.folder || "") : "";
}

async function assignRunToFolder(folderValue) {
  if (!state.selectedRun) return;
  await assignRunsToFolder([state.selectedRun.id], folderValue, { clearSelection: false });
}

async function createFolder(folderName) {
  const data = await api("/api/folders", {
    method: "POST",
    body: JSON.stringify({ name: folderName }),
  });
  state.folders = data.folders || state.folders;
  await loadFolders();
  await loadActivity();
  setStatus(`Created folder ${data.folder}.`);
  return data.folder;
}

async function promptCreateFolder() {
  const name = window.prompt("New folder name:");
  if (!name || !name.trim()) return;
  const folder = await createFolder(name.trim());
  const bulkSelect = $("#bulk-folder-select");
  if (bulkSelect) bulkSelect.value = folder;
}

// ============================================================
// Activity Log
// ============================================================

function analyticsList(items) {
  if (!items || !items.length) return "No data yet";
  return items.slice(0, 3).map((item) => `${item[0]} (${item[1]})`).join(" · ");
}

function activityCard(label, value, detail) {
  return `
    <article class="activity-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value))}</strong>
      <small>${escapeHtml(detail || "")}</small>
    </article>
  `;
}

function activityWindowLabel(value) {
  return { today: "Today", "7d": "7 Days", "30d": "30 Days" }[value] || "7 Days";
}

function activityCategoryLabel(value) {
  if (!value) return "All Categories";
  const labels = {
    training: "Training",
    artifact: "Videos / ONNX",
    preset: "Presets",
    metadata: "Notes / Folders",
    admin: "Admin",
    system: "System",
  };
  return labels[value] || value;
}

function renderActivityControls(analytics) {
  const controls = $("#activity-controls");
  if (!controls) return;
  const leaderboard = analytics.leaderboard || [];
  controls.innerHTML = `
    <div class="segmented-control activity-window-control" aria-label="Activity time window">
      ${["today", "7d", "30d"].map((windowKey) => `
        <button type="button" data-activity-window="${windowKey}" class="${state.activityFilters.window === windowKey ? "active" : ""}">${activityWindowLabel(windowKey)}</button>
      `).join("")}
    </div>
    <label>Member
      <select id="activity-member-filter">
        <option value="">All members</option>
        ${leaderboard.map((member) => `<option value="${escapeHtml(member.actor_id || member.name)}" ${state.activityFilters.member === (member.actor_id || member.name) ? "selected" : ""}>${escapeHtml(member.name || "Unknown")}</option>`).join("")}
      </select>
    </label>
    <label>Category
      <select id="activity-category-filter">
        <option value="">All categories</option>
        ${["training", "artifact", "preset", "metadata", "admin", "system"].map((category) => `<option value="${category}" ${state.activityFilters.category === category ? "selected" : ""}>${activityCategoryLabel(category)}</option>`).join("")}
      </select>
    </label>
  `;
}

function activityBars(items, total) {
  if (!items || !items.length) return `<article class="empty-panel">No signal yet.</article>`;
  return `<div class="activity-bars">${items.slice(0, 8).map(([label, value]) => {
    const pct = total ? Math.max(4, Math.round((Number(value) / total) * 100)) : 0;
    return `
      <div class="activity-bar-row">
        <span>${escapeHtml(activityCategoryLabel(label) || label)}</span>
        <strong>${escapeHtml(String(value))}</strong>
        <div><i style="width: ${pct}%"></i></div>
      </div>
    `;
  }).join("")}</div>`;
}

function activityActorKey(event) {
  return String(event.actor_id || event.actor_name || event.actor_role || "Local panel");
}

function activityActorName(event) {
  return String(event.actor_name || event.actor_email || event.actor_role || "Local panel");
}

function activityEventDetail(event) {
  const payload = event.payload || {};
  return [
    event.subject_id || payload.run_id || payload.job_id || "",
    event.status || payload.status || event.outcome || "",
  ].filter(Boolean).join(" · ");
}

function groupActivityByActor(events) {
  const groups = new Map();
  for (const event of events) {
    const key = activityActorKey(event);
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        name: activityActorName(event),
        role: event.actor_role || "",
        points: 0,
        events: [],
        lastAt: event.created_at || "",
      });
    }
    const group = groups.get(key);
    group.events.push(event);
    group.points += Number(event.points || 0);
    if (String(event.created_at || "") > String(group.lastAt || "")) group.lastAt = event.created_at;
  }
  return Array.from(groups.values()).sort((a, b) => String(b.lastAt || "").localeCompare(String(a.lastAt || "")));
}

function renderActivityEvent(event) {
  const detail = activityEventDetail(event);
  const outcomeClass = event.outcome === "completed"
    ? "status-completed"
    : event.outcome === "failed" || event.outcome === "interrupted"
      ? "status-failed"
      : event.source === "remote"
        ? "status-running"
        : "muted-pill";
  return `
    <article class="activity-event ${event.source === "remote" ? "remote" : "local"}">
      <div>
        <strong>${escapeHtml(event.summary || event.event_type)}</strong>
        <small>${escapeHtml(activityActorName(event))} · ${escapeHtml(formatRelativeTime(event.created_at))}</small>
        ${detail ? `<small>${escapeHtml(detail)}</small>` : ""}
      </div>
      <span class="status-badge ${outcomeClass}">${escapeHtml(activityCategoryLabel(event.category) || event.category || event.source || "local")}</span>
    </article>
  `;
}

function renderActivityGroups(events) {
  if (!events.length) return `<article class="empty-panel">No activity recorded yet.</article>`;
  const groups = groupActivityByActor(events);
  return groups.map((group) => {
    const collapsed = state.activityCollapsedGroups.has(group.key);
    const completed = group.events.filter((event) => event.outcome === "completed").length;
    const failed = group.events.filter((event) => event.outcome === "failed" || event.outcome === "interrupted").length;
    return `
      <section class="activity-user-group ${collapsed ? "collapsed" : ""}">
        <button type="button" class="activity-user-summary" data-activity-group="${escapeHtml(group.key)}" aria-expanded="${collapsed ? "false" : "true"}">
          <span class="folder-chevron">${collapsed ? "+" : "-"}</span>
          <span>
            <strong>${escapeHtml(group.name || "Unknown member")}</strong>
            <small>${escapeHtml(group.role || "member")} · ${escapeHtml(String(group.events.length))} logs · ${escapeHtml(String(group.points))} pts</small>
          </span>
          <span class="activity-folder-stats">${escapeHtml(String(completed))} done · ${escapeHtml(String(failed))} failed</span>
        </button>
        <div class="activity-user-events">
          ${group.events.map(renderActivityEvent).join("")}
        </div>
      </section>
    `;
  }).join("");
}

function activityCategoryColor(label, index = 0) {
  const palette = {
    training: "#2563eb",
    artifact: "#059669",
    preset: "#7c3aed",
    metadata: "#d97706",
    admin: "#991b1b",
    system: "#64748b",
    completed: "#059669",
    failed: "#dc2626",
    interrupted: "#d97706",
    running: "#2563eb",
    queued: "#64748b",
    claimed: "#7c3aed",
    info: "#64748b",
  };
  return palette[label] || ["#2563eb", "#059669", "#7c3aed", "#d97706", "#dc2626", "#64748b"][index % 6];
}

function activityDonut(items, title) {
  const total = items.reduce((sum, item) => sum + Number(item[1] || 0), 0);
  if (!total) return `<article class="activity-chart-card"><h3>${escapeHtml(title)}</h3><p class="muted-copy">No data yet.</p></article>`;
  let cursor = 0;
  const stops = items.map(([label, value], index) => {
    const start = cursor;
    cursor += (Number(value) / total) * 100;
    return `${activityCategoryColor(label, index)} ${start.toFixed(2)}% ${cursor.toFixed(2)}%`;
  }).join(", ");
  return `
    <article class="activity-chart-card">
      <div class="activity-panel-head"><h3>${escapeHtml(title)}</h3></div>
      <div class="activity-donut-wrap">
        <div class="activity-donut" style="background: conic-gradient(${stops})">
          <strong>${escapeHtml(String(total))}</strong>
          <small>events</small>
        </div>
        <div class="activity-legend">
          ${items.slice(0, 6).map(([label, value], index) => `
            <span><i style="background:${activityCategoryColor(label, index)}"></i>${escapeHtml(activityCategoryLabel(label) || label)} <strong>${escapeHtml(String(value))}</strong></span>
          `).join("")}
        </div>
      </div>
    </article>
  `;
}

function activityTrendBuckets(events) {
  const days = state.activityFilters.window === "today" ? 12 : state.activityFilters.window === "30d" ? 15 : 7;
  const now = new Date();
  const buckets = Array.from({ length: days }, (_, index) => {
    const date = new Date(now);
    if (state.activityFilters.window === "today") {
      date.setHours(now.getHours() - (days - 1 - index), 0, 0, 0);
      return { key: date.toISOString().slice(0, 13), label: `${date.getHours()}:00`, value: 0 };
    }
    date.setDate(now.getDate() - (days - 1 - index));
    date.setHours(0, 0, 0, 0);
    return { key: date.toISOString().slice(0, 10), label: `${date.getMonth() + 1}/${date.getDate()}`, value: 0 };
  });
  const byKey = new Map(buckets.map((bucket) => [bucket.key, bucket]));
  for (const event of events) {
    const date = new Date(event.created_at || "");
    if (Number.isNaN(date.getTime())) continue;
    const key = state.activityFilters.window === "today" ? date.toISOString().slice(0, 13) : date.toISOString().slice(0, 10);
    const bucket = byKey.get(key);
    if (bucket) bucket.value += 1;
  }
  return buckets;
}

function activityTrendChart(events) {
  const buckets = activityTrendBuckets(events);
  const maxValue = Math.max(1, ...buckets.map((bucket) => bucket.value));
  return `
    <article class="activity-chart-card">
      <div class="activity-panel-head">
        <h3>Activity Rhythm</h3>
      </div>
      <div class="activity-spark-bars">
        ${buckets.map((bucket) => `
          <span title="${escapeHtml(bucket.label)} · ${escapeHtml(String(bucket.value))} events">
            <i style="height:${Math.max(6, Math.round((bucket.value / maxValue) * 100))}%"></i>
            <small>${escapeHtml(bucket.label)}</small>
          </span>
        `).join("")}
      </div>
    </article>
  `;
}

function activityContributionStack(leaderboard) {
  const total = leaderboard.reduce((sum, member) => sum + Number(member.points || 0), 0);
  if (!total) return `<article class="activity-chart-card"><h3>Contribution Share</h3><p class="muted-copy">No score yet.</p></article>`;
  return `
    <article class="activity-chart-card">
      <div class="activity-panel-head"><h3>Contribution Share</h3></div>
      <div class="activity-stack">
        ${leaderboard.slice(0, 6).map((member, index) => {
          const width = Math.max(5, Math.round((Number(member.points || 0) / total) * 100));
          return `<i style="width:${width}%;background:${activityCategoryColor(member.name, index)}" title="${escapeHtml(member.name || "Member")} · ${escapeHtml(String(member.points || 0))} pts"></i>`;
        }).join("")}
      </div>
      <div class="activity-legend compact">
        ${leaderboard.slice(0, 6).map((member, index) => `
          <span><i style="background:${activityCategoryColor(member.name, index)}"></i>${escapeHtml(member.name || "Member")} <strong>${escapeHtml(String(member.points || 0))}</strong></span>
        `).join("")}
      </div>
    </article>
  `;
}

function renderActivityCharts(analytics) {
  const leaderboard = analytics.leaderboard || [];
  return `
    <section class="activity-charts">
      ${activityContributionStack(leaderboard)}
      ${activityDonut(analytics.action_mix || [], "Action Orbit")}
      ${activityTrendChart(state.activityEvents || [])}
    </section>
  `;
}

function renderActivityMission(analytics) {
  const mission = $("#activity-mission");
  if (!mission) return;
  const leaderboard = analytics.leaderboard || [];
  const recentFailures = analytics.recent_failures || [];
  mission.innerHTML = `
    <section class="activity-panel activity-leaderboard">
      <div class="activity-panel-head">
        <div>
          <h3>Member Leaderboard</h3>
          <p class="muted-copy">Contribution mix over ${activityWindowLabel(state.activityFilters.window).toLowerCase()}.</p>
        </div>
      </div>
      ${leaderboard.length ? leaderboard.map((member, index) => `
        <article class="member-row">
          <span class="rank-chip">#${index + 1}</span>
          <div>
            <strong>${escapeHtml(member.name || "Unknown member")}</strong>
            <small>${escapeHtml(member.role || "member")} · ${escapeHtml(String(member.events || 0))} events</small>
          </div>
          <strong>${escapeHtml(String(member.points || 0))}</strong>
          <small>${escapeHtml(String(member.runs || 0))} runs · ${escapeHtml(String(member.completions || 0))} done · ${escapeHtml(String(member.failures || 0))} failed · ${escapeHtml(String(member.videos || 0))} videos</small>
        </article>
      `).join("") : `<article class="empty-panel">No member activity in this window.</article>`}
    </section>
    <section class="activity-panel">
      <div class="activity-panel-head">
        <h3>Experiment Mix</h3>
      </div>
      ${activityBars(analytics.action_mix || [], analytics.total_events || 0)}
    </section>
    <section class="activity-panel">
      <div class="activity-panel-head">
        <h3>Outcomes</h3>
      </div>
      ${activityBars(analytics.outcome_mix || [], analytics.total_events || 0)}
    </section>
    <section class="activity-panel">
      <div class="activity-panel-head">
        <h3>Team Pulse</h3>
      </div>
      ${recentFailures.length ? recentFailures.map((event) => `
        <article class="pulse-row">
          <strong>${escapeHtml(event.summary || event.event_type)}</strong>
          <small>${escapeHtml(event.actor_name || "Unknown")} · ${escapeHtml(formatRelativeTime(event.created_at))}</small>
        </article>
      `).join("") : `<article class="empty-panel">No recent failures or interruptions.</article>`}
    </section>
    ${renderActivityCharts(analytics)}
  `;
}

function renderActivity() {
  const analytics = state.activityAnalytics || {};
  renderActivityControls(analytics);
  const cards = $("#activity-analytics");
  if (cards) {
    const kpis = analytics.kpis || {};
    cards.innerHTML = [
      activityCard("Contribution", kpis.total_points || 0, "team points"),
      activityCard("Training Runs", kpis.training_runs || 0, `${kpis.success_rate || 0}% success`),
      activityCard("Videos / ONNX", kpis.artifacts_completed || 0, "completed artifacts"),
      activityCard("Active Members", kpis.active_members || 0, analyticsList(analytics.requests_by_member)),
    ].join("");
  }
  renderActivityMission(analytics);
  const events = $("#activity-events");
  if (!events) return;
  events.innerHTML = `
    <div class="activity-log-head">
      <div>
        <h3>Detailed Run Logs</h3>
        <p class="muted-copy">Grouped by user/account. Open a member folder to inspect the run-level timeline.</p>
      </div>
    </div>
    ${renderActivityGroups(state.activityEvents)}
  `;
}

async function loadActivity() {
  try {
    const params = new URLSearchParams({
      limit: "160",
      window: state.activityFilters.window,
    });
    if (state.activityFilters.member) params.set("member", state.activityFilters.member);
    if (state.activityFilters.category) params.set("category", state.activityFilters.category);
    const data = await api(`/api/activity?${params.toString()}`);
    state.activityEvents = data.events || [];
    state.activityAnalytics = data.analytics || {};
  } catch {
    state.activityEvents = [];
    state.activityAnalytics = {};
  }
  renderActivity();
}

async function refreshAll() {
  await Promise.all([loadSystem(), loadRemoteStatus(), loadConvergenceSettings(), loadRewardsPage(), loadTerrainPage(), loadActivity()]);
  await loadRuns();
  await loadFolders();
  if (state.selectedRun) setDebugTarget({ type: "run", id: state.selectedRun.id });
}

// ---------------------------------------------------------------------------
// Convergence Detection
// ---------------------------------------------------------------------------

const CONVERGENCE_PRESET_HINTS = {
  loose:   "Window: 100 iters · Threshold: 5% — triggers earlier, may be a short plateau",
  default: "Window: 200 iters · Threshold: 2% — balanced, good for most runs",
  strict:  "Window: 400 iters · Threshold: 1% — very conservative, fewer false positives",
  custom:  "Set your own window size and improvement threshold below",
};

async function loadConvergenceSettings() {
  try {
    const data = await api("/api/convergence/settings");
    renderConvergenceCard(data.config, data.presets);
  } catch (_) {
    // convergence API unavailable — leave card in default state
  }
}

function renderConvergenceCard(config, presets) {
  const enabledEl = $("#convergence-enabled");
  if (enabledEl) enabledEl.checked = Boolean(config.enabled);

  // Preset buttons
  document.querySelectorAll("#convergence-presets .segment-button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.preset === config.preset);
  });

  const hint = $("#convergence-preset-hint");
  if (hint) hint.textContent = CONVERGENCE_PRESET_HINTS[config.preset] || "";

  const customDiv = $("#convergence-custom-inputs");
  if (customDiv) customDiv.style.display = config.preset === "custom" ? "" : "none";

  const windowEl = $("#convergence-window");
  if (windowEl) windowEl.value = config.window_iterations;
  const threshEl = $("#convergence-threshold");
  if (threshEl) threshEl.value = config.min_improvement_pct;

  const autoRecEl = $("#convergence-auto-record");
  if (autoRecEl) autoRecEl.checked = Boolean(config.auto_record_video);

  const badge = $("#convergence-badge");
  if (badge) {
    if (!config.enabled) {
      badge.textContent = "Off";
      badge.className = "status-badge muted-pill";
    } else {
      badge.textContent = config.preset.charAt(0).toUpperCase() + config.preset.slice(1);
      badge.className = "status-badge info-pill";
    }
  }
}

async function saveConvergenceSettings() {
  const enabled = Boolean($("#convergence-enabled")?.checked);
  const preset = document.querySelector("#convergence-presets .segment-button.active")?.dataset.preset || "default";
  const updates = { enabled, preset, auto_record_video: Boolean($("#convergence-auto-record")?.checked) };
  if (preset === "custom") {
    const w = parseInt($("#convergence-window")?.value || "200", 10);
    const t = parseFloat($("#convergence-threshold")?.value || "2.0");
    if (!Number.isNaN(w)) updates.window_iterations = w;
    if (!Number.isNaN(t)) updates.min_improvement_pct = t;
  }
  const data = await api("/api/convergence/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  renderConvergenceCard(data.config, data.presets);
  setStatus("Convergence settings saved.");
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let amount = value / 1024;
  let index = 0;
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024;
    index += 1;
  }
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${units[index]}`;
}

function formatCompactPreview(preview) {
  const paths = preview.delete_paths.length
    ? preview.delete_paths.map((item) => `- model_${item.iteration}.pt (${formatBytes(item.bytes)}): ${item.path}`).join("\n")
    : "- No old checkpoints will be deleted.";
  return [
    "This permanently deletes old top-level model_*.pt checkpoints.",
    "",
    `Kept checkpoint: ${preview.kept_checkpoint}`,
    `Deleting: ${preview.delete_count} file(s), ${formatBytes(preview.bytes_to_free)} total`,
    "",
    paths,
    "",
    "Videos, TensorBoard logs, params, notes, and exported policy files are preserved.",
    "",
    `To confirm, type this exact run id: ${preview.id}`,
  ].join("\n");
}

async function compactSelectedRun() {
  if (!state.selectedRun) {
    setStatus("Select a run first.");
    return;
  }
  const runId = state.selectedRun.id;
  const preview = await api(`/api/runs/${encodeURIComponent(runId)}/compact-preview`);
  if (preview.delete_count === 0) {
    setStatus(`Nothing to compact. Keeping ${preview.kept_checkpoint}.`);
    return;
  }
  const confirmation = window.prompt(formatCompactPreview(preview), "");
  if (confirmation === null) {
    setStatus("Compact cancelled.");
    return;
  }
  const result = await api(`/api/runs/${encodeURIComponent(runId)}/compact`, {
    method: "POST",
    body: JSON.stringify({ confirmation }),
  });
  await loadRuns();
  await loadActivity();
  setStatus(`Compacted ${result.run_id}. Deleted ${result.deleted_paths.length} checkpoint(s), freed ${formatBytes(result.bytes_freed)}.`);
}

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});

applyTheme(preferredTheme());
$("#theme-toggle").addEventListener("click", toggleTheme);
$("#train-form").addEventListener("submit", startTraining);
$("#smoke-button").addEventListener("click", () => applyPreset("smoke"));
$("#debug-button").addEventListener("click", () => applyPreset("debug"));
$("#clear-resume").addEventListener("click", clearResume);
$("#refresh-button").addEventListener("click", () => refreshAll().catch((error) => setStatus(error.message)));
$("#save-name").addEventListener("click", () => saveName().catch(handleActionError));
$("#run-name").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    saveName().catch(handleActionError);
  }
});
$("#run-name").addEventListener("input", () => {
  if (!state.selectedRun) return;
  state.renameDirty = true;
  state.renameDraftRunId = state.selectedRun.id;
});
$("#save-notes").addEventListener("click", () => saveNotes().catch(handleActionError));
$("#delete-run").addEventListener("click", () => deleteSelectedRun().catch(handleActionError));
$("#compact-run").addEventListener("click", () => compactSelectedRun().catch(handleActionError));
document.querySelectorAll("#reward-compare-mode [data-compare-mode]").forEach((button) => {
  button.addEventListener("click", () => setRewardCompareMode(button.dataset.compareMode).catch(handleActionError));
});
$("#open-run-folder").addEventListener("click", () => openRunFolder().catch(handleActionError));
$("#export-onnx").addEventListener("click", () => exportOnnx().catch(handleActionError));
$("#copy-onnx-path").addEventListener("click", () => copyOnnxPath().catch(handleActionError));
$("#open-onnx-folder").addEventListener("click", () => openOnnxFolder().catch(handleActionError));
$("#record-video").addEventListener("click", () => recordVideo().catch(handleActionError));
$("#stop-recording").addEventListener("click", () => stopVideoRecording().catch(handleActionError));
$("#open-video-folder").addEventListener("click", () => openVideoFolder().catch(handleActionError));
$("#copy-video-path").addEventListener("click", () => copyVideoPath().catch(handleActionError));
$("#debug-refresh").addEventListener("click", refreshDebug);
$("#copy-debug").addEventListener("click", () => copyDebugOutput().catch(handleActionError));
$("#copy-command").addEventListener("click", () => copyLaunchCommand().catch(handleActionError));
$("#terminal-view").addEventListener("click", () => openTerminalView());
$("#open-process-log-folder").addEventListener("click", () => openProcessLogFolder().catch(handleActionError));
$("#stop-process").addEventListener("click", () => stopSelectedProcess().catch(handleActionError));
$("#resume-run").addEventListener("click", () => state.selectedRun && handleRunAction("resume", state.selectedRun.id));
$("#tweak-run").addEventListener("click", () => state.selectedRun && handleRunAction("tweak", state.selectedRun.id));
$("#tweak-last-run").addEventListener("click", tweakFromLastRun);
$("#play-run").addEventListener("click", () => {
  if (!state.selectedRun) return;
  const playProcessId = activeProcessIdForRun(state.selectedRun.id, "play");
  if (playProcessId) {
    handleRunAction("stop-play", state.selectedRun.id, playProcessId);
  } else {
    handleRunAction("play", state.selectedRun.id);
  }
});
$("#tensorboard-run").addEventListener("click", () => state.selectedRun && handleRunAction("tensorboard", state.selectedRun.id));

// Rewards page event listeners
const presetActivateBtn = $("#preset-activate-btn");
if (presetActivateBtn) {
  presetActivateBtn.addEventListener("click", () => {
    if (state.selectedPresetId) activatePreset(state.selectedPresetId).catch(handleActionError);
  });
}
$("#preset-duplicate-btn").addEventListener("click", () => {
  if (state.selectedPresetId) duplicatePreset(state.selectedPresetId).catch(handleActionError);
});
$("#preset-delete-btn").addEventListener("click", () => {
  if (state.selectedPresetId) deletePreset(state.selectedPresetId).catch(handleActionError);
});
$("#preset-save-btn").addEventListener("click", () => {
  if (state.selectedPresetId) savePresetChanges(state.selectedPresetId).catch(handleActionError);
});
$("#preset-collapse-all-btn").addEventListener("click", toggleRewardCategoriesCollapsed);

// Terrain page event listeners
const terrainPresetActivateBtn = $("#terrain-preset-activate-btn");
if (terrainPresetActivateBtn) {
  terrainPresetActivateBtn.addEventListener("click", () => {
    if (state.selectedTerrainPresetId) activateTerrainPreset(state.selectedTerrainPresetId).catch(handleActionError);
  });
}
$("#terrain-preset-duplicate-btn").addEventListener("click", () => {
  if (state.selectedTerrainPresetId) duplicateTerrainPreset(state.selectedTerrainPresetId).catch(handleActionError);
});
$("#terrain-preset-delete-btn").addEventListener("click", () => {
  if (state.selectedTerrainPresetId) deleteTerrainPreset(state.selectedTerrainPresetId).catch(handleActionError);
});
$("#terrain-preset-save-btn").addEventListener("click", () => {
  if (state.selectedTerrainPresetId) saveTerrainPresetChanges(state.selectedTerrainPresetId).catch(handleActionError);
});
$("#terrain-preset-collapse-all-btn").addEventListener("click", toggleTerrainCategoriesCollapsed);
// Search / filter / sort toolbar
const runSearch = $("#run-search");
const statusFilterEl = $("#status-filter");
const sortRunsEl = $("#sort-runs");
if (runSearch) runSearch.addEventListener("input", () => { state.searchQuery = runSearch.value; renderRuns(); });
if (statusFilterEl) statusFilterEl.addEventListener("change", () => { state.statusFilter = statusFilterEl.value; renderRuns(); });
if (sortRunsEl) sortRunsEl.addEventListener("change", () => { state.sortKey = sortRunsEl.value; renderRuns(); });

$("#new-preset-btn").addEventListener("click", () => createNewPreset().catch(handleActionError));
$("#new-terrain-preset-btn").addEventListener("click", () => createNewTerrainPreset().catch(handleActionError));
const newFolderBtn = $("#new-folder-btn");
if (newFolderBtn) newFolderBtn.addEventListener("click", () => promptCreateFolder().catch(handleActionError));
const folderSelect = $("#run-folder-select");
if (folderSelect) folderSelect.addEventListener("change", () => assignRunToFolder(folderSelect.value).catch(handleActionError));
const selectVisibleBtn = $("#select-visible-runs");
if (selectVisibleBtn) selectVisibleBtn.addEventListener("click", selectVisibleRuns);
const clearSelectedBtn = $("#clear-selected-runs");
if (clearSelectedBtn) clearSelectedBtn.addEventListener("click", clearRunSelection);
const moveSelectedBtn = $("#move-selected-runs");
if (moveSelectedBtn) moveSelectedBtn.addEventListener("click", () => moveSelectedRunsToFolder().catch(handleActionError));
const deleteSelectedBtn = $("#delete-selected-runs");
if (deleteSelectedBtn) deleteSelectedBtn.addEventListener("click", () => deleteSelectedRuns().catch(handleActionError));
const trainChangePreset = $("#train-change-preset");
if (trainChangePreset) trainChangePreset.addEventListener("click", () => setView("rewards"));
const trainChangeTerrainPreset = $("#train-change-terrain-preset");
if (trainChangeTerrainPreset) trainChangeTerrainPreset.addEventListener("click", () => setView("terrain"));
const activityRefresh = $("#activity-refresh");
if (activityRefresh) activityRefresh.addEventListener("click", () => loadActivity().catch(handleActionError));
document.addEventListener("click", (event) => {
  const groupButton = event.target.closest("[data-activity-group]");
  if (groupButton) {
    const key = groupButton.dataset.activityGroup || "";
    if (state.activityCollapsedGroups.has(key)) {
      state.activityCollapsedGroups.delete(key);
    } else {
      state.activityCollapsedGroups.add(key);
    }
    renderActivity();
    return;
  }
  const button = event.target.closest("[data-activity-window]");
  if (!button) return;
  state.activityFilters.window = button.dataset.activityWindow || "7d";
  loadActivity().catch(handleActionError);
});
document.addEventListener("change", (event) => {
  if (event.target.id === "activity-member-filter") {
    state.activityFilters.member = event.target.value;
    loadActivity().catch(handleActionError);
  }
  if (event.target.id === "activity-category-filter") {
    state.activityFilters.category = event.target.value;
    loadActivity().catch(handleActionError);
  }
});
const remoteAcceptToggle = $("#remote-accept-toggle");
if (remoteAcceptToggle) remoteAcceptToggle.addEventListener("change", () => saveRemoteAcceptance(remoteAcceptToggle.checked).catch(handleActionError));
const remoteWorkerStart = $("#remote-worker-start");
if (remoteWorkerStart) remoteWorkerStart.addEventListener("click", () => startRemoteWorker().catch(handleActionError));
const remoteWorkerStop = $("#remote-worker-stop");
if (remoteWorkerStop) remoteWorkerStop.addEventListener("click", () => stopRemoteWorker().catch(handleActionError));
const remoteWorkerRestart = $("#remote-worker-restart");
if (remoteWorkerRestart) remoteWorkerRestart.addEventListener("click", () => restartRemoteWorker().catch(handleActionError));
const remoteModeTmux = $("#remote-mode-tmux");
if (remoteModeTmux) remoteModeTmux.addEventListener("click", () => setRemoteWorkerMode("tmux").catch(handleActionError));
const remoteModeChild = $("#remote-mode-child");
if (remoteModeChild) remoteModeChild.addEventListener("click", () => setRemoteWorkerMode("child").catch(handleActionError));
const remoteAutostart = $("#remote-autostart");
if (remoteAutostart) remoteAutostart.addEventListener("change", () => setRemoteAutostart(remoteAutostart.checked).catch(handleActionError));
const copyWorkerAttachBtn = $("#copy-worker-attach");
if (copyWorkerAttachBtn) copyWorkerAttachBtn.addEventListener("click", () => copyWorkerAttach().catch(handleActionError));
const copyWorkerOutputBtn = $("#copy-worker-output");
if (copyWorkerOutputBtn) copyWorkerOutputBtn.addEventListener("click", () => copyWorkerOutput().catch(handleActionError));
const copyEnvPathBtn = $("#copy-env-path");
if (copyEnvPathBtn) copyEnvPathBtn.addEventListener("click", () => copyRemoteEnvPath().catch(handleActionError));
const copyPhoneUrlBtn = $("#copy-phone-url");
if (copyPhoneUrlBtn) copyPhoneUrlBtn.addEventListener("click", () => copyRemotePhoneUrl().catch(handleActionError));

// Convergence card
const convergenceSaveBtn = $("#convergence-save");
if (convergenceSaveBtn) convergenceSaveBtn.addEventListener("click", () => saveConvergenceSettings().catch(handleActionError));
document.querySelectorAll("#convergence-presets .segment-button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#convergence-presets .segment-button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const customDiv = $("#convergence-custom-inputs");
    if (customDiv) customDiv.style.display = btn.dataset.preset === "custom" ? "" : "none";
    const hint = $("#convergence-preset-hint");
    if (hint) hint.textContent = CONVERGENCE_PRESET_HINTS[btn.dataset.preset] || "";
  });
});

loadNotificationState();
renderNotificationBadges();
renderRunDetails();
updateBulkToolbar();
refreshAll().catch((error) => setStatus(error.message));
