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
  activeFolder: null,
  folders: [],
  selectedRunIds: new Set(),
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
  rewardDefaults: {},
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
    history: ["History", "Review runs, notes, checkpoints, TensorBoard, and playbacks."],
    access: ["Control Center", "Manage local access, V2.0 remote worker status, and remote launch acceptance."],
  };
  $("#view-title").textContent = titles[name][0];
  $("#view-subtitle").textContent = titles[name][1];
}

function formData(form) {
  const data = Object.fromEntries(new FormData(form).entries());
  data.headless = form.elements.headless.checked;
  data.resume = Boolean(data.checkpoint);
  data.num_envs = Number(data.num_envs);
  data.max_iterations = Number(data.max_iterations);
  // Include active reward preset
  data.reward_preset_id = state.activePresetId || "baseline";
  data.reward_overrides = state.activePresetOverrides || {};
  return data;
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

async function loadRemoteStatus() {
  const status = await api("/api/remote/status");
  const badge = $("#remote-config-badge");
  if (badge) {
    badge.textContent = status.configured ? "Configured" : "Needs Setup";
    badge.className = status.configured ? "status-badge status-completed" : "status-badge status-interrupted";
  }
  const accept = $("#remote-accept-status");
  if (accept) {
    accept.textContent = status.accept_jobs
      ? "Remote queued jobs are accepted by this machine."
      : "Remote queued jobs are paused on this machine.";
  }
  const workerCommand = $("#remote-worker-command");
  if (workerCommand) workerCommand.textContent = status.worker_command;
  const tunnelCommand = $("#remote-tunnel-command");
  if (tunnelCommand) tunnelCommand.textContent = status.cloudflare_tunnel_command;
  renderKvGrid("#remote-machine-grid", [
    ["Machine ID", status.machine_id || "-"],
    ["Version", status.version || "-"],
    ["Accept Jobs", status.accept_jobs ? "enabled" : "disabled"],
    ["Active Processes", status.active_process_count || 0],
    ["Isaac/GPU Lock", status.active_isaac_process_count ? "busy" : "free"],
  ]);
  renderKvGrid("#remote-integrations-grid", [
    ["Supabase", status.configured ? status.supabase_url : "not configured"],
    ["Cloudflare", status.cloudflare_tunnel_host || "not configured"],
    ["Discord", status.discord_configured ? "configured" : "not configured"],
    ["Email", status.email_configured ? "configured" : "not configured"],
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
  const normalized = String(status || "unknown").toLowerCase();
  if (normalized === "completed") return "status-completed";
  if (normalized === "running" || normalized === "stopping") return "status-running";
  if (normalized === "failed") return "status-failed";
  if (normalized === "interrupted") return "status-interrupted";
  return "status-unknown";
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

function mediaLockMessage(process) {
  if (!process) return "";
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
  const mediaProcess = activeMediaProcess();
  $("#runs").innerHTML = runs
    .map((run) => {
      const active = state.selectedRun && state.selectedRun.id === run.id ? "active" : "";
      const title = run.display_name || run.id;
      const canTensorboard = Boolean(run.log_dir);
      const canCheckpoint = Boolean(run.latest_checkpoint);
      const playProcessId = activeProcessIdForRun(run.id, "play");
      const videoProcessId = activeProcessIdForRun(run.id, "video");
      const onnxProcessId = activeProcessIdForRun(run.id, "onnx");
      const paramSummary = runParamSummary(run);
      const timeSummary = runTimeSummary(run);
      const videoText = videoProcessId ? "recording video" : videoSummary(run);
      const onnxText = onnxProcessId ? "exporting ONNX" : onnxSummary(run);
      const selected = state.selectedRunIds.has(run.id) ? "checked" : "";
      const playAction = playProcessId ? "stop-play" : "play";
      const playLabel = playProcessId ? "Stop Play" : "Play";
      const playProcessAttr = playProcessId ? `data-process-id="${escapeHtml(playProcessId)}"` : "";
      const playDisabled = (!canCheckpoint && !playProcessId) || Boolean(mediaProcess && !playProcessId);
      const unread = state.notifications.unreadRunIds.has(run.id);
      return `
        <article class="run-card ${active} ${unread ? "unread" : ""}" data-run-id="${escapeHtml(run.id)}">
          <input class="run-select-checkbox" type="checkbox" data-run-id="${escapeHtml(run.id)}" ${selected} aria-label="Select ${escapeHtml(title)} for folder move" data-tooltip="Select for folder move">
          <div class="run-top">
            <div class="run-title">
              ${unread ? `<span class="unread-dot" data-tooltip="Unread history update"></span>` : ""}
              <strong>${escapeHtml(title)}</strong>
            </div>
            <span class="pill status-pill ${statusClass(run.status)}">${escapeHtml(run.status || "unknown")}</span>
          </div>
          ${paramSummary ? `<small>${escapeHtml(paramSummary)}</small>` : ""}
          ${timeSummary ? `<small>${escapeHtml(timeSummary)}</small>` : ""}
          <small>${escapeHtml(runLogSummary(run))}</small>
          ${run.reward_preset_id && run.reward_preset_id !== "baseline"
            ? `<small><span class="reward-diff-badge">preset: ${escapeHtml(run.reward_preset_id)}</span></small>`
            : run.reward_diff_count > 0
              ? `<small><span class="reward-diff-badge">${escapeHtml(String(run.reward_diff_count))} reward override${run.reward_diff_count !== 1 ? "s" : ""}</span></small>`
              : ""}
          <small>${escapeHtml(checkpointSummary(run))}${videoText ? ` · ${escapeHtml(videoText)}` : ""}${onnxText ? ` · ${escapeHtml(onnxText)}` : ""}${escapeHtml(runStatusDetail(run))}${run.has_notes ? " <strong>+ notes</strong>" : ""}</small>
          <div class="run-actions">
            <button type="button" data-action="tensorboard" data-run-id="${escapeHtml(run.id)}" ${runButtonDisabled(!canTensorboard)} data-tooltip="Open metrics">TensorBoard</button>
            <button type="button" data-action="${playAction}" data-run-id="${escapeHtml(run.id)}" ${playProcessAttr} ${runButtonDisabled(playDisabled)} data-tooltip="${playProcessId ? "Stop Isaac playback" : "Play checkpoint"}">${escapeHtml(playLabel)}</button>
            <button type="button" data-action="resume" data-run-id="${escapeHtml(run.id)}" ${runButtonDisabled(!canCheckpoint)} data-tooltip="Resume training from checkpoint">Resume to Train</button>
            <button type="button" data-action="console" data-run-id="${escapeHtml(run.id)}" data-tooltip="Show Process Console">Console</button>
            ${videoProcessId
              ? `<button type="button" data-action="stop-video" data-run-id="${escapeHtml(run.id)}" data-process-id="${escapeHtml(videoProcessId)}" data-tooltip="Stop recording">Stop Recording</button>`
              : ""}
            ${state.selectedRun && state.selectedRun.id !== run.id
              ? `<button type="button" data-action="compare" data-run-id="${escapeHtml(run.id)}" data-tooltip="Compare with selected">Compare</button>`
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
  const mediaProcess = activeMediaProcess();
  const videoProcessId = activeVideoProcessId(run);
  $("#record-video").disabled = !hasCheckpoint || Boolean(mediaProcess);
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
  message.textContent = "Recording failed. Use the Process Console for logs and the tmux attach command.";
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
  const mediaProcess = activeMediaProcess();

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
  } else if (!(state.renameDirty && state.renameDraftRunId === run.id)) {
    runName.value = run.display_name || "";
  }
  runName.disabled = !run;

  // Folder select
  renderFolderSelect(run);

  // Inputs
  $("#notes-editor").disabled = !run;
  $("#save-name").disabled = !run;
  $("#save-notes").disabled = !run;

  // Action buttons
  $("#delete-run").disabled = !run;
  $("#compact-run").disabled = !run || !run.log_dir || Boolean(run && activeProcessForRun(run.id));
  $("#open-run-folder").disabled = !run || !run.log_dir;
  $("#tensorboard-run").disabled = !run || !run.log_dir;
  $("#play-run").disabled = !run || (!run.latest_checkpoint && !playProcessId) || Boolean(mediaProcess && !playProcessId);
  $("#play-run").textContent = playProcessId ? "Stop Play" : "Play";
  $("#export-onnx").disabled = !run || !run.latest_checkpoint || Boolean(mediaProcess);
  $("#export-onnx").textContent = onnxProcessId ? "Exporting ONNX" : "Export ONNX";
  $("#copy-onnx-path").hidden = !run || !run.onnx_path;
  $("#copy-onnx-path").disabled = !run || !run.onnx_path;
  $("#open-onnx-folder").hidden = !run || !run.onnx_path;
  $("#open-onnx-folder").disabled = !run || !run.onnx_path;
  $("#resume-run").disabled = !run || !run.latest_checkpoint;
  $("#stop-process").disabled = !state.debugTarget && !run;

  const hasAttach = Boolean(state.lastDebug && state.lastDebug.attach_command);
  $("#copy-attach").hidden = !hasAttach;
  $("#copy-attach").disabled = !hasAttach;
  $("#open-process-log-folder").disabled = !state.lastDebug || !(state.lastDebug.process_log || state.lastDebug.log_file);

  renderVideoPanel(run);
}

function hasActiveRun() {
  return (
    Object.keys(state.activeProcessMap).length > 0 ||
    state.runs.some((run) => run.status === "running" || run.status === "stopping" || run.video_status === "recording")
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
  state.selectedRun = selectedId ? findRun(selectedId) || null : state.selectedRun;
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
  const [notesData] = await Promise.all([
    api(`/api/runs/${encodeURIComponent(runId)}/notes`),
    run.log_dir ? loadRewardConfigForRun(runId) : Promise.resolve(),
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

async function copyAttachCommand() {
  const command = state.lastDebug && state.lastDebug.attach_command;
  if (!command) {
    setStatus("No tmux attach command is available for this process.");
    return;
  }
  await copyText(command);
  setStatus(`Attach command copied: ${command}`);
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
  if (debug.attach_command) rows.push(["Attach", debug.attach_command]);
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
  $("#debug-command").textContent = debug.command || "";
  $("#debug-log").textContent = logTail || debug.debug_hint || "No terminal output captured yet.";
  $("#debug-log").scrollTop = $("#debug-log").scrollHeight;
  renderRunDetails();
}

async function refreshDebug() {
  if (!state.debugTarget) return;
  try {
    const debug = await api(debugEndpoint(state.debugTarget));
    renderDebug(debug);
    if (!isLiveDebug(debug) && state.debugTarget.type === "process") stopDebugPolling();
  } catch (error) {
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

async function startTraining(event) {
  event.preventDefault();
  $("#train-status").textContent = "Starting training...";
  try {
    const run = await api("/api/training/start", {
      method: "POST",
      body: JSON.stringify(formData($("#train-form"))),
    });
    $("#train-status").textContent = `Started ${run.id} with pid ${run.pid}`;
    markHistoryUnread(run.id);
    await loadRuns();
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
  const mediaProcess = activeMediaProcess();
  if (mediaProcess) {
    setStatus(mediaLockMessage(mediaProcess));
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
  const mediaProcess = activeMediaProcess();
  if (mediaProcess) {
    setStatus(mediaLockMessage(mediaProcess));
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
  const mediaProcess = activeMediaProcess();
  if (mediaProcess) {
    setStatus(mediaLockMessage(mediaProcess));
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
    `To confirm, type this exact run id: ${preview.id}`,
  ].join("\n");
}

async function deleteSelectedRun() {
  if (!state.selectedRun) {
    setStatus("Select a run first.");
    return;
  }
  const runId = state.selectedRun.id;
  const preview = await api(`/api/runs/${encodeURIComponent(runId)}/delete-preview`);
  const confirmation = window.prompt(formatDeletePreview(preview), "");
  if (confirmation === null) {
    setStatus("Delete cancelled.");
    return;
  }
  const result = await api(`/api/runs/${encodeURIComponent(runId)}/delete`, {
    method: "POST",
    body: JSON.stringify({ confirmation, delete_logs: true }),
  });
  state.selectedRun = null;
  state.debugTarget = null;
  state.lastDebug = null;
  $("#notes-editor").value = "";
  $("#debug-command").textContent = "";
  $("#debug-log").textContent = "";
  $("#debug-status").textContent = "";
  renderVideoPanel(null);
  await loadRuns();
  setStatus(`Deleted ${result.run_id}. Removed ${result.deleted_paths.length} log/note path(s).`);
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
    if (!state.selectedRun || state.selectedRun.id !== runId) {
      await selectRun(runId);
    }
    if (action === "tensorboard") await startTensorBoardForRun(runId, pendingWindow);
    if (action === "play") await playRun(runId);
    if (action === "resume") resumeRun(runId);
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
}

function toggleRewardCategory(header) {
  header.classList.toggle("collapsed");
  header.nextElementSibling.classList.toggle("collapsed");
}

function renderPresets() {
  const { presets, activePresetId, selectedPresetId } = state;
  $("#preset-list").innerHTML = presets.map((p) => `
    <div class="preset-card ${p.id === selectedPresetId ? "selected" : ""} ${p.id === activePresetId ? "active-for-training" : ""}"
         data-preset-id="${escapeHtml(p.id)}"
         title="${escapeHtml(p.description)}">
      <div class="preset-card-name">${escapeHtml(p.name)}</div>
      <div class="preset-card-desc">${escapeHtml(p.description)}</div>
    </div>`
  ).join("");
  document.querySelectorAll(".preset-card").forEach((card) => {
    card.addEventListener("click", () => selectPresetForEdit(card.dataset.presetId));
  });
  // Update training form indicator
  const activePreset = presets.find((p) => p.id === activePresetId) || { name: activePresetId };
  const el = $("#train-active-preset-name");
  if (el) el.textContent = activePreset.name || activePresetId;
}

function selectPresetForEdit(presetId) {
  const preset = state.presets.find((p) => p.id === presetId);
  if (!preset) return;
  state.selectedPresetId = presetId;
  renderPresets();

  $("#reward-editor-title").textContent = preset.name;
  $("#reward-editor-desc").textContent = preset.description;
  const builtInBadge = $("#preset-builtin-badge");
  builtInBadge.hidden = !preset.built_in;

  const isActive = preset.id === state.activePresetId;
  const activateBtn = $("#preset-activate-btn");
  activateBtn.disabled = false;
  activateBtn.textContent = isActive ? "✓ Active for Training" : "Use for Training";
  activateBtn.style.fontWeight = isActive ? "900" : "";

  $("#preset-duplicate-btn").disabled = false;
  $("#preset-delete-btn").disabled = preset.built_in;
  $("#preset-save-btn").disabled = preset.built_in;

  renderRewardEditor(preset, state.rewardDefaults, !preset.built_in);
}

async function loadRewardsPage() {
  const [presetsData, tweakData] = await Promise.all([
    api("/api/presets"),
    api("/api/tweakables"),
  ]);
  state.presets = presetsData.presets || [];
  state.activePresetId = presetsData.active_preset_id || "baseline";
  state.rewardDefaults = tweakData.reward_defaults || {};
  // Populate active preset overrides for training form
  const active = state.presets.find((p) => p.id === state.activePresetId);
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

  // Auto-select the active preset for display
  if (state.activePresetId) selectPresetForEdit(state.activePresetId);
}

async function activatePreset(presetId) {
  await api("/api/presets/activate", { method: "POST", body: JSON.stringify({ preset_id: presetId }) });
  state.activePresetId = presetId;
  const active = state.presets.find((p) => p.id === presetId);
  state.activePresetOverrides = active ? (active.values || {}) : {};
  renderPresets();
  if (state.selectedPresetId === presetId) selectPresetForEdit(presetId);
}

async function duplicatePreset(sourcePresetId) {
  const source = state.presets.find((p) => p.id === sourcePresetId);
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
  const preset = state.presets.find((p) => p.id === presetId);
  if (!preset || preset.built_in) return;
  if (!window.confirm(`Delete preset "${preset.name}"? This cannot be undone.`)) return;
  await api(`/api/presets/${encodeURIComponent(presetId)}/delete`, { method: "POST", body: JSON.stringify({}) });
  await loadRewardsPage();
}

async function savePresetChanges(presetId) {
  const preset = state.presets.find((p) => p.id === presetId);
  if (!preset || preset.built_in) return;
  // Collect values from inputs
  const values = {};
  document.querySelectorAll("#reward-categories .reward-row-input").forEach((input) => {
    const key = input.dataset.key;
    const val = parseFloat(input.value);
    if (key && !Number.isNaN(val)) values[key] = val;
  });
  await api(`/api/presets/${encodeURIComponent(presetId)}/update`, {
    method: "POST",
    body: JSON.stringify({ values }),
  });
  // Reload and re-select
  state.presets = (await api("/api/presets")).presets;
  const updated = state.presets.find((p) => p.id === presetId);
  if (updated) {
    renderPresets();
    selectPresetForEdit(presetId);
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
}

// Run detail: reward config panel
async function loadRewardConfigForRun(runId) {
  const panel = $("#reward-config-panel");
  const content = $("#reward-config-content");
  if (!panel || !content) return;
  try {
    const data = await api(`/api/runs/${encodeURIComponent(runId)}/reward-config`);
    if (!data.changed || data.changed.length === 0) {
      content.innerHTML = `<p class="muted-copy">All reward values are at default for this run.</p>`;
      panel.hidden = false;
      return;
    }
    const presetLine = data.preset_id && data.preset_id !== "baseline"
      ? `<p class="muted-copy">Preset: <strong>${escapeHtml(data.preset_id)}</strong></p>`
      : "";
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
        <span class="diff-baseline">← default: ${item.default_value !== null ? item.default_value : "?"}</span>
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
  const selectedCount = state.selectedRunIds.size;
  if (count) count.textContent = `${selectedCount} selected`;
  if (move) move.disabled = selectedCount === 0;
  if (clear) clear.disabled = selectedCount === 0;
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
  const allActive = state.activeFolder === null ? "active" : "";
  const uncatActive = state.activeFolder === "" ? "active" : "";
  const folderItems = state.folders
    .map((f) => {
      const active = state.activeFolder === f ? "active" : "";
      return `<div class="folder-item ${active}" data-folder="${escapeHtml(f)}">
        <span class="folder-name">${escapeHtml(f)}</span>
        <span class="folder-count">${folderCounts[f] || 0}</span>
        <button type="button" class="folder-delete-button" data-folder="${escapeHtml(f)}" data-tooltip="Remove folder">×</button>
      </div>`;
    })
    .join("");
  sidebar.innerHTML = `
    <button type="button" id="create-folder-btn" class="folder-create-button" data-tooltip="Create empty folder">
      <span class="folder-create-symbol">+</span>
      <span>New Folder</span>
    </button>
    <div class="folder-item ${allActive}" data-folder="__all__">
      <span class="folder-name">All Runs</span>
      <span class="folder-count">${total}</span>
    </div>
    <div class="folder-item ${uncatActive}" data-folder="__uncategorized__">
      <span class="folder-name">Uncategorized</span>
      <span class="folder-count">${uncategorized}</span>
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
  setStatus(`Removed folder ${folder}. Moved ${data.moved_count || 0} run${data.moved_count === 1 ? "" : "s"} to Uncategorized.`);
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

async function refreshAll() {
  await Promise.all([loadSystem(), loadRemoteStatus(), loadRewardsPage()]);
  await loadRuns();
  await loadFolders();
  if (state.selectedRun) setDebugTarget({ type: "run", id: state.selectedRun.id });
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
$("#copy-attach").addEventListener("click", () => copyAttachCommand().catch(handleActionError));
$("#terminal-view").addEventListener("click", () => openTerminalView());
$("#open-process-log-folder").addEventListener("click", () => openProcessLogFolder().catch(handleActionError));
$("#stop-process").addEventListener("click", () => stopSelectedProcess().catch(handleActionError));
$("#resume-run").addEventListener("click", () => state.selectedRun && handleRunAction("resume", state.selectedRun.id));
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
$("#preset-activate-btn").addEventListener("click", () => {
  if (state.selectedPresetId) activatePreset(state.selectedPresetId).catch(handleActionError);
});
$("#preset-duplicate-btn").addEventListener("click", () => {
  if (state.selectedPresetId) duplicatePreset(state.selectedPresetId).catch(handleActionError);
});
$("#preset-delete-btn").addEventListener("click", () => {
  if (state.selectedPresetId) deletePreset(state.selectedPresetId).catch(handleActionError);
});
$("#preset-save-btn").addEventListener("click", () => {
  if (state.selectedPresetId) savePresetChanges(state.selectedPresetId).catch(handleActionError);
});
// Search / filter / sort toolbar
const runSearch = $("#run-search");
const statusFilterEl = $("#status-filter");
const sortRunsEl = $("#sort-runs");
if (runSearch) runSearch.addEventListener("input", () => { state.searchQuery = runSearch.value; renderRuns(); });
if (statusFilterEl) statusFilterEl.addEventListener("change", () => { state.statusFilter = statusFilterEl.value; renderRuns(); });
if (sortRunsEl) sortRunsEl.addEventListener("change", () => { state.sortKey = sortRunsEl.value; renderRuns(); });

$("#new-preset-btn").addEventListener("click", () => createNewPreset().catch(handleActionError));
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
const trainChangePreset = $("#train-change-preset");
if (trainChangePreset) trainChangePreset.addEventListener("click", () => setView("rewards"));
const remoteEnable = $("#remote-enable");
if (remoteEnable) remoteEnable.addEventListener("click", () => saveRemoteAcceptance(true).catch(handleActionError));
const remoteDisable = $("#remote-disable");
if (remoteDisable) remoteDisable.addEventListener("click", () => saveRemoteAcceptance(false).catch(handleActionError));

loadNotificationState();
renderNotificationBadges();
renderRunDetails();
updateBulkToolbar();
refreshAll().catch((error) => setStatus(error.message));
