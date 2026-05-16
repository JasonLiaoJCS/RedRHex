import { DEFAULT_MACHINE_ID, SUPABASE_URL } from "./config.js";
import {
  createSignedVideoUrl,
  currentUser,
  insert,
  invokeFunction,
  loadRemoteSnapshot,
  remove,
  select,
  signIn,
  signOut,
  update,
  upsert,
} from "./api.js";
import {
  REWARD_FIELDS,
  TERRAIN_DEFAULT_VALUES,
  TERRAIN_FIELDS,
  buildRunMetadataPatch,
  buildActionJob,
  buildTrainingJob,
  buildTweakDraftFromRun,
  canEditPreset,
  canEditRun,
  canOperate,
  canBuildTweakFromRun,
  escapeHtml,
  friendlyErrorMessage,
  formatRelativeTime,
  hasActiveRemoteWork,
  latestFinishedTweakRun,
  jobQueueLabel,
  machineState,
  normalizePreset,
  normalizeTerrainPreset,
  checkpointIteration,
  checkpointOptionsForRun,
  refreshDelayForSnapshot,
  shouldReplaceVideoPanel,
  slugify,
  statusTone,
  videoArtifactForCheckpoint,
  videoStateForCheckpoint,
  videoStateForRun,
} from "./core.js?v=3.0.0-discord-test-feedback";

const PHONE_MEDIA = window.matchMedia
  ? window.matchMedia("(max-width: 720px)")
  : { matches: false, addEventListener: null, addListener: null };

const TEXT_AUTOSAVE_DELAY_MS = 350;
const THEME_KEY = "redrhex_to_go_theme";
const VIEW_IDS = ["train", "rewards", "terrain", "history", "connection", "dashboard"];
const NOTIFICATION_EVENTS = [
  ["notify_training_converged", "Converged", "Reward improvement has flattened."],
  ["notify_training_completed", "Completed", "Training finished successfully."],
  ["notify_training_failed", "Failed / Interrupted", "Training failed or was stopped before finishing."],
  ["notify_video_ready", "Video Ready", "A requested or automatic result video is available."],
];

function initialView() {
  const stored = localStorage.getItem("redrhex_child_view");
  if (stored === "dashboard") return "train";
  return VIEW_IDS.includes(stored) ? stored : "train";
}

function preferredTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

const state = {
  user: null,
  profile: null,
  view: initialView(),
  machineId: localStorage.getItem("redrhex_machine_id") || DEFAULT_MACHINE_ID,
  snapshot: {
    machines: [],
    machine: null,
    jobs: [],
    runs: [],
    artifacts: [],
    presets: [],
    terrainPresets: [],
    notificationSettings: null,
    schema: { artifacts: true, rewardPresets: true, terrainPresets: true, warnings: [] },
  },
  selectedPresetId: localStorage.getItem("redrhex_child_preset") || "baseline",
  draftPreset: null,
  tweakDraftPreset: null,
  selectedTerrainPresetId: localStorage.getItem("redrhex_child_terrain_preset") || "baseline",
  draftTerrainPreset: null,
  tweakTerrainPresetId: "",
  tweakTerrainOverrides: null,
  trainForm: {
    task: "Template-Redrhex-Direct-v0",
    num_envs: 4,
    max_iterations: 8,
    device: "cuda:0",
    seed: "",
  },
  selectedRunId: "",
  runSearch: "",
  folderFilter: "all",
  signedVideos: {},
  videoCheckpointByRun: {},
  runDrafts: {},
  runMetadataAutosaveTimer: null,
  runMetadataSaveInFlight: false,
  runMetadataQueuedRunId: "",
  runMetadataSaveStatus: "saved",
  rewardPresetMetadataAutosaveTimer: null,
  rewardPresetMetadataSaveInFlight: false,
  rewardPresetMetadataQueued: false,
  terrainPresetMetadataAutosaveTimer: null,
  terrainPresetMetadataSaveInFlight: false,
  terrainPresetMetadataQueued: false,
  notificationSettings: null,
  notificationSaveStatus: "saved",
  notificationTestResult: null,
  message: "",
  loading: false,
  loadError: "",
  lastUpdated: "",
  refreshTimer: null,
  refreshing: false,
  isPhone: PHONE_MEDIA.matches,
  theme: preferredTheme(),
};

const app = document.querySelector("#app");
document.documentElement.dataset.theme = state.theme;

function role() {
  return state.profile?.role || "viewer";
}

function defaultNotificationSettings() {
  return {
    user_id: state.user?.id || null,
    machine_id: state.machineId,
    discord_enabled: false,
    discord_webhook_url: "",
    notify_training_converged: true,
    notify_training_completed: true,
    notify_training_failed: true,
    notify_video_ready: true,
  };
}

function normalizeNotificationSettings(raw = null) {
  return {
    ...defaultNotificationSettings(),
    ...(raw || {}),
    user_id: raw?.user_id || state.user?.id || null,
    machine_id: raw?.machine_id || state.machineId,
    discord_enabled: Boolean(raw?.discord_enabled),
    discord_webhook_url: String(raw?.discord_webhook_url || ""),
    notify_training_converged: raw?.notify_training_converged !== false,
    notify_training_completed: raw?.notify_training_completed !== false,
    notify_training_failed: raw?.notify_training_failed !== false,
    notify_video_ready: raw?.notify_video_ready !== false,
  };
}

function selectedPreset() {
  const presets = rewardPresetsForView();
  return presets.find((preset) => preset.id === state.selectedPresetId) || presets[0] || normalizePreset({ id: "baseline", name: "Baseline" });
}

function rewardPresetsForView() {
  const presets = state.snapshot.presets.map(normalizePreset);
  if (!state.tweakDraftPreset) return presets;
  return [state.tweakDraftPreset, ...presets.filter((preset) => preset.id !== state.tweakDraftPreset.id)];
}

function selectedTerrainPreset() {
  const presets = state.snapshot.terrainPresets.map(normalizeTerrainPreset);
  return presets.find((preset) => preset.id === state.selectedTerrainPresetId) || presets[0] || normalizeTerrainPreset({ id: "baseline", name: "Baseline" });
}

function selectedTerrainPresetForTraining() {
  const preset = selectedTerrainPreset();
  if (
    state.tweakDraftPreset
    && state.selectedPresetId === state.tweakDraftPreset.id
    && state.tweakTerrainOverrides
    && state.selectedTerrainPresetId === state.tweakTerrainPresetId
  ) {
    return { ...preset, values: state.tweakTerrainOverrides };
  }
  return preset;
}

function selectedRun({ fallback = true } = {}) {
  const selected = state.snapshot.runs.find((run) => run.id === state.selectedRunId);
  if (selected) return selected;
  return fallback ? state.snapshot.runs[0] || null : null;
}

function runById(runId) {
  return state.snapshot.runs.find((run) => run.id === runId) || null;
}

function selectedHistoryRun() {
  if (state.folderFilter === "all") return null;
  return selectedRun({ fallback: !state.isPhone });
}

function setMessage(message, options = {}) {
  state.message = message;
  if (options.forceRender || !app.querySelector("#message-notice")) {
    render();
  } else {
    patchShellStatus();
  }
}

function setTheme(theme) {
  state.theme = theme === "dark" ? "dark" : "light";
  localStorage.setItem(THEME_KEY, state.theme);
  document.documentElement.dataset.theme = state.theme;
  patchShellStatus();
}

function toggleTheme() {
  setTheme(state.theme === "dark" ? "light" : "dark");
}

function setView(view) {
  if (view === "history") {
    state.folderFilter = "all";
    state.selectedRunId = "";
    state.runSearch = "";
  }
  state.view = view;
  localStorage.setItem("redrhex_child_view", view);
  render();
}

function scheduleRefresh() {
  if (state.refreshTimer) clearTimeout(state.refreshTimer);
  if (!state.user || document.hidden) return;
  const delay = refreshDelayForSnapshot(state.snapshot);
  state.refreshTimer = setTimeout(() => {
    refresh({ silent: true }).catch((error) => {
      state.loadError = friendlyErrorMessage(error);
      patchCurrentView();
      scheduleRefresh();
    });
  }, delay);
}

function currentRunDraft(run) {
  if (!run) return {};
  return state.runDrafts[run.id] || {};
}

function normalizeRunMetadata(values = {}) {
  return {
    display_name: String(values.display_name ?? values.displayName ?? "").trim(),
    folder: String(values.folder ?? "").trim(),
    notes: String(values.notes ?? ""),
  };
}

function runMetadataValuesMatch(a = {}, b = {}) {
  const left = normalizeRunMetadata(a);
  const right = normalizeRunMetadata(b);
  return left.display_name === right.display_name
    && left.folder === right.folder
    && left.notes === right.notes;
}

function currentRunMetadata(run) {
  return normalizeRunMetadata({
    display_name: run?.display_name || "",
    folder: run?.folder || "",
    notes: run?.notes || "",
  });
}

function isRunMetadataElement(element) {
  return ["run-name", "run-folder", "run-notes"].includes(element?.id);
}

function runMetadataStatusText(status) {
  if (status === "saving") return "Saving";
  if (status === "dirty") return "Editing";
  if (status === "error") return "Save failed";
  return "Saved";
}

function setRunMetadataSaveStatus(status, runId = "") {
  const marker = document.querySelector("#run-autosave-status");
  const visibleRunId = document.querySelector("#run-name")?.dataset.runId || state.selectedRunId;
  if (runId && visibleRunId && visibleRunId !== runId) return;
  state.runMetadataSaveStatus = status;
  if (!marker) return;
  marker.dataset.state = status;
  marker.textContent = runMetadataStatusText(status);
}

function signedVideoEntry(storagePath) {
  const entry = state.signedVideos[storagePath];
  if (!entry) return null;
  if (typeof entry === "string") return { url: entry, expiresAt: 0 };
  return entry;
}

function selectedVideoCheckpoint(run) {
  if (!run) return "";
  const options = checkpointOptionsForRun(run, state.snapshot.artifacts);
  const saved = state.videoCheckpointByRun[run.id];
  if (saved && options.some((option) => option.path === saved)) return saved;
  return run.latest_checkpoint || options[0]?.path || "";
}

async function ensureSelectedVideoSigned() {
  const run = selectedRun();
  if (!run) return;
  const video = videoStateForCheckpoint(run, state.snapshot.artifacts, selectedVideoCheckpoint(run));
  const storagePath = video.artifact?.storage_path;
  if (!storagePath) return;
  const existing = signedVideoEntry(storagePath);
  if (existing?.url && existing.expiresAt && existing.expiresAt - Date.now() > 5 * 60_000) return;
  try {
    state.signedVideos[storagePath] = {
      url: await createSignedVideoUrl(storagePath),
      expiresAt: Date.now() + 55 * 60_000,
    };
  } catch (error) {
    state.message = friendlyErrorMessage(error);
  }
}

async function refresh(options = {}) {
  if (!state.user) return;
  if (state.refreshing) return;
  state.refreshing = true;
  const silent = Boolean(options.silent);
  if (!silent) state.loading = true;
  state.loadError = "";
  if (!silent) render();
  try {
    state.snapshot = await loadRemoteSnapshot(state.machineId, state.user?.id || "");
    state.snapshot.presets = state.snapshot.presets.map(normalizePreset);
    state.snapshot.terrainPresets = (state.snapshot.terrainPresets || []).map(normalizeTerrainPreset);
    state.notificationSettings = normalizeNotificationSettings(state.snapshot.notificationSettings);
    if (!state.selectedRunId && state.snapshot.runs[0] && !(state.view === "history" && state.isPhone)) {
      state.selectedRunId = state.snapshot.runs[0].id;
    }
    if (!rewardPresetsForView().find((preset) => preset.id === state.selectedPresetId) && state.snapshot.presets[0]) {
      state.selectedPresetId = state.snapshot.presets[0].id;
    }
    if (!state.snapshot.terrainPresets.find((preset) => preset.id === state.selectedTerrainPresetId) && state.snapshot.terrainPresets[0]) {
      state.selectedTerrainPresetId = state.snapshot.terrainPresets[0].id;
    }
    state.lastUpdated = new Date().toISOString();
    await ensureSelectedVideoSigned();
  } catch (error) {
    state.loadError = friendlyErrorMessage(error);
  } finally {
    state.loading = false;
    state.refreshing = false;
    if (silent && app.querySelector(".nav-tabs")) {
      patchCurrentView();
    } else {
      render();
    }
    scheduleRefresh();
  }
}

async function loadProfile() {
  if (!state.user) return;
  const rows = await select("profiles", `id=eq.${encodeURIComponent(state.user.id)}&select=*`);
  state.profile = rows[0] || { id: state.user.id, email: state.user.email, role: "viewer" };
}

async function boot() {
  state.user = await currentUser();
  if (state.user) {
    await loadProfile();
    await refresh();
  }
  render();
}

function healthChecks() {
  const machine = state.snapshot.targetMachine || state.snapshot.machine;
  const machineStatus = machineState(machine);
  return [
    ["auth", "Signed in", Boolean(state.user), state.user?.email || "No active session"],
    ["role", "Profile role", Boolean(state.profile?.role), role()],
    ["db", "Supabase database", !state.loadError, state.loadError || "Queries responded"],
    ["machine", "Machine heartbeat", machineStatus !== "missing" && machineStatus !== "offline", machine?.heartbeat_at ? `${machine.machine_id} - ${formatRelativeTime(machine.heartbeat_at)}` : "No heartbeat"],
    ["accept", "Worker accepting jobs", Boolean(machine?.accept_jobs), machine?.accept_jobs ? "Ready to queue" : "Paused by mother panel"],
    ["gpu", "GPU lock", !machine?.gpu_locked, machine?.gpu_locked ? "Busy" : "Free"],
    ["rewards", "Reward preset schema", Boolean(state.snapshot.schema?.rewardPresets), state.snapshot.schema?.rewardPresets ? "Shared presets ready" : "Apply schema.sql in Supabase"],
    ["terrain", "Terrain preset schema", Boolean(state.snapshot.schema?.terrainPresets), state.snapshot.schema?.terrainPresets ? "Shared terrain presets ready" : "Apply schema.sql in Supabase"],
    ["video", "Video storage", Boolean(state.snapshot.schema?.artifacts), state.snapshot.schema?.artifacts ? "Private signed playback ready" : "Apply schema.sql in Supabase"],
  ];
}

function shell() {
  const views = [
    ["train", "Train"],
    ["rewards", "Rewards"],
    ["terrain", "Terrain"],
    ["history", "History"],
    ["connection", "Connection"],
    ["dashboard", "Dashboard"],
  ];
  const machine = state.snapshot.targetMachine || state.snapshot.machine;
  const tone = statusTone(machineState(machine));
  return `
    <header class="topbar">
      <div>
        <p class="eyebrow">BioRoLa ABAD RHex Team</p>
        <h1>RedRHex To Go</h1>
        <p class="subcopy">Team training, reward tuning, history, and shared results from anywhere.</p>
      </div>
      <div class="top-status">
        <span id="machine-state-badge" class="badge ${tone}">${escapeHtml(machineState(machine))}</span>
        <span id="role-badge" class="badge">${escapeHtml(role())}</span>
        <span id="last-updated-badge" class="badge">${state.lastUpdated ? `Updated ${escapeHtml(formatRelativeTime(state.lastUpdated))}` : "Not updated yet"}</span>
        <span id="refresh-mode-badge" class="badge ${hasActiveRemoteWork(state.snapshot) ? "info" : ""}">${hasActiveRemoteWork(state.snapshot) ? "Auto-refresh 3s" : "Auto-refresh 15s"}</span>
        <button class="theme-toggle" data-action="toggle-theme">${state.theme === "dark" ? "Light Mode" : "Dark Mode"}</button>
      </div>
    </header>
    <nav class="nav-tabs">
      ${views.map(([id, label]) => `<button class="${state.view === id ? "active" : ""}" data-action="view" data-view="${id}">${label}</button>`).join("")}
    </nav>
    <div id="message-notice" class="notice" ${state.message ? "" : "hidden"}>${escapeHtml(state.message)}</div>
    <div id="schema-warnings">${(state.snapshot.schema?.warnings || []).map((warning) => `<div class="notice warning">${escapeHtml(warning)}</div>`).join("")}</div>
    <div id="load-error-notice" class="notice danger" ${state.loadError ? "" : "hidden"}>${escapeHtml(state.loadError)}</div>
    ${state.user ? welcomeBanner() : ""}
    ${state.user ? page() : loginPage()}
  `;
}

const READY_MESSAGES = [
  "The whegs are warmed up. Time to teach the robot a new trick.",
  "Mission board is clear. RedRHex is waiting for its next adventure.",
  "Mother says all systems are go. Let the little rover cook.",
  "Training lane is open. Tiny legs, serious science.",
  "The robot is caffeinated in spirit and ready in silicon.",
  "Green lights across the board. Pick a reward and send it.",
  "RedRHex is stretching dramatically. Queue the next run.",
  "Fresh run energy detected. The lab is ready.",
];

function readyMessage(seed) {
  let hash = 0;
  for (const char of seed) hash = ((hash << 5) - hash + char.charCodeAt(0)) >>> 0;
  return READY_MESSAGES[hash % READY_MESSAGES.length];
}

function welcomeBanner() {
  const machine = state.snapshot.targetMachine || state.snapshot.machine;
  const displayName = state.profile?.display_name || state.user?.email?.split("@")[0] || "teammate";
  const machineStatus = machineState(machine);
  const isWarning = ["missing", "offline", "paused"].includes(machineStatus);
  const bannerTone = isWarning ? "warning" : machineStatus === "busy" ? "busy" : "ready";
  const readyText = machineStatus === "ready"
    ? readyMessage(`${displayName}-${new Date().toDateString()}`)
    : machineStatus === "busy"
      ? "Mother is busy right now. You can prepare the next job, but launch may wait for the GPU."
      : machineStatus === "paused"
        ? "WARNING: Remote launch is paused in mother. Jobs will not start until mother accepts them."
        : machineStatus === "offline"
          ? "WARNING: Mother looks offline. Check the worker before queuing training."
          : "WARNING: No training machine is connected. Open Connection and verify the worker.";
  return `
    <section class="welcome-banner ${bannerTone}">
      <div>
        <p class="eyebrow">Welcome</p>
        <h2>Hi ${escapeHtml(displayName)}, where should RedRHex go next?</h2>
        <p class="subcopy">${escapeHtml(readyText)}</p>
      </div>
      <div class="welcome-actions">
        <button class="primary" data-action="view" data-view="train">Start Training</button>
        <button data-action="view" data-view="history">View History</button>
      </div>
    </section>
  `;
}

function loginPage() {
  return `
    <section class="login-grid">
      <article class="panel intro-panel">
        <h2>Team Sign In</h2>
        <p>This child panel connects to the RedRHex Supabase control plane. It uses the public project URL and publishable key; the private machine token stays only on the training PC.</p>
        <div class="health-row">
          <span>Project</span>
          <strong>${escapeHtml(new URL(SUPABASE_URL).host)}</strong>
        </div>
      </article>
      <article class="panel">
        <h2>Login</h2>
        <label>Email <input id="login-email" type="email" autocomplete="email"></label>
        <label>Password <input id="login-password" type="password" autocomplete="current-password"></label>
        <button class="primary wide" data-action="login">Sign In</button>
      </article>
    </section>
  `;
}

function page() {
  if (state.view === "train") return trainView();
  if (state.view === "rewards") return rewardsView();
  if (state.view === "terrain") return terrainView();
  if (state.view === "history") return historyView();
  if (state.view === "connection") return connectionView();
  return dashboardView();
}

function dashboardView() {
  return `
    <section class="dashboard-stack">
      <div id="dashboard-status-cards" class="dashboard-status-grid">
        ${dashboardStatusCards()}
      </div>
      <section class="dashboard-grid">
        <article class="panel dashboard-actions-panel">
          <div class="section-head">
            <div>
              <h2>Go</h2>
              <p class="muted">The three places your phone usually needs.</p>
            </div>
            <button data-action="refresh">${state.loading ? "Refreshing" : "Refresh"}</button>
          </div>
          <div class="quick-actions">
            <button class="primary" data-action="view" data-view="train">Train</button>
            <button data-action="view" data-view="terrain">Terrain</button>
            <button data-action="view" data-view="history">History</button>
            <button data-action="view" data-view="connection">Connection</button>
          </div>
        </article>
        <article class="panel">
          <div class="section-head compact">
            <h2>Queue</h2>
            <span class="badge">${escapeHtml(state.lastUpdated ? `Updated ${formatRelativeTime(state.lastUpdated)}` : "Not updated")}</span>
          </div>
          <div id="dashboard-queue-summary">${dashboardQueueSummary()}</div>
        </article>
        <article class="panel span-2">
          <div class="section-head compact">
            <h2>Latest Runs</h2>
            <button data-action="view" data-view="history">View All</button>
          </div>
          <div id="dashboard-latest-runs" class="run-strip">${dashboardLatestRuns()}</div>
        </article>
      </section>
    </section>
  `;
}

function dashboardStatusCards() {
  const machine = state.snapshot.targetMachine || state.snapshot.machine;
  const machineStatus = machineState(machine);
  const jobs = state.snapshot.jobs || [];
  const queued = jobs.filter((job) => String(job.status || "").toLowerCase() === "queued").length;
  const running = jobs.filter((job) => ["claimed", "running"].includes(String(job.status || "").toLowerCase())).length;
  const failed = jobs.filter((job) => String(job.status || "").toLowerCase() === "failed").slice(0, 5).length;
  const latestRun = state.snapshot.runs[0];
  const stateCopy = {
    ready: ["Ready", "Mother is online and accepting jobs."],
    busy: ["Busy", "An Isaac/GPU action is running."],
    paused: ["Paused", "Remote launch is paused in mother."],
    offline: ["Offline", "Mother heartbeat is stale."],
    missing: ["No Machine", "Start the worker from mother Control Center."],
  }[machineStatus] || [machineStatus, "Machine state is unknown."];
  return `
    <article class="panel dashboard-status-card ${statusTone(machineStatus)}">
      <span>Machine</span>
      <strong>${escapeHtml(stateCopy[0])}</strong>
      <small>${escapeHtml(stateCopy[1])}</small>
    </article>
    <article class="panel dashboard-status-card ${machine?.accept_jobs ? "good" : "muted"}">
      <span>Remote Launch</span>
      <strong>${machine?.accept_jobs ? "Accepting" : "Paused"}</strong>
      <small>${machine ? escapeHtml(machine.machine_id) : "No machine selected"}</small>
    </article>
    <article class="panel dashboard-status-card ${running || queued ? "info" : failed ? "bad" : "good"}">
      <span>Queue</span>
      <strong>${running} running · ${queued} queued</strong>
      <small>${failed ? `${failed} recent failed` : "No recent failures"}</small>
    </article>
    <article class="panel dashboard-status-card ${statusTone(latestRun?.status)}">
      <span>Latest Run</span>
      <strong>${escapeHtml(latestRun?.display_name || latestRun?.id || "None yet")}</strong>
      <small>${latestRun ? `${escapeHtml(latestRun.status || "unknown")} · ${escapeHtml(formatRelativeTime(latestRun.created_at))}` : "Queue a training job to begin."}</small>
    </article>
  `;
}

function dashboardQueueSummary() {
  const machine = state.snapshot.targetMachine || state.snapshot.machine;
  const jobs = state.snapshot.jobs || [];
  const active = jobs.filter((job) => ["queued", "claimed", "running"].includes(String(job.status || "").toLowerCase()));
  const visibleJobs = (active.length ? active : jobs).slice(0, 5);
  if (!visibleJobs.length) return empty("No remote jobs yet.");
  return `<div class="mini-list">${visibleJobs.map((job) => `
    <div>
      <strong>${escapeHtml(job.type)}</strong>
      <span class="badge ${statusTone(job.status)}">${escapeHtml(job.status)}</span>
      <small>${escapeHtml(jobQueueLabel(job, machine))}</small>
      <small>${escapeHtml(formatRelativeTime(job.created_at))}</small>
    </div>`).join("")}</div>`;
}

function dashboardLatestRuns() {
  const latestRuns = state.snapshot.runs.slice(0, state.isPhone ? 3 : 5);
  return latestRuns.map(runCard).join("") || empty("No runs synced yet.");
}

function machineCard(machine) {
  if (!machine) return empty("No machine heartbeat yet. Start the worker from mother Control Center.");
  return `
    <div class="machine-card">
      <strong>${escapeHtml(machine.machine_id)}</strong>
      <span class="badge ${statusTone(machineState(machine))}">${escapeHtml(machineState(machine))}</span>
      <small>Heartbeat ${escapeHtml(formatRelativeTime(machine.heartbeat_at))}</small>
      <small>Version ${escapeHtml(machine.panel_version || "unknown")}</small>
      <small>${machine.accept_jobs ? "Accepting remote jobs" : "Remote launch paused"}</small>
    </div>
  `;
}

function jobSummary(jobs) {
  if (!jobs.length) return empty("No recent jobs.");
  const machine = state.snapshot.targetMachine || state.snapshot.machine;
  return `<div class="mini-list">${jobs.map((job) => `
    <div>
      <strong>${escapeHtml(job.type)}</strong>
      <span class="badge ${statusTone(job.status)}">${escapeHtml(job.status)}</span>
      <small>${escapeHtml(jobQueueLabel(job, machine))}</small>
      <small>${escapeHtml(formatRelativeTime(job.created_at))}</small>
    </div>`).join("")}</div>`;
}

function trainView() {
  const preset = selectedPreset();
  const terrainPreset = selectedTerrainPresetForTraining();
  const disabled = !canOperate(role());
  const form = state.trainForm;
  return `
    <section class="split-grid">
      <article class="panel">
        <h2>Launch Training</h2>
        <p class="muted">Queues a job for mother. The worker will run one Isaac/GPU action at a time.</p>
        <label>Machine ID <input id="machine-id" value="${escapeHtml(state.machineId)}"></label>
        <label>Task <input id="task" value="${escapeHtml(form.task)}"></label>
        <div class="input-row">
          <label>Envs <input id="num-envs" type="number" min="1" max="8192" value="${escapeHtml(form.num_envs)}"></label>
          <label>Iterations <input id="max-iterations" type="number" min="1" max="100000" value="${escapeHtml(form.max_iterations)}"></label>
        </div>
        <label>Device <input id="device" value="${escapeHtml(form.device)}"></label>
        <label>Seed <input id="seed" type="number" placeholder="optional" value="${escapeHtml(form.seed ?? "")}"></label>
        <label>Reward Preset
          <select id="train-preset">
            ${rewardPresetsForView().map((item) => `<option value="${escapeHtml(item.id)}" ${item.id === preset.id ? "selected" : ""}>${escapeHtml(item.name)}${item.draft ? " (draft)" : ""}</option>`).join("")}
          </select>
        </label>
        <label>Terrain Preset
          <select id="train-terrain-preset">
            ${state.snapshot.terrainPresets.map((item) => `<option value="${escapeHtml(item.id)}" ${item.id === terrainPreset.id ? "selected" : ""}>${escapeHtml(item.name)}</option>`).join("")}
          </select>
        </label>
        <div class="button-row wrap">
          <button class="primary" data-action="queue-training" ${disabled ? "disabled" : ""}>Queue Training</button>
          <button data-action="tweak-last-run" ${disabled ? "disabled" : ""}>Tweak From Last Run</button>
        </div>
        ${disabled ? `<p class="muted">Viewer accounts can inspect but cannot launch training.</p>` : ""}
      </article>
      <article class="panel">
        <div id="train-preset-snapshot">${trainPresetSnapshot(preset, terrainPreset)}</div>
      </article>
    </section>
  `;
}

function trainPresetSnapshot(preset, terrainPreset = selectedTerrainPreset()) {
  return `
    <h2>Preset Snapshot</h2>
    <h3>${escapeHtml(preset.name)}</h3>
    <p class="muted">${escapeHtml(preset.description || "No description.")}</p>
    ${rewardSnapshot(preset.values)}
    <h3>${escapeHtml(terrainPreset.name)}</h3>
    <p class="muted">${escapeHtml(terrainPreset.description || "No description.")}</p>
    ${terrainSnapshot(terrainPreset.values)}
  `;
}

function rewardSnapshot(values) {
  const entries = Object.entries(values || {});
  if (!entries.length) return empty("Baseline uses the current local defaults from mother.");
  return `<div class="reward-snapshot">${entries.map(([key, value]) => `
    <div><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}</div>`;
}

function terrainSnapshot(values) {
  const entries = Object.entries(values || {});
  if (!entries.length) return empty("Baseline uses the current local terrain defaults from mother.");
  return `<div class="reward-snapshot">${entries.slice(0, 12).map(([key, value]) => `
    <div><span>${escapeHtml(key)}</span><strong>${escapeHtml(formatTerrainValue(value))}</strong></div>`).join("")}${entries.length > 12 ? `<div><span>More</span><strong>${entries.length - 12} override${entries.length - 12 === 1 ? "" : "s"}</strong></div>` : ""}</div>`;
}

function rewardsView() {
  const preset = state.draftPreset || selectedPreset();
  const rewardSchemaReady = Boolean(state.snapshot.schema?.rewardPresets);
  const editable = (rewardSchemaReady || preset.draft) && canEditPreset(role()) && !preset.built_in;
  return `
    <section class="rewards-page">
      ${presetRail("reward", preset, rewardSchemaReady)}
      <article class="panel reward-workspace">
        ${rewardHeader(preset, rewardSchemaReady, editable)}
        <div class="preset-meta-grid">
          <label>Name <input id="preset-name" value="${escapeHtml(preset.name)}" ${editable ? "" : "disabled"}></label>
          <label>Description <textarea id="preset-description" ${editable ? "" : "disabled"}>${escapeHtml(preset.description)}</textarea></label>
        </div>
        <div class="reward-editor">
          ${REWARD_FIELDS.map((group) => rewardGroup(group, preset, editable, "reward")).join("")}
        </div>
      </article>
    </section>
  `;
}

function terrainView() {
  const preset = state.draftTerrainPreset || selectedTerrainPreset();
  const terrainSchemaReady = Boolean(state.snapshot.schema?.terrainPresets);
  const editable = terrainSchemaReady && canEditPreset(role()) && !preset.built_in;
  return `
    <section class="rewards-page terrain-page">
      ${presetRail("terrain", preset, terrainSchemaReady)}
      <article class="panel reward-workspace">
        ${terrainHeader(preset, terrainSchemaReady, editable)}
        <div class="preset-meta-grid">
          <label>Name <input id="terrain-preset-name" value="${escapeHtml(preset.name)}" ${editable ? "" : "disabled"}></label>
          <label>Description <textarea id="terrain-preset-description" ${editable ? "" : "disabled"}>${escapeHtml(preset.description)}</textarea></label>
        </div>
        <div class="reward-editor terrain-editor">
          ${TERRAIN_FIELDS.map((group) => terrainGroup(group, preset, editable)).join("")}
        </div>
      </article>
    </section>
  `;
}

function presetRail(kind, preset, schemaReady) {
  const isTerrain = kind === "terrain";
  const presets = isTerrain ? state.snapshot.terrainPresets : rewardPresetsForView();
  const title = isTerrain ? "Terrain" : "Presets";
  const subtitle = isTerrain ? "Shared terrain recipes" : "Shared reward recipes";
  const selectAction = isTerrain ? "select-terrain-preset" : "select-preset";
  const newAction = isTerrain ? "new-terrain-preset" : "new-preset";
  return `
    <aside class="panel preset-list rewards-rail">
      <div class="section-head compact reward-rail-head">
        <div>
          <h2>${escapeHtml(title)}</h2>
          <p class="muted">${escapeHtml(subtitle)}</p>
        </div>
        <button class="icon-action" title="New preset" data-action="${newAction}" ${!schemaReady || !canEditPreset(role()) ? "disabled" : ""}>+</button>
      </div>
      ${schemaReady ? "" : `<p class="muted">Using built-in fallback presets until Supabase schema is updated.</p>`}
      <div class="preset-scroll">
        ${presets.map((item) => `
        <button class="preset-button ${item.id === preset.id ? "active" : ""} ${item.draft ? "draft-preset" : ""}" data-action="${selectAction}" data-id="${escapeHtml(item.id)}">
          <strong>${escapeHtml(item.name)}</strong>
          <small>${item.draft ? "Unsaved draft" : item.built_in ? "Built-in" : "Team preset"} · ${escapeHtml(formatRelativeTime(item.updated_at))}</small>
        </button>`).join("") || empty("Apply the latest schema to create presets.")}
      </div>
    </aside>
  `;
}

function rewardHeader(preset, rewardSchemaReady, editable) {
  const summary = preset.draft ? "Unsaved tweak draft. Edit rewards, then queue training or save it as a preset." : preset.built_in ? "Built-in preset. Duplicate it before editing." : editable ? "Editable team preset." : "Read-only preset.";
  return `
    <div class="section-head reward-head">
      <div>
        <h2>Reward Tuning</h2>
        <p class="muted">${escapeHtml(summary)}</p>
      </div>
      <div class="button-row reward-actions">
        <button data-action="toggle-all-groups" data-kind="reward">Collapse All</button>
        <button data-action="duplicate-preset" ${!rewardSchemaReady || !canEditPreset(role()) ? "disabled" : ""}>Duplicate</button>
        <button class="danger" data-action="delete-preset" ${!editable ? "disabled" : ""}>${preset.draft ? "Discard Draft" : "Delete"}</button>
        <button class="primary" data-action="save-preset" ${!editable || !rewardSchemaReady ? "disabled" : ""}>${preset.draft ? "Save as Preset" : "Save Preset"}</button>
      </div>
    </div>
  `;
}

function terrainHeader(preset, terrainSchemaReady, editable) {
  return `
    <div class="section-head reward-head">
      <div>
        <h2>Terrain Tuning</h2>
        <p class="muted">${escapeHtml(preset.built_in ? "Built-in preset. Duplicate it before editing." : editable ? "Editable team preset." : "Read-only preset.")}</p>
      </div>
      <div class="button-row reward-actions">
        <button data-action="toggle-all-groups" data-kind="terrain">Collapse All</button>
        <button data-action="duplicate-terrain-preset" ${!terrainSchemaReady || !canEditPreset(role()) ? "disabled" : ""}>Duplicate</button>
        <button class="danger" data-action="delete-terrain-preset" ${!editable ? "disabled" : ""}>Delete</button>
        <button class="primary" data-action="save-terrain-preset" ${!editable ? "disabled" : ""}>Save Preset</button>
      </div>
    </div>
  `;
}

function rewardGroup(group, preset, editable, kind = "reward") {
  return `
    <section class="reward-group editor-group" data-kind="${escapeHtml(kind)}">
      <button class="group-toggle" data-action="toggle-editor-group" type="button">
        <span>${escapeHtml(group.name)}</span>
        <span class="chevron">▾</span>
      </button>
      <div class="group-body">
        ${group.fields.map(([key, label, help]) => {
        const value = Number(preset.values?.[key] ?? 0);
        return rewardInputRow(key, label, help, value, editable);
      }).join("")}
      </div>
    </section>`;
}

function rewardInputRow(key, label, help, value, editable) {
  return `
    <label class="reward-row">
      <span><strong>${escapeHtml(label)}</strong><small>${escapeHtml(help)}</small><code>${escapeHtml(key)}</code></span>
      <input class="reward-input" data-key="${escapeHtml(key)}" type="number" step="0.01" value="${escapeHtml(value)}" ${editable ? "" : "disabled"}>
    </label>`;
}

function terrainGroup(group, preset, editable) {
  return `
    <section class="reward-group editor-group" data-kind="terrain">
      <button class="group-toggle" data-action="toggle-editor-group" type="button">
        <span>${escapeHtml(group.name)}</span>
        <span class="chevron">▾</span>
      </button>
      <div class="group-body">
        ${group.fields.map((field) => terrainInputRow(field, preset, editable)).join("")}
      </div>
    </section>`;
}

function terrainEffectiveValue(preset, key) {
  if (Object.prototype.hasOwnProperty.call(preset.values || {}, key)) return preset.values[key];
  return TERRAIN_DEFAULT_VALUES[key];
}

function formatTerrainValue(value) {
  if (Array.isArray(value)) return `[${value.join(", ")}]`;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "";
  return String(value);
}

function terrainInputRow(field, preset, editable) {
  const value = terrainEffectiveValue(preset, field.key);
  const defaultValue = TERRAIN_DEFAULT_VALUES[field.key];
  return `
    <label class="reward-row terrain-row">
      <span>
        <strong>${escapeHtml(field.label)}</strong>
        <small>${escapeHtml(field.help || "")}</small>
        <code>${escapeHtml(field.key)}</code>
      </span>
      <span class="terrain-control">
        ${terrainInput(field, value, editable)}
        <small>default ${escapeHtml(formatTerrainValue(defaultValue))}</small>
      </span>
    </label>`;
}

function terrainInput(field, value, editable) {
  const disabled = editable ? "" : "disabled";
  const safeValue = escapeHtml(formatTerrainValue(value));
  if (field.type === "bool") {
    return `<input class="terrain-input" data-key="${escapeHtml(field.key)}" data-type="bool" type="checkbox" ${value ? "checked" : ""} ${disabled}>`;
  }
  if (field.type === "choice") {
    return `<select class="terrain-input" data-key="${escapeHtml(field.key)}" data-type="choice" ${disabled}>
      ${(field.choices || []).map((choice) => `<option value="${escapeHtml(choice)}" ${String(choice) === String(value) ? "selected" : ""}>${escapeHtml(choice)}</option>`).join("")}
    </select>`;
  }
  if (field.type === "int" || field.type === "float") {
    return `<input class="terrain-input" data-key="${escapeHtml(field.key)}" data-type="${escapeHtml(field.type)}" type="number" step="${escapeHtml(String(field.step || (field.type === "int" ? 1 : 0.01)))}" value="${safeValue}" ${disabled}>`;
  }
  return `<input class="terrain-input terrain-wide-input" data-key="${escapeHtml(field.key)}" data-type="${escapeHtml(field.type)}" value="${safeValue}" ${disabled}>`;
}

function filteredRuns() {
  const q = state.runSearch.trim().toLowerCase();
  return state.snapshot.runs.filter((run) => {
    const folder = run.folder || "";
    if (state.folderFilter === "uncategorized" && folder) return false;
    if (state.folderFilter !== "all" && state.folderFilter !== "uncategorized" && folder !== state.folderFilter) return false;
    if (!q) return true;
    return `${run.id} ${run.display_name || ""} ${run.status || ""} ${folder}`.toLowerCase().includes(q);
  });
}

function runMatchesSearch(run) {
  const q = state.runSearch.trim().toLowerCase();
  if (!q) return true;
  return `${run.id} ${run.display_name || ""} ${run.status || ""} ${run.folder || ""}`.toLowerCase().includes(q);
}

function folders() {
  const set = new Set(state.snapshot.runs.map((run) => run.folder).filter(Boolean));
  return [...set].sort((a, b) => a.localeCompare(b));
}

function folderLabel(folderKey) {
  return folderKey === "uncategorized" ? "Uncategorized" : folderKey;
}

function folderRuns(folderKey) {
  return state.snapshot.runs.filter((run) => {
    if (folderKey === "uncategorized") return !run.folder;
    return run.folder === folderKey;
  });
}

function folderSummaries() {
  const summaries = folders().map((folder) => ({
    key: folder,
    label: folder,
    runs: folderRuns(folder),
  }));
  const uncategorized = folderRuns("uncategorized");
  if (uncategorized.length) {
    summaries.push({ key: "uncategorized", label: "Uncategorized", runs: uncategorized });
  }
  const q = state.runSearch.trim().toLowerCase();
  return summaries
    .map((folder) => ({
      ...folder,
      latest: folder.runs.slice().sort((a, b) => String(b.updated_at || b.created_at || "").localeCompare(String(a.updated_at || a.created_at || "")))[0],
      matches: !q || folder.label.toLowerCase().includes(q) || folder.runs.some(runMatchesSearch),
    }))
    .filter((folder) => folder.matches)
    .sort((a, b) => a.label.localeCompare(b.label));
}

function folderCard(folder) {
  const completed = folder.runs.filter((run) => run.status === "completed").length;
  const running = folder.runs.filter((run) => ["running", "stopping"].includes(String(run.status || "").toLowerCase())).length;
  return `
    <button class="folder-card" data-action="open-folder" data-folder="${escapeHtml(folder.key)}">
      <span class="folder-card-top">
        <strong>${escapeHtml(folder.label)}</strong>
        <span class="badge">${folder.runs.length} run${folder.runs.length === 1 ? "" : "s"}</span>
      </span>
      <small>${running ? `${running} running - ` : ""}${completed} completed</small>
      <small>Latest ${escapeHtml(formatRelativeTime(folder.latest?.updated_at || folder.latest?.created_at))}</small>
    </button>
  `;
}

function historyBrowserContent() {
  if (state.folderFilter === "all") {
    const summaries = folderSummaries();
    return `<div class="folder-list">${summaries.map(folderCard).join("") || empty("No folders yet. Add a folder name to a run from its details.")}</div>`;
  }
  const runs = filteredRuns();
  return `
    ${folderReturnCard()}
    <div class="run-list">${runs.map(runCardWithOptionalInlineDetails).join("") || empty("No matching runs in this folder.")}</div>
  `;
}

function historyView() {
  return historyLayout();
}

function historyLayout() {
  const selected = selectedHistoryRun();
  const phoneClass = state.isPhone ? "inline-history" : "";
  const root = state.folderFilter === "all";
  const currentLabel = root ? "Folders" : folderLabel(state.folderFilter);
  const folderCount = folderSummaries().length;
  return `
    <section class="history-layout ${phoneClass}">
      ${root ? "" : historyDesktopBackBar(currentLabel)}
      <aside class="panel run-list-panel">
        <div class="section-head compact">
          <div>
            <h2>History</h2>
            <p class="muted">${escapeHtml(currentLabel)}</p>
          </div>
          <button data-action="refresh">Refresh</button>
        </div>
        ${historyFolderNav({ root, currentLabel, folderCount })}
        <input id="run-search" placeholder="${root ? "Search folders and runs" : "Search this folder"}" value="${escapeHtml(state.runSearch)}">
        <div id="history-browser">${historyBrowserContent()}</div>
      </aside>
      ${state.isPhone ? "" : `<article class="panel run-details">${selected ? runDetails(selected, { context: "desktop" }) : empty(root ? "Open a folder to see runs." : "Select a run.")}</article>`}
    </section>
  `;
}

function historyDesktopBackBar(currentLabel) {
  const runCount = filteredRuns().length;
  return `
    <div class="desktop-folder-backbar">
      <button class="desktop-folder-back" data-action="open-folder-root">
        <span aria-hidden="true">←</span>
        Folder Library
      </button>
      <div>
        <small>Viewing folder</small>
        <strong>${escapeHtml(currentLabel)}</strong>
        <span>${runCount} run${runCount === 1 ? "" : "s"}</span>
      </div>
    </div>
  `;
}

function historyFolderNav({ root, currentLabel, folderCount }) {
  if (root) {
    const runCount = state.snapshot.runs.length;
    return `
      <div class="history-folder-nav root">
        <div>
          <strong>Folder Library</strong>
          <small>${folderCount} folder${folderCount === 1 ? "" : "s"} · ${runCount} run${runCount === 1 ? "" : "s"}</small>
        </div>
      </div>
    `;
  }
  const runCount = filteredRuns().length;
  return `
    <div class="history-folder-nav inside">
      <button class="folder-back" data-action="open-folder-root" aria-label="Back to folders">
        <span class="folder-back-icon" aria-hidden="true">←</span>
        <span><strong>Folders</strong><small>Back to library</small></span>
      </button>
      <div class="folder-current">
        <small>Inside</small>
        <strong>${escapeHtml(currentLabel)}</strong>
        <span>${runCount} run${runCount === 1 ? "" : "s"}</span>
      </div>
    </div>
  `;
}

function folderReturnCard() {
  return `
    <button class="folder-return-card" data-action="open-folder-root">
      <span class="folder-return-icon" aria-hidden="true">←</span>
      <span>
        <strong>Back to Folder Library</strong>
        <small>Choose another folder</small>
      </span>
    </button>
  `;
}

function renderRunListOnly() {
  patchRunList();
}

function runCard(run) {
  const video = videoStateForRun(run, state.snapshot.artifacts);
  const active = run.id === state.selectedRunId;
  const params = run.params || {};
  const terrainPreset = run.terrain_preset_id || params.terrain_preset_id || "baseline";
  return `
    <button class="run-card ${active ? "active" : ""}" data-action="select-run" data-id="${escapeHtml(run.id)}">
      <span class="run-card-top">
        <strong>${escapeHtml(run.display_name || run.id)}</strong>
        <span class="badge ${statusTone(run.status)}">${escapeHtml(run.status || "unknown")}</span>
      </span>
      <small>${escapeHtml(run.folder || "Uncategorized")} - ${escapeHtml(formatRelativeTime(run.created_at))}</small>
      <small>${run.latest_checkpoint ? "checkpoint ready" : "no checkpoint"} - video ${escapeHtml(video.state)} - terrain ${escapeHtml(terrainPreset)}</small>
    </button>
  `;
}

function runCardWithOptionalInlineDetails(run) {
  const active = run.id === state.selectedRunId;
  return `
    <div class="run-card-wrap ${active ? "active" : ""}">
      ${runCard(run)}
      ${state.isPhone && active ? `<article class="inline-run-details">${runDetails(run, { context: "inline" })}</article>` : ""}
    </div>
  `;
}

function runDetailsGrid(run) {
  const video = videoStateForRun(run, state.snapshot.artifacts);
  const params = run.params || {};
  const rewardOverrides = Object.keys(params.reward_overrides || run.reward_overrides || {}).length;
  const terrainOverrides = Object.keys(params.terrain_overrides || run.terrain_overrides || {}).length;
  const checkpointIter = checkpointIteration(run.latest_checkpoint);
  const checkpointLabel = run.latest_checkpoint
    ? Number.isFinite(checkpointIter) ? `Iteration ${checkpointIter}` : run.latest_checkpoint.split("/").pop()
    : "Missing";
  const videoLabel = video.state === "ready"
    ? "Team video ready"
    : video.state === "uploading"
      ? "Uploading to team storage"
      : video.state === "recordable"
        ? "Ready to record"
        : "No checkpoint yet";
  const rewardPreset = run.reward_preset_id || params.reward_preset_id || "baseline";
  const terrainPreset = run.terrain_preset_id || params.terrain_preset_id || "baseline";
  return `
    <article class="run-info-card">
      <span>Run State</span>
      <strong class="${statusTone(run.status)}">${escapeHtml(run.status || "unknown")}</strong>
      <small>Updated ${escapeHtml(formatRelativeTime(run.updated_at || run.created_at))}</small>
    </article>
    <article class="run-info-card">
      <span>Artifacts</span>
      <div class="run-info-list">
        <small>Checkpoint <strong>${escapeHtml(checkpointLabel)}</strong></small>
        <small>Video <strong>${escapeHtml(videoLabel)}</strong></small>
      </div>
    </article>
    <article class="run-info-card">
      <span>Training Recipe</span>
      <div class="run-info-list">
        <small>Reward <strong>${escapeHtml(rewardPreset)}</strong> <em>${rewardOverrides} override${rewardOverrides === 1 ? "" : "s"}</em></small>
        <small>Terrain <strong>${escapeHtml(terrainPreset)}</strong> <em>${terrainOverrides} override${terrainOverrides === 1 ? "" : "s"}</em></small>
      </div>
    </article>
  `;
}

function plainObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function rewardFieldMeta() {
  const meta = new Map();
  for (const group of REWARD_FIELDS) {
    for (const [key, label, help] of group.fields) {
      meta.set(key, { key, label, help, group: group.name });
    }
  }
  return meta;
}

function rewardPresetById(presetId) {
  const id = String(presetId || "baseline");
  return state.snapshot.presets.map(normalizePreset).find((preset) => String(preset.id) === id) || null;
}

function runRewardPresetId(run) {
  const params = plainObject(run.params);
  return String(params.reward_preset_id || run.reward_preset_id || "baseline");
}

function rewardSnapshotForRun(run) {
  const params = plainObject(run.params);
  const paramOverrides = plainObject(params.reward_overrides);
  if (Object.keys(paramOverrides).length) return paramOverrides;
  const runOverrides = plainObject(run.reward_overrides);
  if (Object.keys(runOverrides).length) return runOverrides;
  return plainObject(rewardPresetById(runRewardPresetId(run))?.values);
}

function rewardBaselineValues() {
  return plainObject(rewardPresetById("baseline")?.values);
}

function rewardNumber(value) {
  const num = Number(value ?? 0);
  return Number.isFinite(num) ? num : value;
}

function rewardValuesEqual(left, right) {
  const leftNum = Number(left ?? 0);
  const rightNum = Number(right ?? 0);
  if (Number.isFinite(leftNum) && Number.isFinite(rightNum)) {
    return Math.abs(leftNum - rightNum) < 1e-9;
  }
  return String(left ?? "") === String(right ?? "");
}

function formatRewardNumber(value) {
  const num = Number(value ?? 0);
  if (!Number.isFinite(num)) return String(value ?? "");
  if (Number.isInteger(num)) return String(num);
  return num.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function rewardDeltaLabel(value, baseline) {
  const valueNum = Number(value ?? 0);
  const baselineNum = Number(baseline ?? 0);
  if (!Number.isFinite(valueNum) || !Number.isFinite(baselineNum)) return "";
  const delta = valueNum - baselineNum;
  if (Math.abs(delta) < 1e-9) return "same";
  const pct = Math.abs(baselineNum) > 1e-9 ? ` · ${Math.round((delta / Math.abs(baselineNum)) * 100)}%` : "";
  return `${delta > 0 ? "+" : ""}${formatRewardNumber(delta)}${pct}`;
}

function rewardComparisonRows(run) {
  const meta = rewardFieldMeta();
  const snapshot = rewardSnapshotForRun(run);
  const baseline = rewardBaselineValues();
  const knownKeys = REWARD_FIELDS.flatMap((group) => group.fields.map(([key]) => key));
  const keys = [...new Set([...knownKeys, ...Object.keys(snapshot), ...Object.keys(baseline)])];
  return keys
    .map((key) => {
      const field = meta.get(key) || { key, label: key, help: "", group: "Other" };
      const value = rewardNumber(snapshot[key] ?? 0);
      const base = rewardNumber(baseline[key] ?? 0);
      const valueNum = Number(value ?? 0);
      const baseNum = Number(base ?? 0);
      const delta = Number.isFinite(valueNum) && Number.isFinite(baseNum) ? valueNum - baseNum : null;
      return { ...field, value, baseline: base, delta };
    })
    .filter((row) => !rewardValuesEqual(row.value, row.baseline));
}

function rewardComparisonSection(run) {
  const rows = rewardComparisonRows(run);
  const presetId = runRewardPresetId(run);
  const preset = rewardPresetById(presetId);
  const baseline = rewardPresetById("baseline");
  const groups = REWARD_FIELDS
    .map((group) => ({ name: group.name, rows: rows.filter((row) => row.group === group.name) }))
    .filter((group) => group.rows.length);
  const otherRows = rows.filter((row) => !REWARD_FIELDS.some((group) => group.name === row.group));
  if (otherRows.length) groups.push({ name: "Other", rows: otherRows });
  return `
    <section id="reward-comparison-panel" class="subpanel reward-comparison-panel">
      <div class="section-head compact">
        <div>
          <h3>Reward Comparison</h3>
          <p class="muted">Run snapshot compared with ${escapeHtml(baseline?.name || "Baseline")}.</p>
        </div>
        <span class="badge">${rows.length} changed</span>
      </div>
      <div class="reward-compare-summary">
        <span><strong>Run preset</strong>${escapeHtml(preset?.name || presetId)}</span>
        <span><strong>Baseline</strong>${escapeHtml(baseline?.name || "Baseline")}</span>
      </div>
      ${rows.length ? groups.map((group, index) => `
        <details class="reward-compare-group" data-group="${escapeHtml(group.name)}" ${index === 0 ? "open" : ""}>
          <summary><span>${escapeHtml(group.name)}</span><em>${group.rows.length}</em></summary>
          <div class="reward-compare-list">
            ${group.rows.map(rewardComparisonRow).join("")}
          </div>
        </details>
      `).join("") : empty("No reward changes are recorded for this run.")}
    </section>
  `;
}

function rewardComparisonRow(row) {
  const dir = row.delta === null ? "" : row.delta > 0 ? "up" : "down";
  return `
    <div class="reward-compare-row">
      <span class="reward-compare-name">
        <strong>${escapeHtml(row.label)}</strong>
        <small>${escapeHtml(row.key)}</small>
      </span>
      <span><small>Baseline</small><strong>${escapeHtml(formatRewardNumber(row.baseline))}</strong></span>
      <span><small>This run</small><strong>${escapeHtml(formatRewardNumber(row.value))}</strong></span>
      <span class="reward-compare-delta ${dir}">${escapeHtml(rewardDeltaLabel(row.value, row.baseline))}</span>
    </div>
  `;
}

function patchRewardComparison(run) {
  const panel = document.querySelector("#reward-comparison-panel");
  if (!panel) return;
  const openGroups = [...panel.querySelectorAll(".reward-compare-group[open]")]
    .map((group) => group.dataset.group || "");
  const hadGroups = Boolean(panel.querySelector(".reward-compare-group"));
  panel.outerHTML = rewardComparisonSection(run);
  if (!hadGroups) return;
  document.querySelectorAll("#reward-comparison-panel .reward-compare-group").forEach((group) => {
    group.open = openGroups.includes(group.dataset.group || "");
  });
}

function relatedJobsForRun(run) {
  return state.snapshot.jobs.filter((job) => {
    const payload = job.payload || {};
    const result = job.result || {};
    return payload.run_id === run.id || result.local_run_id === run.id || result.process_id === run.id;
  }).slice(0, 8);
}

function relatedJobsSection(run) {
  const relatedJobs = relatedJobsForRun(run);
  return `
    <section id="related-jobs-panel" class="subpanel">
      <h3>Related Jobs</h3>
      ${relatedJobs.length ? `<div class="mini-list">${relatedJobs.map((job) => `
        <div><strong>${escapeHtml(job.type)}</strong><span class="badge ${statusTone(job.status)}">${escapeHtml(job.status)}</span><small>${escapeHtml(jobQueueLabel(job, state.snapshot.targetMachine || state.snapshot.machine))}</small>${jobExtraLine(job)}<small>${escapeHtml(formatRelativeTime(job.created_at))}</small></div>
      `).join("")}</div>` : empty("No remote jobs linked to this run yet.")}
    </section>
  `;
}

function jobExtraLine(job) {
  const payload = job.result?.payload || {};
  if (job.type === "tensorboard" && payload.url) {
    return `<small><a href="${escapeHtml(payload.url)}" target="_blank" rel="noreferrer">TensorBoard ${escapeHtml(payload.url)}</a></small>`;
  }
  if (job.type === "record_video" && job.payload?.checkpoint_iteration) {
    return `<small>checkpoint iteration ${escapeHtml(job.payload.checkpoint_iteration)}</small>`;
  }
  return "";
}

function teamVideoSection(run) {
  const options = checkpointOptionsForRun(run, state.snapshot.artifacts);
  const checkpoint = selectedVideoCheckpoint(run);
  const selectedOption = options.find((option) => option.path === checkpoint) || options[0] || null;
  const video = videoStateForCheckpoint(run, state.snapshot.artifacts, checkpoint);
  const videoArtifact = video.artifact;
  const signed = videoArtifact ? signedVideoEntry(videoArtifact.storage_path)?.url || "" : "";
  const runnable = canOperate(role()) && Boolean(checkpoint);
  const storagePath = videoArtifact?.storage_path || "";
  const selectedIteration = Number.isFinite(selectedOption?.iteration) ? selectedOption.iteration : checkpointIteration(checkpoint);
  const selectedLabel = Number.isFinite(selectedIteration)
    ? `Iteration ${selectedIteration}`
    : selectedOption?.label || "Selected checkpoint";
  const checkpointName = checkpoint ? checkpoint.split("/").pop() : "";
  return `
    <section id="team-video-panel" class="subpanel" data-video-state="${escapeHtml(video.state)}" data-storage-path="${escapeHtml(storagePath)}">
      <div class="section-head compact">
        <h3>Team Video</h3>
        <span class="badge ${statusTone(video.state)}">${escapeHtml(video.state)}</span>
      </div>
      <div class="video-context">
        <span>Video target</span>
        <strong>${escapeHtml(selectedLabel)}</strong>
        ${checkpointName ? `<small>${escapeHtml(checkpointName)}</small>` : ""}
      </div>
      <label>Choose checkpoint video
        <select id="video-checkpoint-select" ${options.length ? "" : "disabled"}>
          ${options.map((option) => `<option value="${escapeHtml(option.path)}" ${option.path === checkpoint ? "selected" : ""}>${escapeHtml(option.label)}</option>`).join("")}
        </select>
      </label>
      ${video.state === "ready" ? `
        <p class="video-now">Viewing team video for <strong>${escapeHtml(selectedLabel)}</strong>.</p>
        ${signed ? `<video controls src="${escapeHtml(signed)}"></video>` : `<p class="muted">Preparing a signed team-only video link...</p>`}
        <div class="button-row">
          <button data-action="load-video" data-path="${escapeHtml(videoArtifact.storage_path)}">${signed ? "Refresh Secure Link" : "Load Team Video"}</button>
          <button data-action="copy-video-path" data-path="${escapeHtml(videoArtifact.storage_path)}">Copy Storage Path</button>
        </div>` : video.state === "uploading" ? `
        <p class="muted">Video for ${escapeHtml(selectedLabel)} exists locally and is uploading to team storage.</p>
        <button data-action="refresh">Refresh Status</button>
      ` : video.state === "recordable" ? `
        <p class="muted">No team video has been recorded for ${escapeHtml(selectedLabel)} yet.</p>
        <button class="primary" data-action="check-video" ${runnable ? "" : "disabled"}>Record Video for ${escapeHtml(selectedLabel)}</button>
      ` : `<p class="muted">No checkpoint yet, so video recording is not available.</p>`}
    </section>
  `;
}

function runDetails(run, { context = "desktop" } = {}) {
  const draft = currentRunDraft(run);
  const editable = canEditRun(role());
  const tweakable = canOperate(role()) && canBuildTweakFromRun(run, state.snapshot.presets);
  return `
    <div class="section-head run-detail-head ${context === "inline" ? "inline" : ""}">
      <div>
        <h2 id="selected-run-title">${escapeHtml(run.display_name || run.id)}</h2>
        <p id="selected-run-id" class="muted">${escapeHtml(run.id)}</p>
      </div>
      <span id="selected-run-status" class="badge ${statusTone(run.status)}">${escapeHtml(run.status || "unknown")}</span>
    </div>
    <div id="run-details-grid" class="details-grid">
      ${runDetailsGrid(run)}
    </div>
    ${rewardComparisonSection(run)}
    <section class="subpanel">
      <h3>Run Metadata</h3>
      <label>Name <input id="run-name" data-run-id="${escapeHtml(run.id)}" value="${escapeHtml(draft.display_name ?? run.display_name ?? "")}" ${editable ? "" : "disabled"}></label>
      <label>Folder <input id="run-folder" data-run-id="${escapeHtml(run.id)}" value="${escapeHtml(draft.folder ?? run.folder ?? "")}" placeholder="e.g. gait tests" ${editable ? "" : "disabled"}></label>
      <label>Notes <textarea id="run-notes" data-run-id="${escapeHtml(run.id)}" ${editable ? "" : "disabled"}>${escapeHtml(draft.notes ?? run.notes ?? "")}</textarea></label>
      ${editable ? `<div class="autosave-row"><span id="run-autosave-status" class="autosave-status" data-state="${escapeHtml(state.runMetadataSaveStatus)}">${escapeHtml(runMetadataStatusText(state.runMetadataSaveStatus))}</span></div>` : ""}
    </section>
    ${teamVideoSection(run)}
    <section class="subpanel">
      <h3>Safe Remote Actions</h3>
      <div class="button-row wrap">
        <button data-action="tweak-run" ${tweakable ? "" : "disabled"}>Tweak From This Run</button>
        <button data-action="job-tensorboard" ${canOperate(role()) ? "" : "disabled"}>TensorBoard</button>
        <button data-action="job-compact-run" ${canOperate(role()) ? "" : "disabled"}>Compact Run</button>
      </div>
    </section>
    ${relatedJobsSection(run)}
  `;
}

function notificationSettingsForView() {
  return normalizeNotificationSettings(state.notificationSettings);
}

function notificationChannelResult(result) {
  if (!result) return "";
  const parts = Object.entries(result).map(([channel, value]) => {
    const item = value && typeof value === "object" ? value : {};
    if (item.pending) return `${channel}: sending`;
    const detail = item.error || item.reason || item.status || "";
    if (item.skipped) return `${channel}: skipped${detail ? ` (${detail})` : ""}`;
    if (item.ok) return `${channel}: sent`;
    return `${channel}: failed${detail ? ` (${detail})` : ""}`;
  });
  return parts.join(" · ");
}

function notificationSettingsCard() {
  const settings = notificationSettingsForView();
  const testText = state.notificationTestResult ? notificationChannelResult(state.notificationTestResult.results || {}) : "";
  return `
    <article class="panel span-2 notification-card">
      <div class="section-head">
        <div>
          <h2>Notifications</h2>
          <p class="muted">Discord run alerts go only to the person who queued the run.</p>
        </div>
        <span class="badge ${state.notificationSaveStatus === "error" ? "bad" : state.notificationSaveStatus === "saving" ? "info" : "good"}">${escapeHtml(state.notificationSaveStatus)}</span>
      </div>
      <div class="notification-grid">
        <section>
          <label class="switch-row">
            <input id="notify-discord-enabled" type="checkbox" ${settings.discord_enabled ? "checked" : ""}>
            <span>Discord</span>
          </label>
          <label>Discord Webhook
            <input id="notify-discord-webhook" type="url" placeholder="https://discord.com/api/webhooks/..." value="${escapeHtml(settings.discord_webhook_url)}">
          </label>
        </section>
        <section>
          <h3>Events</h3>
          <div class="notification-events">
            ${NOTIFICATION_EVENTS.map(([key, label, detail]) => `
              <label class="switch-row">
                <input class="notify-event-toggle" data-key="${escapeHtml(key)}" type="checkbox" ${settings[key] ? "checked" : ""}>
                <span><strong>${escapeHtml(label)}</strong><small>${escapeHtml(detail)}</small></span>
              </label>`).join("")}
          </div>
        </section>
      </div>
      <div class="button-row wrap">
        <button class="primary" data-action="save-notification-settings">Save Notifications</button>
        <button data-action="send-test-notification">Send Test</button>
      </div>
      ${testText ? `<p class="muted">${escapeHtml(testText)}</p>` : ""}
    </article>
  `;
}

function connectionView() {
  const checks = healthChecks();
  return `
    <section class="connection-grid">
      <article class="panel span-2">
        <div class="section-head">
          <div>
            <h2>Connection</h2>
            <p class="muted">Use this page first when the phone UI feels disconnected.</p>
          </div>
          <div class="button-row">
            <button data-action="refresh">Run Checks</button>
            <button data-action="sign-out">Sign Out</button>
          </div>
        </div>
        <div class="check-list">
          ${checks.map(([, label, ok, detail]) => `
            <div class="${ok ? "ok" : "warn"}">
              <strong>${escapeHtml(label)}</strong>
              <span>${ok ? "pass" : "check"}</span>
              <small>${escapeHtml(detail)}</small>
            </div>`).join("")}
        </div>
      </article>
      <article class="panel">
        <h2>Target Machine</h2>
        <label>Machine ID <input id="connection-machine-id" value="${escapeHtml(state.machineId)}"></label>
        <button data-action="save-machine">Save Target</button>
      </article>
      <article class="panel">
        <h2>Account</h2>
        <p><strong>${escapeHtml(state.user?.email || "")}</strong></p>
        <p class="muted">Role: ${escapeHtml(role())}</p>
        <p class="muted">Project: ${escapeHtml(new URL(SUPABASE_URL).host)}</p>
      </article>
      ${notificationSettingsCard()}
    </section>
  `;
}

function empty(text) {
  return `<p class="muted empty">${escapeHtml(text)}</p>`;
}

function setTextAndClass(selector, text, className) {
  const element = document.querySelector(selector);
  if (!element) return;
  element.textContent = text;
  if (className) element.className = className;
}

function patchShellStatus() {
  const machine = state.snapshot.targetMachine || state.snapshot.machine;
  const machineStatus = machineState(machine);
  setTextAndClass("#machine-state-badge", machineStatus, `badge ${statusTone(machineStatus)}`);
  setTextAndClass("#role-badge", role(), "badge");
  setTextAndClass(
    "#last-updated-badge",
    state.lastUpdated ? `Updated ${formatRelativeTime(state.lastUpdated)}` : "Not updated yet",
    "badge",
  );
  setTextAndClass(
    "#refresh-mode-badge",
    hasActiveRemoteWork(state.snapshot) ? "Auto-refresh 3s" : "Auto-refresh 15s",
    `badge ${hasActiveRemoteWork(state.snapshot) ? "info" : ""}`.trim(),
  );
  const themeToggle = document.querySelector(".theme-toggle");
  if (themeToggle) themeToggle.textContent = state.theme === "dark" ? "Light Mode" : "Dark Mode";

  const message = document.querySelector("#message-notice");
  if (message) {
    message.textContent = state.message;
    message.hidden = !state.message;
  }
  const warningSlot = document.querySelector("#schema-warnings");
  if (warningSlot) {
    warningSlot.innerHTML = (state.snapshot.schema?.warnings || [])
      .map((warning) => `<div class="notice warning">${escapeHtml(warning)}</div>`)
      .join("");
  }
  const error = document.querySelector("#load-error-notice");
  if (error) {
    error.textContent = state.loadError;
    error.hidden = !state.loadError;
  }
}

function patchDashboard() {
  const statusCards = document.querySelector("#dashboard-status-cards");
  if (statusCards) statusCards.innerHTML = dashboardStatusCards();
  const queueSlot = document.querySelector("#dashboard-queue-summary");
  if (queueSlot) queueSlot.innerHTML = dashboardQueueSummary();
  const latestRuns = document.querySelector("#dashboard-latest-runs");
  if (latestRuns) latestRuns.innerHTML = dashboardLatestRuns();
}

function patchRunList() {
  const browser = document.querySelector("#history-browser");
  if (!browser) return;
  const panel = document.querySelector(".run-list-panel");
  const panelScrollTop = panel?.scrollTop || 0;
  const scrollTop = browser.scrollTop;
  const pageScrollTop = window.scrollY;
  browser.innerHTML = historyBrowserContent();
  browser.scrollTop = scrollTop;
  if (panel) panel.scrollTop = panelScrollTop;
  window.scrollTo({ top: pageScrollTop });
}

function patchRunCardsInPlace() {
  document.querySelectorAll(".run-card").forEach((card) => {
    const run = state.snapshot.runs.find((item) => item.id === card.dataset.id);
    if (!run) return;
    const video = videoStateForRun(run, state.snapshot.artifacts);
    const active = run.id === state.selectedRunId;
    card.classList.toggle("active", active);
    card.closest(".run-card-wrap")?.classList.toggle("active", active);
    const title = card.querySelector(".run-card-top strong");
    if (title) title.textContent = run.display_name || run.id;
    const badge = card.querySelector(".run-card-top .badge");
    if (badge) {
      badge.textContent = run.status || "unknown";
      badge.className = `badge ${statusTone(run.status)}`;
    }
    const lines = card.querySelectorAll("small");
    if (lines[0]) lines[0].textContent = `${run.folder || "Uncategorized"} - ${formatRelativeTime(run.created_at)}`;
    const params = run.params || {};
    if (lines[1]) lines[1].textContent = `${run.latest_checkpoint ? "checkpoint ready" : "no checkpoint"} - video ${video.state} - terrain ${run.terrain_preset_id || params.terrain_preset_id || "baseline"}`;
  });
}

function removeInlineDetails(details) {
  if (!details) return;
  details.classList.add("collapsing");
  details.addEventListener("animationend", () => details.remove(), { once: true });
}

function syncHistoryAccordion(card) {
  if (!state.isPhone || state.view !== "history") return false;
  const wrapper = card?.closest(".run-card-wrap");
  const runId = card?.dataset.id || "";
  if (!wrapper || !runId) return false;
  const wasOpen = state.selectedRunId === runId && Boolean(wrapper.querySelector(".inline-run-details"));
  const existingDetails = [...document.querySelectorAll(".inline-run-details")];

  if (wasOpen) {
    existingDetails.forEach(removeInlineDetails);
    state.selectedRunId = "";
    patchRunCardsInPlace();
    return true;
  }

  existingDetails.forEach((details) => details.remove());
  state.selectedRunId = runId;
  patchRunCardsInPlace();
  const run = state.snapshot.runs.find((item) => item.id === runId);
  if (run) {
    wrapper.insertAdjacentHTML("beforeend", `<article class="inline-run-details">${runDetails(run, { context: "inline" })}</article>`);
  }
  return true;
}

function isRunMetadataFocused() {
  return isRunMetadataElement(document.activeElement);
}

function setRunMetadataInputs(run) {
  if (!run) return;
  const draft = currentRunDraft(run);
  const values = {
    display_name: draft.display_name ?? run.display_name ?? "",
    folder: draft.folder ?? run.folder ?? "",
    notes: draft.notes ?? run.notes ?? "",
  };
  const name = document.querySelector("#run-name");
  const folder = document.querySelector("#run-folder");
  const notes = document.querySelector("#run-notes");
  if (name) {
    name.dataset.runId = run.id;
    name.value = values.display_name;
  }
  if (folder) {
    folder.dataset.runId = run.id;
    folder.value = values.folder;
  }
  if (notes) {
    notes.dataset.runId = run.id;
    notes.value = values.notes;
  }
}

function persistActiveRunMetadataDraft({ flush = false } = {}) {
  const active = document.activeElement;
  if (!isRunMetadataElement(active)) return "";
  const runId = active.dataset.runId || state.selectedRunId;
  if (!runId) return "";
  updateRunMetadataDraft(runId);
  if (flush) {
    flushRunMetadataAutosave(runId).catch((error) => {
      setRunMetadataSaveStatus("error", runId);
      setMessage(friendlyErrorMessage(error));
    });
  }
  return runId;
}

function patchTeamVideo(run) {
  const panel = document.querySelector("#team-video-panel");
  if (!panel || !run) return;
  const video = videoStateForCheckpoint(run, state.snapshot.artifacts, selectedVideoCheckpoint(run));
  const nextStorage = video.artifact?.storage_path || "";
  const videoElement = panel.querySelector("video");
  if (videoElement && panel.dataset.storagePath && !nextStorage) return;
  const signedReady = Boolean(nextStorage && signedVideoEntry(nextStorage)?.url);
  const shouldReplace = (video.state === "ready" && signedReady && !videoElement) || shouldReplaceVideoPanel({
    currentState: panel.dataset.videoState || "",
    currentStorage: panel.dataset.storagePath || "",
    nextState: video.state,
    nextStorage,
    isPlaying: Boolean(videoElement && !videoElement.paused),
  });
  if (!shouldReplace) return;
  panel.outerHTML = teamVideoSection(run);
}

function patchSelectedRunDetails() {
  const details = document.querySelector(".run-details, .inline-run-details");
  const run = selectedHistoryRun();
  if (!details) return;
  if (!run) {
    details.innerHTML = empty(state.folderFilter === "all" ? "Open a folder to see runs." : "Select a run.");
    return;
  }

  if (!details.querySelector("#selected-run-title")) {
    details.innerHTML = runDetails(run);
    return;
  }

  const title = document.querySelector("#selected-run-title");
  if (title) title.textContent = run.display_name || run.id;
  const id = document.querySelector("#selected-run-id");
  if (id) id.textContent = run.id;
  setTextAndClass("#selected-run-status", run.status || "unknown", `badge ${statusTone(run.status)}`);

  const grid = document.querySelector("#run-details-grid");
  if (grid) grid.innerHTML = runDetailsGrid(run);

  patchRewardComparison(run);

  const metadataRunId = document.querySelector("#run-name")?.dataset.runId || "";
  if (metadataRunId !== run.id || (!isRunMetadataFocused() && !state.runDrafts[run.id])) {
    setRunMetadataInputs(run);
    if (metadataRunId !== run.id && !state.runDrafts[run.id]) {
      setRunMetadataSaveStatus("saved");
    }
  }

  patchTeamVideo(run);
  const related = document.querySelector("#related-jobs-panel");
  if (related) related.outerHTML = relatedJobsSection(run);
}

function patchHistory({ forceList = false } = {}) {
  if (state.folderFilter === "all") {
    patchRunList();
    patchSelectedRunDetails();
    return;
  }
  if (!forceList && state.isPhone && document.querySelector(".inline-run-details")) {
    patchRunCardsInPlace();
    patchSelectedRunDetails();
    return;
  }
  if (!isRunMetadataFocused()) {
    patchRunList();
  }
  patchSelectedRunDetails();
}

function patchConnection() {
  const connectionGrid = document.querySelector(".connection-grid");
  if (connectionGrid && !["connection-machine-id", "notify-discord-webhook"].includes(document.activeElement?.id)) {
    connectionGrid.outerHTML = connectionView();
  }
}

function patchCurrentView() {
  patchShellStatus();
  if (!state.user) return;
  if (state.view === "dashboard") patchDashboard();
  if (state.view === "history") patchHistory();
  if (state.view === "connection") patchConnection();
}

function collectRewardValues() {
  const values = {};
  document.querySelectorAll(".reward-input").forEach((input) => {
    values[input.dataset.key] = Number(input.value || 0);
  });
  return values;
}

function parseTerrainValue(input) {
  const type = input.dataset.type || "string";
  if (type === "bool") return Boolean(input.checked);
  if (type === "int") return Number.parseInt(input.value || "0", 10);
  if (type === "float") return Number.parseFloat(input.value || "0");
  if (type === "range" || type === "list") {
    const text = String(input.value || "").trim();
    if (!text) return [];
    try {
      const parsed = JSON.parse(text);
      return Array.isArray(parsed) ? parsed : [parsed];
    } catch {
      return text
        .replace(/^\[|\]$/g, "")
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => {
          const value = Number(item);
          return Number.isFinite(value) ? value : item;
        });
    }
  }
  return input.value || "";
}

function collectTerrainValues() {
  const values = {};
  document.querySelectorAll(".terrain-input").forEach((input) => {
    values[input.dataset.key] = parseTerrainValue(input);
  });
  return values;
}

function setGroupsCollapsed(kind, collapsed) {
  document.querySelectorAll(`.editor-group[data-kind="${kind}"]`).forEach((group) => {
    group.classList.toggle("collapsed", collapsed);
  });
  updateGroupToggleLabel(kind);
}

function updateGroupToggleLabel(kind) {
  const button = document.querySelector(`[data-action="toggle-all-groups"][data-kind="${kind}"]`);
  if (!button) return;
  const groups = [...document.querySelectorAll(`.editor-group[data-kind="${kind}"]`)];
  const allCollapsed = groups.length > 0 && groups.every((group) => group.classList.contains("collapsed"));
  button.textContent = allCollapsed ? "Expand All" : "Collapse All";
}

function toggleAllGroups(kind) {
  const groups = [...document.querySelectorAll(`.editor-group[data-kind="${kind}"]`)];
  const allCollapsed = groups.length > 0 && groups.every((group) => group.classList.contains("collapsed"));
  setGroupsCollapsed(kind, !allCollapsed);
}

async function handleLogin() {
  const email = document.querySelector("#login-email")?.value || "";
  const password = document.querySelector("#login-password")?.value || "";
  await signIn(email, password);
  state.user = await currentUser();
  await loadProfile();
  await refresh();
  setMessage("Signed in.");
}

async function queueTraining() {
  state.machineId = document.querySelector("#machine-id")?.value || state.machineId;
  localStorage.setItem("redrhex_machine_id", state.machineId);
  state.trainForm = {
    task: document.querySelector("#task")?.value || "Template-Redrhex-Direct-v0",
    num_envs: Number(document.querySelector("#num-envs")?.value || 4),
    max_iterations: Number(document.querySelector("#max-iterations")?.value || 8),
    device: document.querySelector("#device")?.value || "cuda:0",
    seed: document.querySelector("#seed")?.value || "",
  };
  const presetId = document.querySelector("#train-preset")?.value || state.selectedPresetId;
  state.selectedPresetId = presetId;
  localStorage.setItem("redrhex_child_preset", presetId);
  const terrainPresetId = document.querySelector("#train-terrain-preset")?.value || state.selectedTerrainPresetId;
  state.selectedTerrainPresetId = terrainPresetId;
  localStorage.setItem("redrhex_child_terrain_preset", terrainPresetId);
  if (state.tweakDraftPreset && state.selectedPresetId === state.tweakDraftPreset.id) {
    const values = collectRewardValues();
    if (Object.keys(values).length) state.tweakDraftPreset.values = values;
  }
  const preset = selectedPreset();
  const terrainPreset = selectedTerrainPresetForTraining();
  const params = {
    task: state.trainForm.task,
    num_envs: state.trainForm.num_envs,
    max_iterations: state.trainForm.max_iterations,
    device: state.trainForm.device,
  };
  if (state.trainForm.seed !== "") params.seed = Number(state.trainForm.seed);
  if (preset.draft) {
    params.tweak_source_run_id = preset.source_run_id || "";
    params.tweak_source_label = preset.source_label || preset.source_run_id || "";
  }
  const job = buildTrainingJob({ machineId: state.machineId, params, preset, terrainPreset, role: role(), userId: state.user?.id });
  await insert("jobs", job);
  await refresh({ silent: true });
  setMessage(`Queued training with ${preset.name} and ${terrainPreset.name}.`);
}

function applyTweakPayload(payload) {
  const params = payload.training_params || {};
  state.trainForm = {
    task: params.task || "Template-Redrhex-Direct-v0",
    num_envs: Number(params.num_envs || 4),
    max_iterations: Number(params.max_iterations || 8),
    device: params.device || "cuda:0",
    seed: params.seed ?? "",
  };
  state.tweakDraftPreset = normalizePreset({
    ...payload.reward_preset,
    draft: true,
    source_run_id: payload.source_run?.id || params.tweak_source_run_id || payload.reward_preset?.source_run_id || "",
    source_label: params.tweak_source_label || payload.source_run?.display_name || payload.source_run?.id || "",
  });
  state.tweakDraftPreset.draft = true;
  state.tweakDraftPreset.source_run_id = payload.source_run?.id || params.tweak_source_run_id || "";
  state.tweakDraftPreset.source_label = params.tweak_source_label || payload.source_run?.display_name || payload.source_run?.id || "";
  state.draftPreset = null;
  state.selectedPresetId = state.tweakDraftPreset.id;
  localStorage.setItem("redrhex_child_preset", state.selectedPresetId);
  state.selectedTerrainPresetId = params.terrain_preset_id || payload.terrain_preset_id || "baseline";
  state.tweakTerrainPresetId = state.selectedTerrainPresetId;
  state.tweakTerrainOverrides = params.terrain_overrides || payload.terrain_overrides || {};
  localStorage.setItem("redrhex_child_terrain_preset", state.selectedTerrainPresetId);
  setMessage(payload.message || "Loaded tweak draft.", { forceRender: false });
  setView("rewards");
}

function tweakPayloadFromRun(run) {
  return buildTweakDraftFromRun(run, {
    presets: state.snapshot.presets,
    terrainPresets: state.snapshot.terrainPresets,
  });
}

function tweakFromRun(runId) {
  const run = runById(runId);
  if (!run) return setMessage("Run not found.");
  try {
    applyTweakPayload(tweakPayloadFromRun(run));
  } catch (error) {
    setMessage(friendlyErrorMessage(error));
  }
}

function tweakFromLastRun() {
  const run = latestFinishedTweakRun(state.snapshot.runs, state.snapshot.presets);
  if (!run) return setMessage("No finished run with usable tweak data found.");
  tweakFromRun(run.id);
}

function presetMetadataStateKeys(kind) {
  return kind === "terrain"
    ? {
      timer: "terrainPresetMetadataAutosaveTimer",
      saving: "terrainPresetMetadataSaveInFlight",
      queued: "terrainPresetMetadataQueued",
    }
    : {
      timer: "rewardPresetMetadataAutosaveTimer",
      saving: "rewardPresetMetadataSaveInFlight",
      queued: "rewardPresetMetadataQueued",
    };
}

function editablePresetForKind(kind) {
  const isTerrain = kind === "terrain";
  const preset = isTerrain ? state.draftTerrainPreset || selectedTerrainPreset() : state.draftPreset || selectedPreset();
  const schemaReady = isTerrain ? Boolean(state.snapshot.schema?.terrainPresets) : Boolean(state.snapshot.schema?.rewardPresets);
  if (!isTerrain && preset?.draft && !state.draftPreset) return null;
  if (!preset || preset.built_in || !schemaReady || !canEditPreset(role())) return null;
  return preset;
}

function presetMetadataValues(kind, preset) {
  const nameSelector = kind === "terrain" ? "#terrain-preset-name" : "#preset-name";
  const descriptionSelector = kind === "terrain" ? "#terrain-preset-description" : "#preset-description";
  return {
    name: String(document.querySelector(nameSelector)?.value || preset?.name || "").trim() || preset?.name || "Untitled Preset",
    description: document.querySelector(descriptionSelector)?.value || "",
  };
}

function patchPresetMetadataInSnapshot(kind, presetId, values, updatedAt) {
  const presets = kind === "terrain" ? state.snapshot.terrainPresets : state.snapshot.presets;
  const preset = presets.find((item) => item.id === presetId);
  if (!preset) return;
  preset.name = values.name;
  preset.description = values.description;
  preset.updated_at = updatedAt;
}

function schedulePresetMetadataAutosave(kind) {
  const keys = presetMetadataStateKeys(kind);
  if (state[keys.timer]) clearTimeout(state[keys.timer]);
  state[keys.queued] = true;
  state[keys.timer] = setTimeout(() => {
    flushPresetMetadataAutosave(kind).catch((error) => setMessage(friendlyErrorMessage(error)));
  }, TEXT_AUTOSAVE_DELAY_MS);
}

async function flushPresetMetadataAutosave(kind) {
  const keys = presetMetadataStateKeys(kind);
  if (state[keys.timer]) {
    clearTimeout(state[keys.timer]);
    state[keys.timer] = null;
  }
  state[keys.queued] = false;
  await savePresetMetadata(kind, { autosave: true });
}

async function savePresetMetadata(kind, options = {}) {
  const keys = presetMetadataStateKeys(kind);
  if (state[keys.saving]) {
    state[keys.queued] = true;
    return;
  }
  const preset = editablePresetForKind(kind);
  if (!preset) return;
  const isTerrain = kind === "terrain";
  const isDraft = isTerrain ? Boolean(state.draftTerrainPreset) : Boolean(state.draftPreset);
  const values = presetMetadataValues(kind, preset);
  if (!isDraft && values.name === (preset.name || "") && values.description === (preset.description || "")) {
    return;
  }

  const updatedAt = new Date().toISOString();
  state[keys.saving] = true;
  try {
    if (isDraft) {
      const payload = {
        id: preset.id,
        name: values.name,
        description: values.description,
        values: isTerrain ? collectTerrainValues() : collectRewardValues(),
        built_in: false,
        created_by: preset.created_by || state.user?.id || null,
        updated_by: state.user?.id || null,
        updated_at: updatedAt,
      };
      await upsert(isTerrain ? "terrain_presets" : "reward_presets", payload);
      if (isTerrain) {
        state.draftTerrainPreset = null;
        state.selectedTerrainPresetId = payload.id;
        localStorage.setItem("redrhex_child_terrain_preset", payload.id);
      } else {
        state.draftPreset = null;
        state.selectedPresetId = payload.id;
        localStorage.setItem("redrhex_child_preset", payload.id);
      }
    } else {
      await update(
        isTerrain ? "terrain_presets" : "reward_presets",
        `id=eq.${encodeURIComponent(preset.id)}`,
        {
          name: values.name,
          description: values.description,
          updated_by: state.user?.id || null,
          updated_at: updatedAt,
        },
      );
      patchPresetMetadataInSnapshot(kind, preset.id, values, updatedAt);
    }
    await refresh({ silent: true });
    setMessage(`${isTerrain ? "Terrain" : "Reward"} preset metadata autosaved.`);
  } finally {
    state[keys.saving] = false;
    if (state[keys.queued]) {
      schedulePresetMetadataAutosave(kind);
    }
  }
}

async function savePreset() {
  const preset = state.draftPreset || selectedPreset();
  if (preset.built_in) return;
  const payload = {
    id: preset.id,
    name: document.querySelector("#preset-name")?.value || preset.name,
    description: document.querySelector("#preset-description")?.value || "",
    values: collectRewardValues(),
    built_in: false,
    updated_by: state.user?.id || null,
    updated_at: new Date().toISOString(),
  };
  if (!preset.created_by) payload.created_by = state.user?.id || null;
  await upsert("reward_presets", payload);
  state.draftPreset = null;
  state.tweakDraftPreset = null;
  state.selectedPresetId = payload.id;
  await refresh();
  setMessage(`Saved preset ${payload.name}.`);
}

async function deletePreset() {
  const preset = state.draftPreset || selectedPreset();
  if (!preset || preset.built_in) return;
  if (!window.confirm(`Delete reward preset "${preset.name}"?`)) return;
  if (state.draftPreset || preset.draft) {
    state.draftPreset = null;
    if (preset.draft) state.tweakDraftPreset = null;
    state.selectedPresetId = "baseline";
    localStorage.setItem("redrhex_child_preset", state.selectedPresetId);
    render();
    return;
  }
  await remove("reward_presets", `id=eq.${encodeURIComponent(preset.id)}`);
  state.selectedPresetId = "baseline";
  localStorage.setItem("redrhex_child_preset", state.selectedPresetId);
  await refresh();
  setMessage(`Deleted preset ${preset.name}.`);
}

function duplicatePreset() {
  const source = state.draftPreset || selectedPreset();
  const id = slugify(`${source.id || source.name}-copy`);
  if (source.draft) state.tweakDraftPreset = null;
  state.draftPreset = {
    ...source,
    id,
    name: `${source.name} Copy`,
    built_in: false,
    draft: false,
    values: { ...(source.values || {}) },
    created_by: state.user?.id || null,
  };
  state.selectedPresetId = id;
  render();
}

async function saveTerrainPreset() {
  const preset = state.draftTerrainPreset || selectedTerrainPreset();
  if (preset.built_in) return;
  const payload = {
    id: preset.id,
    name: document.querySelector("#terrain-preset-name")?.value || preset.name,
    description: document.querySelector("#terrain-preset-description")?.value || "",
    values: collectTerrainValues(),
    built_in: false,
    updated_by: state.user?.id || null,
    updated_at: new Date().toISOString(),
  };
  if (!preset.created_by) payload.created_by = state.user?.id || null;
  await upsert("terrain_presets", payload);
  state.draftTerrainPreset = null;
  state.selectedTerrainPresetId = payload.id;
  localStorage.setItem("redrhex_child_terrain_preset", payload.id);
  await refresh();
  setMessage(`Saved terrain preset ${payload.name}.`);
}

async function deleteTerrainPreset() {
  const preset = state.draftTerrainPreset || selectedTerrainPreset();
  if (!preset || preset.built_in) return;
  if (!window.confirm(`Delete terrain preset "${preset.name}"?`)) return;
  if (state.draftTerrainPreset) {
    state.draftTerrainPreset = null;
    state.selectedTerrainPresetId = "baseline";
    render();
    return;
  }
  await remove("terrain_presets", `id=eq.${encodeURIComponent(preset.id)}`);
  state.selectedTerrainPresetId = "baseline";
  localStorage.setItem("redrhex_child_terrain_preset", state.selectedTerrainPresetId);
  await refresh();
  setMessage(`Deleted terrain preset ${preset.name}.`);
}

function duplicateTerrainPreset() {
  const source = state.draftTerrainPreset || selectedTerrainPreset();
  const id = slugify(`${source.id || source.name}-copy`);
  state.draftTerrainPreset = {
    ...source,
    id,
    name: `${source.name} Copy`,
    built_in: false,
    values: { ...(source.values || {}) },
    created_by: state.user?.id || null,
  };
  state.selectedTerrainPresetId = id;
  render();
}

function newTerrainPreset() {
  const id = slugify(`team-terrain-${Date.now()}`);
  state.draftTerrainPreset = {
    id,
    name: "New Terrain Preset",
    description: "",
    values: { ...TERRAIN_DEFAULT_VALUES },
    built_in: false,
    created_by: state.user?.id || null,
  };
  state.selectedTerrainPresetId = id;
  render();
}

function newPreset() {
  const id = slugify(`team-preset-${Date.now()}`);
  state.draftPreset = {
    id,
    name: "New Team Preset",
    description: "",
    values: {},
    built_in: false,
    created_by: state.user?.id || null,
  };
  state.selectedPresetId = id;
  render();
}

function collectRunMetadataDraft(run) {
  if (!run) return null;
  const nameInput = document.querySelector("#run-name");
  if (nameInput?.dataset.runId === run.id) {
    return {
      display_name: nameInput.value ?? run.display_name ?? "",
      folder: document.querySelector("#run-folder")?.value ?? run.folder ?? "",
      notes: document.querySelector("#run-notes")?.value ?? run.notes ?? "",
    };
  }
  return state.runDrafts[run.id] || currentRunMetadata(run);
}

function updateRunMetadataDraft(runId) {
  const run = runById(runId);
  if (!run) return null;
  const draft = collectRunMetadataDraft(run);
  if (!draft) return null;
  state.runDrafts[run.id] = {
    ...state.runDrafts[run.id],
    ...draft,
  };
  return state.runDrafts[run.id];
}

function scheduleRunMetadataAutosave(runId) {
  if (!runId) return;
  if (state.runMetadataAutosaveTimer) clearTimeout(state.runMetadataAutosaveTimer);
  state.runMetadataQueuedRunId = runId;
  setRunMetadataSaveStatus("dirty", runId);
  state.runMetadataAutosaveTimer = setTimeout(() => {
    flushRunMetadataAutosave(runId).catch((error) => {
      setRunMetadataSaveStatus("error", runId);
      setMessage(friendlyErrorMessage(error));
    });
  }, TEXT_AUTOSAVE_DELAY_MS);
}

async function flushRunMetadataAutosave(runId = state.runMetadataQueuedRunId) {
  if (!runId) return;
  if (state.runMetadataAutosaveTimer) {
    clearTimeout(state.runMetadataAutosaveTimer);
    state.runMetadataAutosaveTimer = null;
  }
  state.runMetadataQueuedRunId = "";
  await saveRun(runId, { autosave: true });
}

async function saveRun(runId = state.selectedRunId, options = {}) {
  const targetRunId = runId || state.selectedRunId;
  const run = targetRunId ? runById(targetRunId) : selectedRun({ fallback: false });
  if (!run) return;
  if (state.runMetadataSaveInFlight) {
    state.runMetadataQueuedRunId = run.id;
    return;
  }
  const draft = collectRunMetadataDraft(run);
  const normalized = normalizeRunMetadata(draft);
  if (runMetadataValuesMatch(normalized, currentRunMetadata(run))) {
    delete state.runDrafts[run.id];
    if (options.autosave) setRunMetadataSaveStatus("saved", run.id);
    if (!options.autosave) setMessage("No metadata changes to save.");
    return;
  }
  state.runMetadataSaveInFlight = true;
  if (options.autosave) setRunMetadataSaveStatus("saving", run.id);
  try {
    await update(
      "runs",
      `id=eq.${encodeURIComponent(run.id)}`,
      buildRunMetadataPatch({
        displayName: normalized.display_name,
        folder: normalized.folder,
        notes: normalized.notes,
      }),
    );
    run.display_name = normalized.display_name || null;
    run.folder = normalized.folder || null;
    run.notes = normalized.notes;
    const latestDraft = state.runDrafts[run.id];
    if (!latestDraft || runMetadataValuesMatch(latestDraft, normalized)) {
      delete state.runDrafts[run.id];
    } else {
      state.runMetadataQueuedRunId = run.id;
    }
    await refresh({ silent: true });
    if (options.autosave) {
      setRunMetadataSaveStatus("saved", run.id);
    } else {
      setMessage("Run metadata saved.");
    }
  } catch (error) {
    if (options.autosave) setRunMetadataSaveStatus("error", run.id);
    throw error;
  } finally {
    state.runMetadataSaveInFlight = false;
    if (state.runMetadataQueuedRunId) {
      scheduleRunMetadataAutosave(state.runMetadataQueuedRunId);
    }
  }
}

async function queueRunAction(type, message, payload = {}) {
  const run = selectedRun();
  if (!run) return;
  const job = buildActionJob({ machineId: state.machineId, type, runId: run.id, role: role(), userId: state.user?.id, payload });
  await insert("jobs", job);
  await refresh({ silent: true });
  setMessage(message);
}

async function checkOrCreateVideo() {
  const run = selectedRun();
  if (!run) return;
  const checkpoint = selectedVideoCheckpoint(run);
  const videoArtifact = videoArtifactForCheckpoint(run, state.snapshot.artifacts, checkpoint);
  if (videoArtifact?.storage_path) {
    await loadVideo(videoArtifact.storage_path);
    return setMessage("Loaded the existing video for that checkpoint.");
  }
  const iteration = checkpointIteration(checkpoint);
  await queueRunAction(
    "record_video",
    Number.isFinite(iteration) ? `Queued video for iteration ${iteration}.` : "Queued checkpoint video.",
    {
      checkpoint,
      checkpoint_iteration: Number.isFinite(iteration) ? iteration : null,
    },
  );
}

async function queueTensorBoard() {
  const run = selectedRun();
  if (!run) return;
  await queueRunAction("tensorboard", "Queued TensorBoard startup.", { host: "0.0.0.0" });
}

async function compactSelectedRun() {
  const run = selectedRun();
  if (!run) return;
  if (!window.confirm(`Compact run "${run.display_name || run.id}"? Older checkpoints and bulky cache files may be removed.`)) return;
  await queueRunAction("compact_run", "Queued run compaction.", { confirmation: run.id });
}

function collectNotificationSettings() {
  const settings = notificationSettingsForView();
  const next = {
    ...settings,
    user_id: state.user?.id || settings.user_id,
    machine_id: state.machineId,
    discord_enabled: Boolean(document.querySelector("#notify-discord-enabled")?.checked),
    discord_webhook_url: String(document.querySelector("#notify-discord-webhook")?.value || "").trim(),
    updated_at: new Date().toISOString(),
  };
  document.querySelectorAll(".notify-event-toggle").forEach((input) => {
    next[input.dataset.key] = Boolean(input.checked);
  });
  return next;
}

async function saveNotificationSettings({ silent = false } = {}) {
  const payload = collectNotificationSettings();
  state.notificationSaveStatus = "saving";
  patchConnection();
  try {
    const rows = await upsert("notification_settings", payload, "user_id,machine_id");
    state.notificationSettings = normalizeNotificationSettings(rows?.[0] || payload);
    state.notificationSaveStatus = "saved";
    if (!silent) setMessage("Notification settings saved.");
    await refresh({ silent: true });
  } catch (error) {
    state.notificationSaveStatus = "error";
    throw error;
  } finally {
    patchConnection();
  }
}

async function sendTestNotification() {
  state.notificationTestResult = { results: { discord: { pending: true } } };
  patchConnection();
  try {
    await saveNotificationSettings({ silent: true });
    const result = await invokeFunction("notify", {
      event_type: "test_notification",
      machine_id: state.machineId,
      payload: {
        display_name: "Notification test",
        task: "Connection page",
        remote_url: window.location.href,
      },
    });
    state.notificationTestResult = result;
    patchConnection();
    setMessage(result?.ok ? "Test notification sent." : "Test notification finished with warnings.");
  } catch (error) {
    const message = friendlyErrorMessage(error);
    state.notificationTestResult = { ok: false, results: { discord: { ok: false, error: message } } };
    patchConnection();
    setMessage(message);
  }
}

async function loadVideo(storagePath) {
  state.signedVideos[storagePath] = {
    url: await createSignedVideoUrl(storagePath),
    expiresAt: Date.now() + 55 * 60_000,
  };
  const run = selectedRun();
  if (state.view === "history" && run) {
    const panel = document.querySelector("#team-video-panel");
    if (panel) panel.outerHTML = teamVideoSection(run);
  } else {
    render();
  }
}

function openHistoryFolder(folderKey) {
  persistActiveRunMetadataDraft({ flush: true });
  state.folderFilter = folderKey || "all";
  state.selectedRunId = "";
  if (state.folderFilter !== "all" && !state.isPhone) {
    const firstRun = filteredRuns()[0];
    if (firstRun) state.selectedRunId = firstRun.id;
  }
  patchHistory({ forceList: true });
  patchShellStatus();
}

function render() {
  app.innerHTML = shell();
}

function handlePhoneModeChange(event) {
  const nextIsPhone = Boolean(event.matches);
  if (state.isPhone === nextIsPhone) return;
  state.isPhone = nextIsPhone;
  if (state.user) render();
}

if (PHONE_MEDIA.addEventListener) {
  PHONE_MEDIA.addEventListener("change", handlePhoneModeChange);
} else if (PHONE_MEDIA.addListener) {
  PHONE_MEDIA.addListener(handlePhoneModeChange);
}

document.addEventListener("click", async (event) => {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  event.preventDefault();
  const action = target.dataset.action;
  try {
    if (action === "toggle-theme") return toggleTheme();
    if (action === "view") return setView(target.dataset.view);
    if (action === "login") return await handleLogin();
    if (action === "refresh") return await refresh({ silent: true });
    if (action === "sign-out") {
      await signOut();
      state.user = null;
      state.profile = null;
      if (state.refreshTimer) clearTimeout(state.refreshTimer);
      setMessage("Signed out.", { forceRender: true });
      return;
    }
    if (action === "save-machine") {
      state.machineId = document.querySelector("#connection-machine-id")?.value || state.machineId;
      localStorage.setItem("redrhex_machine_id", state.machineId);
      await refresh({ silent: true });
      return setMessage("Machine target saved.");
    }
    if (action === "queue-training") return await queueTraining();
    if (action === "tweak-last-run") return tweakFromLastRun();
    if (action === "select-preset") {
      state.selectedPresetId = target.dataset.id;
      state.draftPreset = null;
      localStorage.setItem("redrhex_child_preset", state.selectedPresetId);
      return render();
    }
    if (action === "select-terrain-preset") {
      state.selectedTerrainPresetId = target.dataset.id;
      state.draftTerrainPreset = null;
      localStorage.setItem("redrhex_child_terrain_preset", state.selectedTerrainPresetId);
      return render();
    }
    if (action === "duplicate-preset") return duplicatePreset();
    if (action === "new-preset") return newPreset();
    if (action === "save-preset") return await savePreset();
    if (action === "delete-preset") return await deletePreset();
    if (action === "duplicate-terrain-preset") return duplicateTerrainPreset();
    if (action === "new-terrain-preset") return newTerrainPreset();
    if (action === "save-terrain-preset") return await saveTerrainPreset();
    if (action === "delete-terrain-preset") return await deleteTerrainPreset();
    if (action === "toggle-editor-group") {
      const group = target.closest(".editor-group");
      group?.classList.toggle("collapsed");
      updateGroupToggleLabel(group?.dataset.kind || "reward");
      return;
    }
    if (action === "toggle-all-groups") return toggleAllGroups(target.dataset.kind || "reward");
    if (action === "open-folder") return openHistoryFolder(target.dataset.folder || "all");
    if (action === "open-folder-root") return openHistoryFolder("all");
    if (action === "select-run") {
      const previousDraftRunId = persistActiveRunMetadataDraft({ flush: true });
      if (state.view === "history") {
        const opened = !(state.isPhone && state.selectedRunId === target.dataset.id);
        if (state.isPhone && syncHistoryAccordion(target)) {
          if (opened) {
            await ensureSelectedVideoSigned();
            patchSelectedRunDetails();
          }
        } else {
          state.selectedRunId = target.dataset.id;
          if (previousDraftRunId !== state.selectedRunId && !state.runDrafts[state.selectedRunId]) {
            setRunMetadataSaveStatus("saved");
          }
          await ensureSelectedVideoSigned();
          patchHistory({ forceList: true });
        }
        patchShellStatus();
        return;
      }
      state.selectedRunId = target.dataset.id;
      await ensureSelectedVideoSigned();
      return render();
    }
    if (action === "save-run") return await saveRun(target.dataset.runId || state.selectedRunId);
    if (action === "load-video") return await loadVideo(target.dataset.path);
    if (action === "copy-video-path") {
      await navigator.clipboard.writeText(target.dataset.path || "");
      return setMessage("Video storage path copied.");
    }
    if (action === "check-video") return await checkOrCreateVideo();
    if (action === "tweak-run") return tweakFromRun(selectedRun()?.id || "");
    if (action === "job-tensorboard") return await queueTensorBoard();
    if (action === "job-compact-run") return await compactSelectedRun();
    if (action === "save-notification-settings") return await saveNotificationSettings();
    if (action === "send-test-notification") return await sendTestNotification();
  } catch (error) {
    setMessage(friendlyErrorMessage(error));
  }
});

document.addEventListener("change", (event) => {
  if (["task", "num-envs", "max-iterations", "device", "seed"].includes(event.target.id)) {
    state.trainForm = {
      task: document.querySelector("#task")?.value || "Template-Redrhex-Direct-v0",
      num_envs: Number(document.querySelector("#num-envs")?.value || 4),
      max_iterations: Number(document.querySelector("#max-iterations")?.value || 8),
      device: document.querySelector("#device")?.value || "cuda:0",
      seed: document.querySelector("#seed")?.value || "",
    };
  }
  if (event.target.id === "train-preset") {
    state.selectedPresetId = event.target.value;
    localStorage.setItem("redrhex_child_preset", state.selectedPresetId);
    const snapshot = document.querySelector("#train-preset-snapshot");
    if (snapshot) {
      snapshot.innerHTML = trainPresetSnapshot(selectedPreset(), selectedTerrainPresetForTraining());
    } else {
      render();
    }
  }
  if (event.target.id === "train-terrain-preset") {
    state.selectedTerrainPresetId = event.target.value;
    localStorage.setItem("redrhex_child_terrain_preset", state.selectedTerrainPresetId);
    const snapshot = document.querySelector("#train-preset-snapshot");
    if (snapshot) {
      snapshot.innerHTML = trainPresetSnapshot(selectedPreset(), selectedTerrainPresetForTraining());
    } else {
      render();
    }
  }
  if (event.target.id === "video-checkpoint-select") {
    const run = selectedRun();
    if (run) {
      state.videoCheckpointByRun[run.id] = event.target.value;
      patchTeamVideo(run);
      ensureSelectedVideoSigned().then(() => patchTeamVideo(run)).catch((error) => setMessage(friendlyErrorMessage(error)));
    }
  }
  if (event.target.id === "folder-filter") {
    state.folderFilter = event.target.value;
    renderRunListOnly();
  }
  if (event.target.classList.contains("terrain-input") && state.draftTerrainPreset) {
    state.draftTerrainPreset.values[event.target.dataset.key] = parseTerrainValue(event.target);
  }
  if (event.target.id?.startsWith("notify-") || event.target.classList.contains("notify-event-toggle")) {
    state.notificationSettings = normalizeNotificationSettings(collectNotificationSettings());
    state.notificationSaveStatus = "dirty";
    patchConnection();
  }
});

document.addEventListener("input", (event) => {
  if (event.target.id === "run-search") {
    state.runSearch = event.target.value;
    renderRunListOnly();
  }
  if (["run-name", "run-folder", "run-notes"].includes(event.target.id)) {
    const run = runById(event.target.dataset.runId) || selectedRun({ fallback: false });
    if (run) {
      updateRunMetadataDraft(run.id);
      scheduleRunMetadataAutosave(run.id);
    }
  }
  if (event.target.id === "preset-name" && state.draftPreset) {
    state.draftPreset.name = event.target.value;
  }
  if (event.target.id === "preset-name" && state.tweakDraftPreset && state.selectedPresetId === state.tweakDraftPreset.id) {
    state.tweakDraftPreset.name = event.target.value;
  }
  if (event.target.id === "preset-description" && state.draftPreset) {
    state.draftPreset.description = event.target.value;
  }
  if (event.target.id === "preset-description" && state.tweakDraftPreset && state.selectedPresetId === state.tweakDraftPreset.id) {
    state.tweakDraftPreset.description = event.target.value;
  }
  if (["preset-name", "preset-description"].includes(event.target.id)) {
    schedulePresetMetadataAutosave("reward");
  }
  if (event.target.id === "terrain-preset-name" && state.draftTerrainPreset) {
    state.draftTerrainPreset.name = event.target.value;
  }
  if (event.target.id === "terrain-preset-description" && state.draftTerrainPreset) {
    state.draftTerrainPreset.description = event.target.value;
  }
  if (["terrain-preset-name", "terrain-preset-description"].includes(event.target.id)) {
    schedulePresetMetadataAutosave("terrain");
  }
  if (event.target.classList.contains("reward-input") && state.draftPreset) {
    state.draftPreset.values[event.target.dataset.key] = Number(event.target.value || 0);
  }
  if (event.target.classList.contains("reward-input") && state.tweakDraftPreset && state.selectedPresetId === state.tweakDraftPreset.id) {
    state.tweakDraftPreset.values[event.target.dataset.key] = Number(event.target.value || 0);
  }
  if (event.target.classList.contains("terrain-input") && state.draftTerrainPreset) {
    state.draftTerrainPreset.values[event.target.dataset.key] = parseTerrainValue(event.target);
  }
  if (event.target.id === "notify-discord-webhook") {
    state.notificationSettings = normalizeNotificationSettings(collectNotificationSettings());
    state.notificationSaveStatus = "dirty";
  }
});

document.addEventListener("focusout", (event) => {
  if (["run-name", "run-folder", "run-notes"].includes(event.target.id)) {
    const runId = event.target.dataset.runId || state.selectedRunId;
    updateRunMetadataDraft(runId);
    flushRunMetadataAutosave(runId).catch((error) => {
      setRunMetadataSaveStatus("error", runId);
      setMessage(friendlyErrorMessage(error));
    });
  }
  if (["preset-name", "preset-description"].includes(event.target.id)) {
    flushPresetMetadataAutosave("reward").catch((error) => setMessage(friendlyErrorMessage(error)));
  }
  if (["terrain-preset-name", "terrain-preset-description"].includes(event.target.id)) {
    flushPresetMetadataAutosave("terrain").catch((error) => setMessage(friendlyErrorMessage(error)));
  }
});

document.addEventListener("keydown", (event) => {
  if (["run-name", "run-folder", "run-notes"].includes(event.target.id)) {
    const shouldSave = event.target.id === "run-notes"
      ? (event.ctrlKey || event.metaKey) && event.key === "Enter"
      : event.key === "Enter";
    if (!shouldSave) return;
    event.preventDefault();
    const runId = event.target.dataset.runId || state.selectedRunId;
    updateRunMetadataDraft(runId);
    flushRunMetadataAutosave(runId).catch((error) => {
      setRunMetadataSaveStatus("error", runId);
      setMessage(friendlyErrorMessage(error));
    });
    event.target.blur();
  }
  if (["preset-name", "terrain-preset-name"].includes(event.target.id) && event.key === "Enter") {
    event.preventDefault();
    flushPresetMetadataAutosave(event.target.id === "terrain-preset-name" ? "terrain" : "reward")
      .catch((error) => setMessage(friendlyErrorMessage(error)));
    event.target.blur();
  }
  if (["preset-description", "terrain-preset-description"].includes(event.target.id) && (event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    flushPresetMetadataAutosave(event.target.id === "terrain-preset-description" ? "terrain" : "reward")
      .catch((error) => setMessage(friendlyErrorMessage(error)));
    event.target.blur();
  }
});

boot().catch((error) => {
  state.loadError = friendlyErrorMessage(error);
  render();
});

window.addEventListener("focus", () => {
  if (state.user) {
    refresh({ silent: true }).catch((error) => {
      state.loadError = friendlyErrorMessage(error);
      patchCurrentView();
    });
  }
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && state.user) {
    refresh({ silent: true }).catch((error) => {
      state.loadError = friendlyErrorMessage(error);
      patchCurrentView();
    });
  } else if (state.refreshTimer) {
    clearTimeout(state.refreshTimer);
  }
});
