import { DEFAULT_MACHINE_ID, SUPABASE_URL } from "./config.js";
import {
  createSignedVideoUrl,
  currentUser,
  insert,
  loadRemoteSnapshot,
  select,
  signIn,
  signOut,
  update,
  upsert,
} from "./api.js";
import {
  REWARD_FIELDS,
  buildRunMetadataPatch,
  buildActionJob,
  buildTrainingJob,
  canEditPreset,
  canEditRun,
  canOperate,
  escapeHtml,
  friendlyErrorMessage,
  formatRelativeTime,
  hasActiveRemoteWork,
  jobQueueLabel,
  machineState,
  normalizePreset,
  refreshDelayForSnapshot,
  shouldReplaceVideoPanel,
  slugify,
  statusTone,
  videoStateForRun,
} from "./core.js";

const state = {
  user: null,
  profile: null,
  view: localStorage.getItem("redrhex_child_view") || "dashboard",
  machineId: localStorage.getItem("redrhex_machine_id") || DEFAULT_MACHINE_ID,
  snapshot: {
    machines: [],
    machine: null,
    jobs: [],
    runs: [],
    artifacts: [],
    presets: [],
    schema: { artifacts: true, rewardPresets: true, warnings: [] },
  },
  selectedPresetId: localStorage.getItem("redrhex_child_preset") || "baseline",
  draftPreset: null,
  selectedRunId: "",
  runSearch: "",
  folderFilter: "all",
  signedVideos: {},
  runDrafts: {},
  message: "",
  loading: false,
  loadError: "",
  lastUpdated: "",
  refreshTimer: null,
  refreshing: false,
};

const app = document.querySelector("#app");

function role() {
  return state.profile?.role || "viewer";
}

function selectedPreset() {
  const presets = state.snapshot.presets.map(normalizePreset);
  return presets.find((preset) => preset.id === state.selectedPresetId) || presets[0] || normalizePreset({ id: "baseline", name: "Baseline" });
}

function selectedRun() {
  return state.snapshot.runs.find((run) => run.id === state.selectedRunId) || state.snapshot.runs[0] || null;
}

function setMessage(message, options = {}) {
  state.message = message;
  if (options.forceRender || !app.querySelector("#message-notice")) {
    render();
  } else {
    patchShellStatus();
  }
}

function setView(view) {
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

function signedVideoEntry(storagePath) {
  const entry = state.signedVideos[storagePath];
  if (!entry) return null;
  if (typeof entry === "string") return { url: entry, expiresAt: 0 };
  return entry;
}

async function ensureSelectedVideoSigned() {
  const run = selectedRun();
  if (!run) return;
  const video = videoStateForRun(run, state.snapshot.artifacts);
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
    state.snapshot = await loadRemoteSnapshot(state.machineId);
    state.snapshot.presets = state.snapshot.presets.map(normalizePreset);
    if (!state.selectedRunId && state.snapshot.runs[0]) state.selectedRunId = state.snapshot.runs[0].id;
    if (!state.snapshot.presets.find((preset) => preset.id === state.selectedPresetId) && state.snapshot.presets[0]) {
        state.selectedPresetId = state.snapshot.presets[0].id;
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
    ["video", "Video storage", Boolean(state.snapshot.schema?.artifacts), state.snapshot.schema?.artifacts ? "Private signed playback ready" : "Apply schema.sql in Supabase"],
  ];
}

function shell() {
  const views = [
    ["dashboard", "Dashboard"],
    ["train", "Train"],
    ["rewards", "Rewards"],
    ["history", "History"],
    ["connection", "Connection"],
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
      </div>
    </header>
    <nav class="nav-tabs">
      ${views.map(([id, label]) => `<button class="${state.view === id ? "active" : ""}" data-action="view" data-view="${id}">${label}</button>`).join("")}
    </nav>
    <div id="message-notice" class="notice" ${state.message ? "" : "hidden"}>${escapeHtml(state.message)}</div>
    <div id="schema-warnings">${(state.snapshot.schema?.warnings || []).map((warning) => `<div class="notice warning">${escapeHtml(warning)}</div>`).join("")}</div>
    <div id="load-error-notice" class="notice danger" ${state.loadError ? "" : "hidden"}>${escapeHtml(state.loadError)}</div>
    ${state.user ? page() : loginPage()}
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
  if (state.view === "history") return historyView();
  if (state.view === "connection") return connectionView();
  return dashboardView();
}

function dashboardView() {
  const machine = state.snapshot.targetMachine || state.snapshot.machine;
  const latestRuns = state.snapshot.runs.slice(0, 5);
  const jobs = state.snapshot.jobs.slice(0, 6);
  return `
    <section class="dashboard-grid">
      <article class="panel span-2">
        <div class="section-head">
          <div>
            <h2>Connection Health</h2>
            <p class="muted">A quick read on whether the child can talk to mother through Supabase.</p>
          </div>
          <button data-action="refresh">${state.loading ? "Refreshing" : "Refresh"}</button>
        </div>
        <div id="health-grid" class="health-grid">
          ${healthChecks().map(([, label, ok, detail]) => `
            <div class="health-card ${ok ? "ok" : "warn"}">
              <span>${escapeHtml(label)}</span>
              <strong>${ok ? "OK" : "Needs attention"}</strong>
              <small>${escapeHtml(detail)}</small>
            </div>`).join("")}
        </div>
      </article>
      <article class="panel">
        <h2>Machine</h2>
        <div id="machine-card-slot">${machineCard(machine)}</div>
      </article>
      <article class="panel">
        <h2>Queue</h2>
        <div id="queue-summary-slot">${jobSummary(jobs)}</div>
      </article>
      <article class="panel span-2">
        <h2>Latest Runs</h2>
        <div id="latest-runs-slot" class="run-strip">${latestRuns.map(runCard).join("") || empty("No runs synced yet.")}</div>
      </article>
    </section>
  `;
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
  const disabled = !canOperate(role());
  return `
    <section class="split-grid">
      <article class="panel">
        <h2>Launch Training</h2>
        <p class="muted">Queues a job for mother. The worker will run one Isaac/GPU action at a time.</p>
        <label>Machine ID <input id="machine-id" value="${escapeHtml(state.machineId)}"></label>
        <label>Task <input id="task" value="Template-Redrhex-Direct-v0"></label>
        <div class="input-row">
          <label>Envs <input id="num-envs" type="number" min="1" max="8192" value="4"></label>
          <label>Iterations <input id="max-iterations" type="number" min="1" max="100000" value="8"></label>
        </div>
        <label>Device <input id="device" value="cuda:0"></label>
        <label>Reward Preset
          <select id="train-preset">
            ${state.snapshot.presets.map((item) => `<option value="${escapeHtml(item.id)}" ${item.id === preset.id ? "selected" : ""}>${escapeHtml(item.name)}</option>`).join("")}
          </select>
        </label>
        <button class="primary wide" data-action="queue-training" ${disabled ? "disabled" : ""}>Queue Training</button>
        ${disabled ? `<p class="muted">Viewer accounts can inspect but cannot launch training.</p>` : ""}
      </article>
      <article class="panel">
        <div id="train-preset-snapshot">${trainPresetSnapshot(preset)}</div>
      </article>
    </section>
  `;
}

function trainPresetSnapshot(preset) {
  return `
    <h2>Preset Snapshot</h2>
    <h3>${escapeHtml(preset.name)}</h3>
    <p class="muted">${escapeHtml(preset.description || "No description.")}</p>
    ${rewardSnapshot(preset.values)}
  `;
}

function rewardSnapshot(values) {
  const entries = Object.entries(values || {});
  if (!entries.length) return empty("Baseline uses the current local defaults from mother.");
  return `<div class="reward-snapshot">${entries.map(([key, value]) => `
    <div><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}</div>`;
}

function rewardsView() {
  const preset = state.draftPreset || selectedPreset();
  const rewardSchemaReady = Boolean(state.snapshot.schema?.rewardPresets);
  const editable = rewardSchemaReady && canEditPreset(role()) && !preset.built_in;
  return `
    <section class="rewards-page">
      <aside class="panel preset-list rewards-rail">
        <div class="section-head compact reward-rail-head">
          <div>
            <h2>Presets</h2>
            <p class="muted">Shared reward recipes</p>
          </div>
          <button class="icon-action" title="New preset" data-action="new-preset" ${!rewardSchemaReady || !canEditPreset(role()) ? "disabled" : ""}>+</button>
        </div>
        ${rewardSchemaReady ? "" : `<p class="muted">Using built-in fallback presets until Supabase schema is updated.</p>`}
        <div class="preset-scroll">
          ${state.snapshot.presets.map((item) => `
          <button class="preset-button ${item.id === preset.id ? "active" : ""}" data-action="select-preset" data-id="${escapeHtml(item.id)}">
            <strong>${escapeHtml(item.name)}</strong>
            <small>${item.built_in ? "Built-in" : "Team preset"} · ${escapeHtml(formatRelativeTime(item.updated_at))}</small>
          </button>`).join("") || empty("Apply the V2.1 schema to create presets.")}
        </div>
      </aside>
      <article class="panel reward-workspace">
        <div class="section-head reward-head">
          <div>
            <h2>Reward Tuning</h2>
            <p class="muted">${escapeHtml(preset.built_in ? "Built-in preset. Duplicate it before editing." : editable ? "Editable team preset." : "Read-only preset.")}</p>
          </div>
          <div class="button-row">
            <button data-action="duplicate-preset" ${!rewardSchemaReady || !canEditPreset(role()) ? "disabled" : ""}>Duplicate</button>
            <button class="primary" data-action="save-preset" ${!editable ? "disabled" : ""}>Save Preset</button>
          </div>
        </div>
        <div class="preset-meta-grid">
          <label>Name <input id="preset-name" value="${escapeHtml(preset.name)}" ${editable ? "" : "disabled"}></label>
          <label>Description <textarea id="preset-description" ${editable ? "" : "disabled"}>${escapeHtml(preset.description)}</textarea></label>
        </div>
        <div class="reward-editor">
          ${REWARD_FIELDS.map((group) => `
            <section class="reward-group">
              <h3>${escapeHtml(group.name)}</h3>
              ${group.fields.map(([key, label, help]) => {
                const value = Number(preset.values?.[key] ?? 0);
                return `
                  <label class="reward-row">
                    <span><strong>${escapeHtml(label)}</strong><small>${escapeHtml(help)}</small><code>${escapeHtml(key)}</code></span>
                    <input class="reward-input" data-key="${escapeHtml(key)}" type="number" step="0.01" value="${escapeHtml(value)}" ${editable ? "" : "disabled"}>
                  </label>`;
              }).join("")}
            </section>`).join("")}
        </div>
      </article>
    </section>
  `;
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

function folders() {
  const set = new Set(state.snapshot.runs.map((run) => run.folder).filter(Boolean));
  return [...set].sort((a, b) => a.localeCompare(b));
}

function historyView() {
  const run = selectedRun();
  const runs = filteredRuns();
  return `
    <section class="history-layout">
      <aside class="panel run-list-panel">
        <div class="section-head compact">
          <h2>History</h2>
          <button data-action="refresh">Refresh</button>
        </div>
        <input id="run-search" placeholder="Search runs" value="${escapeHtml(state.runSearch)}">
        <select id="folder-filter">
          <option value="all" ${state.folderFilter === "all" ? "selected" : ""}>All runs</option>
          <option value="uncategorized" ${state.folderFilter === "uncategorized" ? "selected" : ""}>Uncategorized</option>
          ${folders().map((folder) => `<option value="${escapeHtml(folder)}" ${state.folderFilter === folder ? "selected" : ""}>${escapeHtml(folder)}</option>`).join("")}
        </select>
        <div class="run-list">${runs.map(runCard).join("") || empty("No matching runs.")}</div>
      </aside>
      <article class="panel run-details">
        ${run ? runDetails(run) : empty("Select a run.")}
      </article>
    </section>
  `;
}

function renderRunListOnly() {
  patchRunList();
}

function runCard(run) {
  const video = videoStateForRun(run, state.snapshot.artifacts);
  const active = run.id === state.selectedRunId;
  return `
    <button class="run-card ${active ? "active" : ""}" data-action="select-run" data-id="${escapeHtml(run.id)}">
      <span class="run-card-top">
        <strong>${escapeHtml(run.display_name || run.id)}</strong>
        <span class="badge ${statusTone(run.status)}">${escapeHtml(run.status || "unknown")}</span>
      </span>
      <small>${escapeHtml(run.folder || "Uncategorized")} - ${escapeHtml(formatRelativeTime(run.created_at))}</small>
      <small>${run.latest_checkpoint ? "checkpoint ready" : "no checkpoint"} - video ${escapeHtml(video.state)}${run.onnx_path ? " - ONNX" : ""}</small>
    </button>
  `;
}

function runDetailsGrid(run) {
  const video = videoStateForRun(run, state.snapshot.artifacts);
  return `
    <div><span>Checkpoint</span><strong>${run.latest_checkpoint ? "ready" : "missing"}</strong></div>
    <div><span>Video</span><strong>${escapeHtml(video.state)}</strong></div>
    <div><span>ONNX</span><strong>${run.onnx_path ? "ready" : "missing"}</strong></div>
    <div><span>Updated</span><strong>${escapeHtml(formatRelativeTime(run.updated_at || run.created_at))}</strong></div>
  `;
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
        <div><strong>${escapeHtml(job.type)}</strong><span class="badge ${statusTone(job.status)}">${escapeHtml(job.status)}</span><small>${escapeHtml(jobQueueLabel(job, state.snapshot.targetMachine || state.snapshot.machine))}</small><small>${escapeHtml(formatRelativeTime(job.created_at))}</small></div>
      `).join("")}</div>` : empty("No remote jobs linked to this run yet.")}
    </section>
  `;
}

function teamVideoSection(run) {
  const video = videoStateForRun(run, state.snapshot.artifacts);
  const videoArtifact = video.artifact;
  const signed = videoArtifact ? signedVideoEntry(videoArtifact.storage_path)?.url || "" : "";
  const runnable = canOperate(role()) && Boolean(run.latest_checkpoint);
  const storagePath = videoArtifact?.storage_path || "";
  return `
    <section id="team-video-panel" class="subpanel" data-video-state="${escapeHtml(video.state)}" data-storage-path="${escapeHtml(storagePath)}">
      <h3>Team Video</h3>
      ${video.state === "ready" ? `
        ${signed ? `<video controls src="${escapeHtml(signed)}"></video>` : `<p class="muted">Preparing a signed team-only video link...</p>`}
        <div class="button-row">
          <button data-action="load-video" data-path="${escapeHtml(videoArtifact.storage_path)}">Load Video</button>
          <button data-action="copy-video-path" data-path="${escapeHtml(videoArtifact.storage_path)}">Copy Storage Path</button>
        </div>` : video.state === "uploading" ? `
        <p class="muted">Video exists locally and is uploading to team storage. This panel will refresh automatically.</p>
      ` : video.state === "recordable" ? `
        <p class="muted">No team video yet. Record one from the latest checkpoint.</p>
        <button class="primary" data-action="job-record-video" ${runnable ? "" : "disabled"}>Record Video</button>
      ` : `<p class="muted">No checkpoint yet, so video recording is not available.</p>`}
    </section>
  `;
}

function runDetails(run) {
  const draft = currentRunDraft(run);
  const editable = canEditRun(role());
  const runnable = canOperate(role()) && Boolean(run.latest_checkpoint);
  return `
    <div class="section-head">
      <div>
        <h2 id="selected-run-title">${escapeHtml(run.display_name || run.id)}</h2>
        <p id="selected-run-id" class="muted">${escapeHtml(run.id)}</p>
      </div>
      <span id="selected-run-status" class="badge ${statusTone(run.status)}">${escapeHtml(run.status || "unknown")}</span>
    </div>
    <div id="run-details-grid" class="details-grid">
      ${runDetailsGrid(run)}
    </div>
    <section class="subpanel">
      <h3>Run Metadata</h3>
      <label>Name <input id="run-name" value="${escapeHtml(draft.display_name ?? run.display_name ?? "")}" ${editable ? "" : "disabled"}></label>
      <label>Folder <input id="run-folder" value="${escapeHtml(draft.folder ?? run.folder ?? "")}" placeholder="e.g. gait tests" ${editable ? "" : "disabled"}></label>
      <label>Notes <textarea id="run-notes" ${editable ? "" : "disabled"}>${escapeHtml(draft.notes ?? run.notes ?? "")}</textarea></label>
      <button data-action="save-run" ${editable ? "" : "disabled"}>Save Notes / Folder</button>
    </section>
    ${teamVideoSection(run)}
    <section class="subpanel">
      <h3>Safe Remote Actions</h3>
      <div class="button-row wrap">
        <button data-action="job-record-video" ${runnable ? "" : "disabled"}>Record Video</button>
        <button data-action="job-export-onnx" ${runnable ? "" : "disabled"}>Export ONNX</button>
        <button data-action="job-stop" ${canOperate(role()) ? "" : "disabled"}>Stop Active Process</button>
      </div>
    </section>
    ${relatedJobsSection(run)}
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
  const machine = state.snapshot.targetMachine || state.snapshot.machine;
  const healthGrid = document.querySelector("#health-grid");
  if (healthGrid) {
    healthGrid.innerHTML = healthChecks().map(([, label, ok, detail]) => `
      <div class="health-card ${ok ? "ok" : "warn"}">
        <span>${escapeHtml(label)}</span>
        <strong>${ok ? "OK" : "Needs attention"}</strong>
        <small>${escapeHtml(detail)}</small>
      </div>`).join("");
  }
  const machineSlot = document.querySelector("#machine-card-slot");
  if (machineSlot) machineSlot.innerHTML = machineCard(machine);
  const queueSlot = document.querySelector("#queue-summary-slot");
  if (queueSlot) queueSlot.innerHTML = jobSummary(state.snapshot.jobs.slice(0, 6));
  const latestRuns = document.querySelector("#latest-runs-slot");
  if (latestRuns) latestRuns.innerHTML = state.snapshot.runs.slice(0, 5).map(runCard).join("") || empty("No runs synced yet.");
}

function patchRunList() {
  const list = document.querySelector(".run-list");
  if (!list) return;
  const panel = document.querySelector(".run-list-panel");
  const panelScrollTop = panel?.scrollTop || 0;
  const scrollTop = list.scrollTop;
  list.innerHTML = filteredRuns().map(runCard).join("") || empty("No matching runs.");
  list.scrollTop = scrollTop;
  if (panel) panel.scrollTop = panelScrollTop;
}

function isRunMetadataFocused() {
  return ["run-name", "run-folder", "run-notes"].includes(document.activeElement?.id);
}

function patchTeamVideo(run) {
  const panel = document.querySelector("#team-video-panel");
  if (!panel || !run) return;
  const video = videoStateForRun(run, state.snapshot.artifacts);
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
  const details = document.querySelector(".run-details");
  const run = selectedRun();
  if (!details || !run) return;

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

  if (!isRunMetadataFocused() && !state.runDrafts[run.id]) {
    const name = document.querySelector("#run-name");
    const folder = document.querySelector("#run-folder");
    const notes = document.querySelector("#run-notes");
    if (name) name.value = run.display_name || "";
    if (folder) folder.value = run.folder || "";
    if (notes) notes.value = run.notes || "";
  }

  patchTeamVideo(run);
  const related = document.querySelector("#related-jobs-panel");
  if (related) related.outerHTML = relatedJobsSection(run);
}

function patchHistory() {
  patchRunList();
  patchSelectedRunDetails();
}

function patchConnection() {
  const connectionGrid = document.querySelector(".connection-grid");
  if (connectionGrid && !["connection-machine-id"].includes(document.activeElement?.id)) {
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
  const presetId = document.querySelector("#train-preset")?.value || state.selectedPresetId;
  state.selectedPresetId = presetId;
  localStorage.setItem("redrhex_child_preset", presetId);
  const preset = selectedPreset();
  const params = {
    task: document.querySelector("#task")?.value || "Template-Redrhex-Direct-v0",
    num_envs: Number(document.querySelector("#num-envs")?.value || 4),
    max_iterations: Number(document.querySelector("#max-iterations")?.value || 8),
    device: document.querySelector("#device")?.value || "cuda:0",
  };
  const job = buildTrainingJob({ machineId: state.machineId, params, preset, role: role(), userId: state.user?.id });
  await insert("jobs", job);
  await refresh({ silent: true });
  setMessage(`Queued training with ${preset.name}.`);
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
  state.selectedPresetId = payload.id;
  await refresh();
  setMessage(`Saved preset ${payload.name}.`);
}

function duplicatePreset() {
  const source = state.draftPreset || selectedPreset();
  const id = slugify(`${source.id || source.name}-copy`);
  state.draftPreset = {
    ...source,
    id,
    name: `${source.name} Copy`,
    built_in: false,
    values: { ...(source.values || {}) },
    created_by: state.user?.id || null,
  };
  state.selectedPresetId = id;
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

async function saveRun() {
  const run = selectedRun();
  if (!run) return;
  const displayName = (document.querySelector("#run-name")?.value || "").trim();
  const folder = (document.querySelector("#run-folder")?.value || "").trim();
  const notes = document.querySelector("#run-notes")?.value || "";
  await update(
    "runs",
    `id=eq.${encodeURIComponent(run.id)}`,
    buildRunMetadataPatch({ displayName, folder, notes }),
  );
  delete state.runDrafts[run.id];
  await refresh({ silent: true });
  setMessage("Run metadata saved.");
}

async function queueRunAction(type, message) {
  const run = selectedRun();
  if (!run) return;
  const job = buildActionJob({ machineId: state.machineId, type, runId: run.id, role: role(), userId: state.user?.id });
  await insert("jobs", job);
  await refresh({ silent: true });
  setMessage(message);
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

function render() {
  app.innerHTML = shell();
}

document.addEventListener("click", async (event) => {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  event.preventDefault();
  const action = target.dataset.action;
  try {
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
    if (action === "select-preset") {
      state.selectedPresetId = target.dataset.id;
      state.draftPreset = null;
      localStorage.setItem("redrhex_child_preset", state.selectedPresetId);
      return render();
    }
    if (action === "duplicate-preset") return duplicatePreset();
    if (action === "new-preset") return newPreset();
    if (action === "save-preset") return await savePreset();
    if (action === "select-run") {
      state.selectedRunId = target.dataset.id;
      await ensureSelectedVideoSigned();
      if (state.view === "history") {
        patchHistory();
        patchShellStatus();
        return;
      }
      return render();
    }
    if (action === "save-run") return await saveRun();
    if (action === "load-video") return await loadVideo(target.dataset.path);
    if (action === "copy-video-path") {
      await navigator.clipboard.writeText(target.dataset.path || "");
      return setMessage("Video storage path copied.");
    }
    if (action === "job-record-video") return await queueRunAction("record_video", "Queued video recording.");
    if (action === "job-export-onnx") return await queueRunAction("export_onnx", "Queued ONNX export.");
    if (action === "job-stop") return await queueRunAction("stop_process", "Queued stop request.");
  } catch (error) {
    setMessage(friendlyErrorMessage(error));
  }
});

document.addEventListener("change", (event) => {
  if (event.target.id === "train-preset") {
    state.selectedPresetId = event.target.value;
    localStorage.setItem("redrhex_child_preset", state.selectedPresetId);
    const snapshot = document.querySelector("#train-preset-snapshot");
    if (snapshot) {
      snapshot.innerHTML = trainPresetSnapshot(selectedPreset());
    } else {
      render();
    }
  }
  if (event.target.id === "folder-filter") {
    state.folderFilter = event.target.value;
    renderRunListOnly();
  }
});

document.addEventListener("input", (event) => {
  if (event.target.id === "run-search") {
    state.runSearch = event.target.value;
    renderRunListOnly();
  }
  if (["run-name", "run-folder", "run-notes"].includes(event.target.id)) {
    const run = selectedRun();
    if (run) {
      state.runDrafts[run.id] = {
        ...state.runDrafts[run.id],
        display_name: document.querySelector("#run-name")?.value ?? run.display_name ?? "",
        folder: document.querySelector("#run-folder")?.value ?? run.folder ?? "",
        notes: document.querySelector("#run-notes")?.value ?? run.notes ?? "",
      };
    }
  }
  if (event.target.id === "preset-name" && state.draftPreset) {
    state.draftPreset.name = event.target.value;
  }
  if (event.target.id === "preset-description" && state.draftPreset) {
    state.draftPreset.description = event.target.value;
  }
  if (event.target.classList.contains("reward-input") && state.draftPreset) {
    state.draftPreset.values[event.target.dataset.key] = Number(event.target.value || 0);
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
