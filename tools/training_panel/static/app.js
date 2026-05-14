const state = {
  selectedRun: null,
  runs: [],
};

const $ = (selector) => document.querySelector(selector);

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

function setView(name) {
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === name);
  });
  const titles = {
    train: ["Train", "Start a controlled RSL-RL run with the repo defaults."],
    tweaks: ["Rewards", "Find reward scales and tuning files without editing them in V1."],
    history: ["History", "Review runs, notes, checkpoints, TensorBoard, and playbacks."],
    access: ["Access", "Run locally, expose on LAN, or use SSH tunneling."],
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
  return data;
}

async function loadSystem() {
  const system = await api("/api/system");
  $("#system-info").textContent = JSON.stringify(system, null, 2);
}

async function loadTweaks() {
  const data = await api("/api/tweakables");
  $("#tweak-files").innerHTML = data.files
    .map(
      (file) => `
        <article class="card">
          <strong>${file.title}</strong>
          <small>${file.why}</small>
          <small>${file.absolute_path}</small>
          <span class="pill">${file.exists ? "found" : "missing"}</span>
        </article>
      `
    )
    .join("");
  $("#reward-scales").innerHTML = data.reward_scales
    .map(
      (scale) => `
        <div class="scale-row">
          <div><strong>${scale.name}</strong><small>${scale.relative_path}:${scale.line}</small></div>
          <code>${scale.value}</code>
          <small>${scale.comment || "No inline note yet."}</small>
        </div>
      `
    )
    .join("");
}

async function loadRuns() {
  const data = await api("/api/runs");
  state.runs = data.runs;
  $("#runs").innerHTML = data.runs
    .map((run) => {
      const active = state.selectedRun && state.selectedRun.id === run.id ? "active" : "";
      const checkpoint = run.latest_checkpoint ? "checkpoint" : "no checkpoint";
      const title = run.display_name || run.id;
      return `
        <article class="run-card ${active}" data-run-id="${escapeHtml(run.id)}">
          <div class="run-top">
            <strong>${escapeHtml(title)}</strong>
            <span class="pill">${escapeHtml(run.status || "unknown")}</span>
          </div>
          <small>${escapeHtml(run.id)}</small>
          <small>${escapeHtml(run.created_at || "")}</small>
          <small>${escapeHtml(run.log_dir || "No RSL-RL log linked yet")}</small>
          <small>${escapeHtml(checkpoint)}${run.has_notes ? " · notes" : ""}</small>
          <div class="run-actions">
            <button data-action="tensorboard" data-run-id="${escapeHtml(run.id)}">TensorBoard</button>
            <button data-action="play" data-run-id="${escapeHtml(run.id)}">Play</button>
            <button data-action="resume" data-run-id="${escapeHtml(run.id)}">Resume</button>
            <button data-action="debug" data-run-id="${escapeHtml(run.id)}">Debug</button>
          </div>
        </article>
      `;
    })
    .join("");
  document.querySelectorAll(".run-card").forEach((card) => {
    card.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-action]");
      if (button) {
        event.stopPropagation();
        handleRunAction(button.dataset.action, button.dataset.runId);
        return;
      }
      selectRun(card.dataset.runId);
    });
  });
}

async function selectRun(runId) {
  state.selectedRun = state.runs.find((run) => run.id === runId);
  $("#details-title").textContent = state.selectedRun.display_name || runId;
  $("#run-name").value = state.selectedRun.display_name || "";
  const data = await api(`/api/runs/${encodeURIComponent(runId)}/notes`);
  $("#notes-editor").value = data.notes;
  $("#notes-status").textContent = state.selectedRun.latest_checkpoint || "No checkpoint available yet.";
  await loadDebug(runId);
  await loadRuns();
}

async function loadDebug(runId) {
  const debug = await api(`/api/runs/${encodeURIComponent(runId)}/debug`);
  $("#debug-command").textContent = debug.command || debug.debug_hint || "";
  $("#debug-log").textContent = debug.process_log_tail || debug.debug_hint || "";
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
    await loadRuns();
  } catch (error) {
    $("#train-status").textContent = error.message;
  }
}

async function saveNotes() {
  if (!state.selectedRun) {
    $("#notes-status").textContent = "Select a run first.";
    return;
  }
  await api(`/api/runs/${encodeURIComponent(state.selectedRun.id)}/notes`, {
    method: "POST",
    body: JSON.stringify({ notes: $("#notes-editor").value }),
  });
  $("#notes-status").textContent = "Notes saved.";
  await loadRuns();
}

async function saveName() {
  if (!state.selectedRun) {
    $("#notes-status").textContent = "Select a run first.";
    return;
  }
  await api(`/api/runs/${encodeURIComponent(state.selectedRun.id)}/rename`, {
    method: "POST",
    body: JSON.stringify({ display_name: $("#run-name").value }),
  });
  $("#notes-status").textContent = "Name saved.";
  await loadRuns();
}

function tensorboardHost() {
  return location.hostname === "127.0.0.1" || location.hostname === "localhost" ? "127.0.0.1" : "0.0.0.0";
}

function displayTensorboardUrl(data, host) {
  return host === "0.0.0.0" ? `http://${location.hostname}:${data.port}` : data.url;
}

async function startTensorBoardForRun(runId) {
  const host = tensorboardHost();
  const data = await api(`/api/runs/${encodeURIComponent(runId)}/tensorboard`, {
    method: "POST",
    body: JSON.stringify({ host }),
  });
  $("#notes-status").textContent = data.already_running
    ? `TensorBoard is already running on port ${data.port}.`
    : `Started TensorBoard on port ${data.port}.`;
  window.open(displayTensorboardUrl(data, host), "_blank", "noopener");
}

async function playRun(runId) {
  const data = await api(`/api/runs/${encodeURIComponent(runId)}/play`, {
    method: "POST",
    body: JSON.stringify({ device: "cuda:0" }),
  });
  $("#notes-status").textContent = `Started play process ${data.pid}.`;
}

function resumeRun(runId) {
  const run = state.runs.find((item) => item.id === runId);
  if (!run || !run.latest_checkpoint) {
    $("#notes-status").textContent = "No checkpoint available for this run.";
    return;
  }
  const form = $("#train-form");
  form.elements.checkpoint.value = run.latest_checkpoint;
  setView("train");
  $("#train-status").textContent = `Resume selected from ${run.display_name || run.id}. Choose iterations/envs, then start training.`;
}

async function handleRunAction(action, runId) {
  if (!state.selectedRun || state.selectedRun.id !== runId) {
    await selectRun(runId);
  }
  try {
    if (action === "tensorboard") await startTensorBoardForRun(runId);
    if (action === "play") await playRun(runId);
    if (action === "resume") resumeRun(runId);
    if (action === "debug") {
      await loadDebug(runId);
      $("#notes-status").textContent = "Debug output loaded.";
    }
  } catch (error) {
    if (error.data && error.data.log_tail) {
      $("#debug-command").textContent = error.data.command || "";
      $("#debug-log").textContent = error.data.log_tail;
    }
    $("#notes-status").textContent = error.message;
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

async function refreshAll() {
  await Promise.all([loadSystem(), loadTweaks(), loadRuns()]);
}

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});

$("#train-form").addEventListener("submit", startTraining);
$("#smoke-button").addEventListener("click", () => applyPreset("smoke"));
$("#debug-button").addEventListener("click", () => applyPreset("debug"));
$("#clear-resume").addEventListener("click", clearResume);
$("#refresh-button").addEventListener("click", refreshAll);
$("#save-name").addEventListener("click", saveName);
$("#save-notes").addEventListener("click", saveNotes);
$("#resume-run").addEventListener("click", () => state.selectedRun && resumeRun(state.selectedRun.id));
$("#play-run").addEventListener("click", () => state.selectedRun && playRun(state.selectedRun.id));
$("#tensorboard-run").addEventListener("click", () => state.selectedRun && startTensorBoardForRun(state.selectedRun.id));

refreshAll();
