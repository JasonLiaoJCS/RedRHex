const state = {
  selectedRun: null,
  runs: [],
};

const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
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
  data.resume = form.elements.resume.checked;
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
      return `
        <article class="run-card ${active}" data-run-id="${run.id}">
          <div class="run-top">
            <strong>${run.id}</strong>
            <span class="pill">${run.status || "unknown"}</span>
          </div>
          <small>${run.created_at || ""}</small>
          <small>${run.log_dir || "No RSL-RL log linked yet"}</small>
          <small>${checkpoint}${run.has_notes ? " · notes" : ""}</small>
        </article>
      `;
    })
    .join("");
  document.querySelectorAll(".run-card").forEach((card) => {
    card.addEventListener("click", () => selectRun(card.dataset.runId));
  });
}

async function selectRun(runId) {
  state.selectedRun = state.runs.find((run) => run.id === runId);
  $("#notes-title").textContent = `Notes: ${runId}`;
  const data = await api(`/api/runs/${encodeURIComponent(runId)}/notes`);
  $("#notes-editor").value = data.notes;
  $("#notes-status").textContent = state.selectedRun.latest_checkpoint || "No checkpoint available yet.";
  await loadRuns();
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

async function startTensorBoard(host) {
  const data = await api("/api/tensorboard/start", {
    method: "POST",
    body: JSON.stringify({ host, port: 6006 }),
  });
  const url = host === "0.0.0.0" ? `http://${location.hostname}:6006` : data.url;
  window.open(url, "_blank", "noopener");
}

async function playSelectedRun() {
  if (!state.selectedRun) {
    $("#notes-status").textContent = "Select a run first.";
    return;
  }
  const data = await api(`/api/runs/${encodeURIComponent(state.selectedRun.id)}/play`, {
    method: "POST",
    body: JSON.stringify({ device: "cuda:0" }),
  });
  $("#notes-status").textContent = `Started play process ${data.pid}.`;
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

async function refreshAll() {
  await Promise.all([loadSystem(), loadTweaks(), loadRuns()]);
}

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});

$("#train-form").addEventListener("submit", startTraining);
$("#smoke-button").addEventListener("click", () => applyPreset("smoke"));
$("#debug-button").addEventListener("click", () => applyPreset("debug"));
$("#refresh-button").addEventListener("click", refreshAll);
$("#save-notes").addEventListener("click", saveNotes);
$("#play-run").addEventListener("click", playSelectedRun);
$("#tensorboard-local").addEventListener("click", () => startTensorBoard("127.0.0.1"));
$("#tensorboard-lan").addEventListener("click", () => startTensorBoard("0.0.0.0"));

refreshAll();

