const state = {
  supabaseUrl: localStorage.getItem("redrhex_supabase_url") || "",
  anonKey: localStorage.getItem("redrhex_supabase_anon_key") || "",
  machineId: localStorage.getItem("redrhex_machine_id") || "",
  accessToken: localStorage.getItem("redrhex_access_token") || "",
  profile: null,
};

const $ = (selector) => document.querySelector(selector);

function status(message) {
  $("#status").textContent = message;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function hydrateForm() {
  $("#supabase-url").value = state.supabaseUrl;
  $("#supabase-anon-key").value = state.anonKey;
  $("#machine-id").value = state.machineId;
  $("#session-status").textContent = state.accessToken ? "Token saved" : "Signed out";
}

function saveConfig() {
  state.supabaseUrl = $("#supabase-url").value.replace(/\/$/, "");
  state.anonKey = $("#supabase-anon-key").value;
  state.machineId = $("#machine-id").value;
  localStorage.setItem("redrhex_supabase_url", state.supabaseUrl);
  localStorage.setItem("redrhex_supabase_anon_key", state.anonKey);
  localStorage.setItem("redrhex_machine_id", state.machineId);
  status("Remote config saved in this browser.");
}

async function supabaseFetch(path, options = {}) {
  if (!state.supabaseUrl || !state.anonKey) throw new Error("Configure Supabase URL and anon key first.");
  const response = await fetch(`${state.supabaseUrl}${path}`, {
    headers: {
      apikey: state.anonKey,
      Authorization: `Bearer ${state.accessToken || state.anonKey}`,
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) throw new Error(data?.message || data?.error_description || response.statusText);
  return data;
}

async function login() {
  saveConfig();
  const data = await supabaseFetch("/auth/v1/token?grant_type=password", {
    method: "POST",
    body: JSON.stringify({
      email: $("#login-email").value,
      password: $("#login-password").value,
    }),
  });
  state.accessToken = data.access_token;
  localStorage.setItem("redrhex_access_token", state.accessToken);
  $("#session-status").textContent = "Signed in";
  await loadProfile();
  await refreshAll();
}

async function loadProfile() {
  const user = await supabaseFetch("/auth/v1/user");
  const profiles = await supabaseFetch(`/rest/v1/profiles?id=eq.${encodeURIComponent(user.id)}&select=*`);
  state.profile = profiles[0] || { id: user.id, role: "viewer" };
  $("#session-status").textContent = `${user.email || "Signed in"} · ${state.profile.role || "viewer"}`;
}

async function queueTraining() {
  saveConfig();
  if (!state.accessToken) throw new Error("Login first.");
  const payload = {
    task: $("#task").value,
    num_envs: Number($("#num-envs").value),
    max_iterations: Number($("#max-iterations").value),
    device: $("#device").value,
    headless: true,
  };
  await supabaseFetch("/rest/v1/jobs", {
    method: "POST",
    headers: { Prefer: "return=representation" },
    body: JSON.stringify({
      machine_id: state.machineId || null,
      type: "start_training",
      payload,
      actor_role: state.profile?.role || "viewer",
    }),
  });
  status("Training job queued.");
  await refreshAll();
}

function renderList(selector, rows, renderer) {
  $(selector).innerHTML = rows.length ? rows.map(renderer).join("") : `<p class="muted">No records yet.</p>`;
}

async function refreshAll() {
  const [machines, jobs, runs] = await Promise.all([
    supabaseFetch("/rest/v1/machines?select=*&order=heartbeat_at.desc"),
    supabaseFetch("/rest/v1/jobs?select=*&order=created_at.desc&limit=20"),
    supabaseFetch("/rest/v1/runs?select=*&order=created_at.desc&limit=40"),
  ]);
  renderList("#machines", machines, (machine) => `
    <div class="item">
      <strong>${escapeHtml(machine.machine_id)}</strong>
      <span>${machine.online ? "online" : "offline"} · ${machine.accept_jobs ? "accepting jobs" : "paused"}</span>
      <small>${escapeHtml(machine.panel_version || "")} · ${escapeHtml(machine.heartbeat_at || "")}</small>
    </div>`);
  renderList("#jobs", jobs, (job) => `
    <div class="item">
      <strong>${escapeHtml(job.type)}</strong>
      <span>${escapeHtml(job.status)} · ${escapeHtml(job.machine_id || "any machine")}</span>
      <small>${escapeHtml(job.created_at || "")}</small>
    </div>`);
  renderList("#runs", runs, (run) => `
    <div class="item">
      <strong>${escapeHtml(run.display_name || run.id)}</strong>
      <span>${escapeHtml(run.status || "unknown")}</span>
      <small>${escapeHtml(run.latest_checkpoint || "no checkpoint")}</small>
      ${run.latest_video ? `<small>video: ${escapeHtml(run.latest_video)}</small>` : ""}
      ${run.onnx_path ? `<small>ONNX: ${escapeHtml(run.onnx_path)}</small>` : ""}
    </div>`);
}

$("#save-config").addEventListener("click", saveConfig);
$("#login").addEventListener("click", () => login().catch((error) => status(error.message)));
$("#queue-training").addEventListener("click", () => queueTraining().catch((error) => status(error.message)));
$("#refresh").addEventListener("click", () => refreshAll().catch((error) => status(error.message)));

hydrateForm();
if (state.accessToken) refreshAll().catch((error) => status(error.message));
