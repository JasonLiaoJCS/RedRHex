const POLL_MS = 1500;

const $ = (selector) => document.querySelector(selector);
const params = new URLSearchParams(location.search);
const target = {
  type: params.get("type") === "process" ? "process" : "run",
  id: params.get("id") || "",
};
const THEME_KEY = "redrhex-training-panel-theme";

function preferredTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("#terminal-theme").textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
}

function endpoint() {
  if (target.type === "process") return `/api/processes/${encodeURIComponent(target.id)}/debug`;
  return `/api/runs/${encodeURIComponent(target.id)}/debug`;
}

async function api(path) {
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

async function post(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function isLive(debug) {
  if (debug.kind) return debug.returncode === null;
  return debug.status === "running" || debug.status === "stopping" || debug.status === "video recording";
}

async function copyText(text) {
  if (!text.trim()) return;
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

async function copyDebug() {
  const text = [
    $("#terminal-target").textContent,
    "",
    "Command",
    $("#terminal-command").textContent,
    "",
    "Output",
    $("#terminal-output").textContent,
  ].join("\n");
  await copyText(text);
  $("#terminal-target").textContent = "Console output copied.";
}

async function copyAttach() {
  if (!window.lastDebug || !window.lastDebug.attach_command) {
    $("#terminal-target").textContent = "No tmux attach command is available for this process.";
    return;
  }
  await copyText(window.lastDebug.attach_command);
  $("#terminal-target").textContent = `Attach command copied: ${window.lastDebug.attach_command}`;
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
    return "Diagnosis: the video process ended but no MP4 was found. Check the play/video output and encoder dependencies.";
  }
  if (/policy\.onnx was not produced|ONNX export finished/i.test(output)) {
    return "Diagnosis: ONNX export finished without exported/policy.onnx. Check the checkpoint load and exporter output.";
  }
  return "";
}

function render(debug) {
  window.lastDebug = debug;
  const live = isLive(debug);
  const startingHint =
    target.type === "run" && debug.status === "running"
      ? "A panel process for this run is active. Use Stop here, or attach with the tmux command when available."
      : "";
  const capturedOutput = debug.log_tail ?? debug.process_log_tail ?? "";
  const output = capturedOutput || startingHint || debug.debug_hint || "";
  const pieces = [];
  if (debug.kind) pieces.push(`${debug.kind} process`);
  if (debug.id) pieces.push(`run ${debug.id}`);
  if (debug.run_id && !debug.id) pieces.push(`process ${debug.run_id}`);
  if (debug.pid) pieces.push(`pid ${debug.pid}`);
  if (debug.status) pieces.push(`status ${debug.status}`);
  if (debug.returncode !== undefined && debug.returncode !== null) pieces.push(`return ${debug.returncode}`);
  if (debug.attach_command) pieces.push(`attach ${debug.attach_command}`);
  if (debug.process_log || debug.log_file) pieces.push(debug.process_log || debug.log_file);
  const diagnosis = outputDiagnosis(output);
  if (diagnosis) pieces.push(diagnosis);
  $("#terminal-target").textContent = pieces.join(" | ") || `${target.type}: ${target.id}`;
  $("#terminal-live").textContent = live ? "Live" : "Snapshot";
  $("#terminal-live").className = live ? "status-badge live-pill" : "status-badge muted-pill";
  $("#terminal-command").textContent = debug.command || "";
  $("#terminal-output").textContent = output || "No terminal output captured yet.";
  $("#terminal-output").scrollTop = $("#terminal-output").scrollHeight;
  $("#terminal-stop").disabled = target.type === "process" ? !live : false;
}

async function refresh() {
  if (!target.id) {
    $("#terminal-target").textContent = "No run or process selected.";
    return;
  }
  try {
    render(await api(endpoint()));
  } catch (error) {
    $("#terminal-live").textContent = "Error";
    $("#terminal-live").className = "status-badge error-pill";
    $("#terminal-target").textContent = error.message;
  }
}

async function stopProcess() {
  if (!target.id) return;
  try {
    const data =
      target.type === "process"
        ? await post("/api/training/stop", { run_id: target.id })
        : await post(`/api/runs/${encodeURIComponent(target.id)}/stop`, {});
    const stopped = data.stopped_ids || (data.stopped ? [target.id] : []);
    $("#terminal-target").textContent = stopped.length
      ? `Stopping ${stopped.join(", ")}...`
      : "No running processes were found for this run.";
    refresh();
  } catch (error) {
    $("#terminal-live").textContent = "Error";
    $("#terminal-live").className = "status-badge error-pill";
    $("#terminal-target").textContent = error.message;
  }
}

$("#terminal-refresh").addEventListener("click", refresh);
$("#terminal-theme").addEventListener("click", toggleTheme);
$("#terminal-copy").addEventListener("click", () => copyDebug().catch((error) => {
  $("#terminal-live").textContent = "Error";
  $("#terminal-live").className = "status-badge error-pill";
  $("#terminal-target").textContent = error.message;
}));
$("#terminal-attach").addEventListener("click", () => copyAttach().catch((error) => {
  $("#terminal-live").textContent = "Error";
  $("#terminal-live").className = "status-badge error-pill";
  $("#terminal-target").textContent = error.message;
}));
$("#terminal-stop").addEventListener("click", stopProcess);
applyTheme(preferredTheme());
refresh();
setInterval(refresh, POLL_MS);
