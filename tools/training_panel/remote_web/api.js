import { SUPABASE_ANON_KEY, SUPABASE_URL, VIDEO_BUCKET } from "./config.js?v=3.4-first-release";
import { BUILT_IN_REWARD_PRESETS, BUILT_IN_TERRAIN_PRESETS } from "./core.js?v=3.4-first-release";

const TOKEN_KEY = "redrhex_child_access_token";
const REFRESH_KEY = "redrhex_child_refresh_token";
const DEFAULT_FETCH_TIMEOUT_MS = 15000;

export const sessionStore = {
  get accessToken() {
    return localStorage.getItem(TOKEN_KEY) || "";
  },
  get refreshToken() {
    return localStorage.getItem(REFRESH_KEY) || "";
  },
  save(session) {
    if (session?.access_token) localStorage.setItem(TOKEN_KEY, session.access_token);
    if (session?.refresh_token) localStorage.setItem(REFRESH_KEY, session.refresh_token);
  },
  clear() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REFRESH_KEY);
  },
};

function authHeaders(extra = {}) {
  const token = sessionStore.accessToken || SUPABASE_ANON_KEY;
  return {
    apikey: SUPABASE_ANON_KEY,
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    ...extra,
  };
}

async function parseResponse(response) {
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { message: text };
    }
  }
  if (!response.ok) {
    throw new Error(data?.message || data?.error || data?.error_description || data?.hint || response.statusText);
  }
  return data;
}

export async function supabaseFetch(path, options = {}) {
  const { timeoutMs = DEFAULT_FETCH_TIMEOUT_MS, ...requestOptions } = options;
  const controller = typeof AbortController !== "undefined" && !requestOptions.signal
    ? new AbortController()
    : null;
  const timeoutId = controller && timeoutMs > 0
    ? globalThis.setTimeout(() => controller.abort(), timeoutMs)
    : null;
  try {
    const response = await fetch(`${SUPABASE_URL}${path}`, {
      ...requestOptions,
      signal: requestOptions.signal || controller?.signal,
      headers: authHeaders(requestOptions.headers || {}),
    });
    return await parseResponse(response);
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("Supabase request timed out. Check the connection and try refreshing.");
    }
    throw error;
  } finally {
    if (timeoutId) globalThis.clearTimeout(timeoutId);
  }
}

export async function signIn(email, password) {
  const session = await supabaseFetch("/auth/v1/token?grant_type=password", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  sessionStore.save(session);
  return session;
}

export async function signOut() {
  if (sessionStore.accessToken) {
    try {
      await supabaseFetch("/auth/v1/logout", { method: "POST" });
    } catch {
      // Local sign-out should still complete if the token is already invalid.
    }
  }
  sessionStore.clear();
}

export async function currentUser() {
  if (!sessionStore.accessToken) return null;
  try {
    return await supabaseFetch("/auth/v1/user");
  } catch {
    sessionStore.clear();
    return null;
  }
}

export async function select(table, query = "") {
  const suffix = query ? `?${query}` : "";
  return supabaseFetch(`/rest/v1/${table}${suffix}`);
}

async function optionalSelect(table, query = "") {
  try {
    return { rows: await select(table, query), ok: true, error: "" };
  } catch (error) {
    return { rows: [], ok: false, error: error.message || String(error) };
  }
}

export async function insert(table, payload) {
  return supabaseFetch(`/rest/v1/${table}`, {
    method: "POST",
    headers: { Prefer: "return=representation" },
    body: JSON.stringify(payload),
  });
}

export async function update(table, filterQuery, payload) {
  return supabaseFetch(`/rest/v1/${table}?${filterQuery}`, {
    method: "PATCH",
    headers: { Prefer: "return=representation" },
    body: JSON.stringify(payload),
  });
}

export async function remove(table, filterQuery) {
  return supabaseFetch(`/rest/v1/${table}?${filterQuery}`, {
    method: "DELETE",
    headers: { Prefer: "return=representation" },
  });
}

export async function upsert(table, payload, onConflict = "id") {
  return supabaseFetch(`/rest/v1/${table}?on_conflict=${encodeURIComponent(onConflict)}`, {
    method: "POST",
    headers: { Prefer: "resolution=merge-duplicates,return=representation" },
    body: JSON.stringify(payload),
  });
}

export async function invokeFunction(name, payload) {
  return supabaseFetch(`/functions/v1/${encodeURIComponent(name)}`, {
    method: "POST",
    body: JSON.stringify(payload || {}),
  });
}

export async function createSignedVideoUrl(storagePath, expiresIn = 3600) {
  const encoded = String(storagePath || "")
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
  const data = await supabaseFetch(`/storage/v1/object/sign/${VIDEO_BUCKET}/${encoded}`, {
    method: "POST",
    body: JSON.stringify({ expiresIn }),
  });
  const signed = data?.signedURL || data?.signedUrl || "";
  if (!signed) throw new Error("Supabase did not return a signed video URL.");
  if (signed.startsWith("http")) return signed;
  if (signed.startsWith("/storage/v1")) return `${SUPABASE_URL}${signed}`;
  if (signed.startsWith("/")) return `${SUPABASE_URL}/storage/v1${signed}`;
  return `${SUPABASE_URL}/storage/v1/${signed}`;
}

export async function loadRemoteSnapshot(machineId, userId = "") {
  const encodedMachine = encodeURIComponent(machineId);
  const encodedUser = encodeURIComponent(userId || "");
  const notificationQuery = userId
    ? `user_id=eq.${encodedUser}&machine_id=eq.${encodedMachine}&select=*&limit=1`
    : "select=*&limit=0";
  const [machines, jobs, runs, deletionsResult, artifactsResult, presetsResult, terrainPresetsResult, profilesResult, notificationResult] = await Promise.all([
    select("machines", `select=*&order=heartbeat_at.desc`),
    select("jobs", `select=*&order=created_at.desc&limit=60`),
    select("runs", `select=*&order=created_at.desc&limit=120`),
    optionalSelect("run_deletions", `select=*&order=deleted_at.desc&limit=500`),
    optionalSelect("artifacts", `select=*&order=created_at.desc&limit=500`),
    optionalSelect("reward_presets", `select=*&order=built_in.desc,updated_at.desc,name.asc`),
    optionalSelect("terrain_presets", `select=*&order=built_in.desc,updated_at.desc,name.asc`),
    optionalSelect("profiles", `select=id,email,display_name,role`),
    optionalSelect("notification_settings", notificationQuery),
  ]);
  const artifacts = artifactsResult.rows;
  const presets = presetsResult.ok ? presetsResult.rows : BUILT_IN_REWARD_PRESETS;
  const terrainPresets = terrainPresetsResult.ok ? terrainPresetsResult.rows : BUILT_IN_TERRAIN_PRESETS;
  return {
    machines,
    machine: machines.find((machine) => machine.machine_id === machineId) || machines[0] || null,
    targetMachine: machines.find((machine) => machine.machine_id === machineId) || null,
    jobs: jobs.filter((job) => !job.machine_id || job.machine_id === machineId || job.claimed_by === machineId),
    runs: runs.filter((run) => !run.machine_id || run.machine_id === machineId || machineId === ""),
    runDeletions: deletionsResult.rows.filter((item) => !item.machine_id || item.machine_id === machineId || machineId === ""),
    artifacts: artifacts.filter((artifact) => !artifact.machine_id || artifact.machine_id === machineId || machineId === ""),
    profiles: profilesResult.rows,
    notificationSettings: notificationResult.rows[0] || null,
    presets,
    terrainPresets,
    encodedMachine,
    schema: {
      artifacts: artifactsResult.ok,
      runDeletions: deletionsResult.ok,
      rewardPresets: presetsResult.ok,
      terrainPresets: terrainPresetsResult.ok,
      warnings: [
        deletionsResult.ok ? "" : `Run deletion tombstones unavailable: ${deletionsResult.error}`,
        artifactsResult.ok ? "" : `Artifacts table unavailable: ${artifactsResult.error}`,
        presetsResult.ok ? "" : `Reward presets table unavailable: ${presetsResult.error}`,
        terrainPresetsResult.ok ? "" : `Terrain presets table unavailable: ${terrainPresetsResult.error}`,
        profilesResult.ok ? "" : `Profiles table unavailable: ${profilesResult.error}`,
        notificationResult.ok ? "" : `Notification settings unavailable: ${notificationResult.error}`,
      ].filter(Boolean),
    },
  };
}
