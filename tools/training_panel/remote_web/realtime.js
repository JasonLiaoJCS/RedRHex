import { SUPABASE_ANON_KEY, SUPABASE_URL } from "./config.js?v=3.4-first-release";
import { sessionStore } from "./api.js?v=3.4-first-release";

const REALTIME_IMPORT_URL = "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm";
const TABLES = ["runs", "jobs", "artifacts", "machines", "run_deletions"];

function rowMachineId(row = {}) {
  return String(row.machine_id || row.claimed_by || row.machine_id_eq || "");
}

function shouldRefreshForPayload(machineId, payload = {}) {
  const table = payload.table || "";
  const row = payload.new || payload.old || {};
  if (!machineId) return true;
  if (table === "jobs") {
    const jobMachine = rowMachineId(row);
    return !jobMachine || jobMachine === machineId;
  }
  const eventMachine = rowMachineId(row);
  return !eventMachine || eventMachine === machineId;
}

export function createRemoteRealtime({ machineId = "", onChange = () => {}, onStatus = () => {} } = {}) {
  let client = null;
  let channel = null;
  let stopped = false;
  const diagnostics = {
    enabled: false,
    status: "off",
    error: "",
    lastEventAt: "",
    lastTable: "",
  };

  function update(fields) {
    Object.assign(diagnostics, fields);
    onStatus({ ...diagnostics });
  }

  async function start() {
    stopped = false;
    update({ enabled: false, status: "connecting", error: "" });
    try {
      const module = await import(REALTIME_IMPORT_URL);
      if (stopped) return { ...diagnostics };
      client = module.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        auth: { persistSession: false, autoRefreshToken: false },
        global: {
          headers: {
            Authorization: `Bearer ${sessionStore.accessToken || SUPABASE_ANON_KEY}`,
          },
        },
      });
      if (sessionStore.accessToken && client.realtime?.setAuth) {
        client.realtime.setAuth(sessionStore.accessToken);
      }
      channel = client.channel(`redrhex-child-history-${machineId || "all"}`);
      for (const table of TABLES) {
        channel.on("postgres_changes", { event: "*", schema: "public", table }, (payload) => {
          if (!shouldRefreshForPayload(machineId, payload)) return;
          const now = new Date().toISOString();
          update({ enabled: true, status: "event", error: "", lastEventAt: now, lastTable: table });
          onChange({ table, payload, at: now });
        });
      }
      channel.subscribe((status, error) => {
        if (stopped) return;
        const normalized = String(status || "").toLowerCase();
        const active = normalized === "subscribed";
        update({
          enabled: active,
          status: active ? "subscribed" : normalized || "connecting",
          error: error?.message || "",
        });
      });
    } catch (error) {
      update({ enabled: false, status: "fallback", error: error?.message || String(error) });
    }
    return { ...diagnostics };
  }

  async function stop() {
    stopped = true;
    try {
      if (channel && client) await client.removeChannel(channel);
      else if (channel?.unsubscribe) await channel.unsubscribe();
    } catch {
      // Polling fallback remains active if realtime shutdown is noisy.
    }
    channel = null;
    client = null;
    update({ enabled: false, status: "off" });
  }

  return {
    start,
    stop,
    diagnostics: () => ({ ...diagnostics }),
  };
}
