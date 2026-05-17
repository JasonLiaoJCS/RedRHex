import { jobDisplayStatus } from "./status_catalog.js?v=3.4-first-release";

export const PENDING_SYNTHETIC_JOB_MAX_AGE_MS = 30 * 60 * 1000;

function clean(value) {
  return String(value || "").trim();
}

function basename(path) {
  const value = clean(path);
  if (!value) return "";
  return value.split(/[\\/]/).filter(Boolean).pop() || "";
}

export function normalizeRunDeletions(rows = []) {
  const ids = new Set();
  const logDirs = new Set();
  const logDirNames = new Set();
  const records = [];
  for (const row of rows || []) {
    const id = clean(row?.id);
    const logDir = clean(row?.log_dir);
    const logDirName = clean(row?.log_dir_name) || basename(logDir);
    if (id) ids.add(id);
    if (logDir) logDirs.add(logDir);
    if (logDirName) {
      ids.add(logDirName);
      logDirNames.add(logDirName);
    }
    records.push({ ...row, id, log_dir: logDir, log_dir_name: logDirName });
  }
  return { ids, logDirs, logDirNames, records };
}

export function jobRunId(job = {}) {
  const payload = job.payload && typeof job.payload === "object" ? job.payload : {};
  const result = job.result && typeof job.result === "object" ? job.result : {};
  const resultPayload = result.payload && typeof result.payload === "object" ? result.payload : {};
  return clean(
    payload.run_id
    || result.local_run_id
    || result.process_id
    || resultPayload.id
    || resultPayload.run_id
    || resultPayload.source_run_id
    || "",
  );
}

export function runDeletionKeys(run = {}) {
  const keys = {
    ids: [clean(run.id), clean(run.linked_run_id)],
    logDirs: [clean(run.log_dir)],
    logDirNames: [basename(run.log_dir)],
  };
  if (run.synthetic_job) {
    keys.ids.push(clean(run.linked_run_id));
  }
  return keys;
}

export function isDeletedRunLike(run = {}, deletions = normalizeRunDeletions()) {
  const keys = runDeletionKeys(run);
  return keys.ids.some((id) => id && deletions.ids.has(id))
    || keys.logDirs.some((logDir) => logDir && deletions.logDirs.has(logDir))
    || keys.logDirNames.some((name) => name && deletions.logDirNames.has(name));
}

export function filterDeletedRuns(runs = [], runDeletions = []) {
  const deletions = normalizeRunDeletions(runDeletions);
  return (runs || []).filter((run) => !isDeletedRunLike(run, deletions));
}

export function historyTimeValue(run = {}) {
  const parsed = Date.parse(run.created_at || run.started_at || run.updated_at || "");
  return Number.isFinite(parsed) ? parsed : 0;
}

function nameValue(run = {}) {
  return clean(run.display_name || run.id || run.job_id).toLowerCase();
}

export function normalizeHistorySort(sortBy = "newest") {
  const value = clean(sortBy).toLowerCase();
  if (value === "name" || value === "oldest") return value;
  return "newest";
}

export function compareHistoryRuns(a = {}, b = {}, sortBy = "newest") {
  const mode = normalizeHistorySort(sortBy);
  if (mode === "name") {
    return nameValue(a).localeCompare(nameValue(b)) || historyTimeValue(b) - historyTimeValue(a);
  }
  if (mode === "oldest") {
    return historyTimeValue(a) - historyTimeValue(b) || nameValue(a).localeCompare(nameValue(b));
  }
  return historyTimeValue(b) - historyTimeValue(a) || nameValue(a).localeCompare(nameValue(b));
}

export function isFreshPendingJob(job = {}, nowMs = Date.now(), maxAgeMs = PENDING_SYNTHETIC_JOB_MAX_AGE_MS) {
  const status = clean(job.status).toLowerCase();
  if (!["queued", "claimed", "running"].includes(status)) return false;
  const timestamp = Date.parse(job.updated_at || job.created_at || "");
  return Number.isFinite(timestamp) && timestamp >= nowMs - maxAgeMs;
}

function jobTimeValue(job = {}) {
  const parsed = Date.parse(job.created_at || job.updated_at || "");
  return Number.isFinite(parsed) ? parsed : 0;
}

function clientRequestIdFromPayload(payload = {}) {
  return clean(payload.client_request_id || payload.clientRequestId || "");
}

export function jobClientRequestId(job = {}) {
  const payload = job.payload && typeof job.payload === "object" ? job.payload : {};
  const result = job.result && typeof job.result === "object" ? job.result : {};
  const resultPayload = result.payload && typeof result.payload === "object" ? result.payload : {};
  return clientRequestIdFromPayload(payload)
    || clientRequestIdFromPayload(result)
    || clientRequestIdFromPayload(resultPayload);
}

export function runClientRequestId(run = {}) {
  const params = run.params && typeof run.params === "object" ? run.params : {};
  return clientRequestIdFromPayload(run) || clientRequestIdFromPayload(params);
}

export function realRunConfirmsJob(job = {}, realRuns = []) {
  const payload = job.payload && typeof job.payload === "object" ? job.payload : {};
  const clientRequestId = jobClientRequestId(job);
  if (clientRequestId && (realRuns || []).some((run) => runClientRequestId(run) === clientRequestId)) {
    return true;
  }

  const linkedRunId = jobRunId(job);
  if (linkedRunId && (realRuns || []).some((run) => clean(run.id) === linkedRunId)) {
    return true;
  }

  const jobCreated = jobTimeValue(job);
  const jobName = clean(payload.display_name).toLowerCase();
  return (realRuns || []).some((run) => {
    const runCreated = historyTimeValue(run);
    if (jobCreated && runCreated && runCreated < jobCreated - 60_000) return false;

    const runName = clean(run.display_name).toLowerCase();
    if (jobName && runName && jobName === runName) return true;
    return false;
  });
}

function mergedJobs(remoteJobs = [], localPendingJobs = []) {
  const remoteKeys = new Set();
  for (const job of remoteJobs || []) {
    const id = clean(job?.id);
    const clientRequestId = jobClientRequestId(job);
    if (id) remoteKeys.add(`id:${id}`);
    if (clientRequestId) remoteKeys.add(`client:${clientRequestId}`);
  }
  const locals = (localPendingJobs || []).filter((job) => {
    const id = clean(job?.id);
    const clientRequestId = jobClientRequestId(job);
    return !(id && remoteKeys.has(`id:${id}`)) && !(clientRequestId && remoteKeys.has(`client:${clientRequestId}`));
  });
  return [...locals, ...(remoteJobs || [])];
}

export function syntheticRunsFromJobs(jobs = [], realRuns = [], runDeletions = [], options = {}) {
  const deletions = normalizeRunDeletions(runDeletions);
  const nowMs = options.nowMs ?? Date.now();
  const maxAgeMs = options.maxAgeMs ?? PENDING_SYNTHETIC_JOB_MAX_AGE_MS;
  return (jobs || [])
    .filter((job) => job.type === "start_training")
    .filter((job) => !realRunConfirmsJob(job, realRuns))
    .filter((job) => isFreshPendingJob(job, nowMs, maxAgeMs))
    .map((job) => {
      const payload = job.payload && typeof job.payload === "object" ? job.payload : {};
      const linkedRunId = jobRunId(job);
      const status = jobDisplayStatus(job);
      return {
        id: `job:${job.id}`,
        synthetic_job: true,
        job_id: job.id,
        linked_run_id: linkedRunId,
        source: "remote_job",
        status,
        pending_confirmation: true,
        display_name: payload.display_name || "Pending training",
        created_at: job.created_at,
        updated_at: job.updated_at || job.created_at,
        folder: payload.folder || "",
        params: payload,
        created_by: job.actor_id || payload.requester_id || null,
      };
    })
    .filter((run) => !isDeletedRunLike(run, deletions));
}

export function historyRunsForSnapshot(snapshot = {}, options = {}) {
  const runDeletions = snapshot.runDeletions || [];
  const realRuns = filterDeletedRuns(snapshot.runs || [], runDeletions);
  const jobs = mergedJobs(snapshot.jobs || [], options.localPendingTrainingJobs || snapshot.localPendingTrainingJobs || []);
  return [...syntheticRunsFromJobs(jobs, realRuns, runDeletions, options), ...realRuns]
    .sort((a, b) => compareHistoryRuns(a, b, options.sortBy || "newest"));
}
