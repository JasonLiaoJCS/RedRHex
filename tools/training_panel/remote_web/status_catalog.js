const STATUS = {
  run: {
    queued: ["queued", "info", "Waiting for mother to start this training request."],
    running: ["running", "info", "Training is currently running on mother."],
    stopping: ["stopping", "info", "A stop request is in progress."],
    completed: ["completed", "good", "Training finished successfully."],
    failed: ["failed", "bad", "Training exited with an error."],
    interrupted: ["interrupted", "bad", "Training stopped before completion."],
    cancelled: ["cancelled", "bad", "The queued request was cancelled."],
    deleting: ["deleting", "muted", "Mother is deleting this history item."],
    deleted: ["deleted", "muted", "This history item was removed."],
    unknown: ["unknown", "muted", "Mother has not reported a final status yet."],
  },
  job: {
    queued: ["queued", "info", "The request is waiting for the worker."],
    claimed: ["claimed", "info", "The worker has claimed the request."],
    running: ["running", "info", "The worker is executing the request."],
    launched: ["launched", "info", "Training has launched; waiting for the run record to sync."],
    completed: ["completed", "good", "The request completed successfully."],
    failed: ["failed", "bad", "The request failed."],
    cancelled: ["cancelled", "bad", "The request was cancelled."],
    unknown: ["unknown", "muted", "The request status is unknown."],
  },
  machine: {
    ready: ["ready", "good", "Mother is online and accepting jobs."],
    busy: ["busy", "info", "Mother is online but an Isaac/GPU action is running."],
    paused: ["paused", "bad", "Mother is online but remote launch is paused."],
    offline: ["offline", "bad", "Mother heartbeat is stale."],
    missing: ["missing", "bad", "No matching mother machine was found."],
    unknown: ["unknown", "muted", "Machine state is unknown."],
  },
  artifact: {
    ready: ["ready", "good", "The artifact is available."],
    uploading: ["uploading", "info", "The artifact exists locally and is uploading to team storage."],
    recordable: ["recordable", "info", "A checkpoint is ready for video recording."],
    missing: ["missing", "muted", "No artifact is available yet."],
    failed: ["failed", "bad", "Artifact generation failed."],
    unknown: ["unknown", "muted", "Artifact state is unknown."],
  },
  video: {
    recording: ["recording", "info", "Mother is recording a video of this checkpoint."],
    completed: ["recorded", "good", "Video recording finished successfully."],
    failed: ["record failed", "bad", "Video recording failed. Check the Process Console on mother."],
    missing_checkpoint: ["no checkpoint", "muted", "Cannot record — no checkpoint has been saved yet."],
    unknown: ["unknown", "muted", "Video recording state is unknown."],
  },
};

export function normalizeStatus(status) {
  return String(status || "unknown").trim().toLowerCase() || "unknown";
}

export function jobDisplayStatus(job = {}) {
  const status = normalizeStatus(job.status);
  if (status === "completed" && String(job.type || "") === "start_training") return "launched";
  return status;
}

export function statusDescriptor(kind, status, context = {}) {
  const normalizedKind = STATUS[kind] ? kind : "run";
  const normalizedStatus = normalizedKind === "job" && context.job
    ? jobDisplayStatus(context.job)
    : normalizeStatus(status);
  const entry = STATUS[normalizedKind][normalizedStatus] || STATUS[normalizedKind].unknown;
  return {
    kind: normalizedKind,
    status: normalizedStatus,
    label: entry[0],
    tone: entry[1],
    description: entry[2],
  };
}

export function statusTone(kindOrStatus, status = "", context = {}) {
  if (status === "") return statusDescriptor("run", kindOrStatus).tone;
  return statusDescriptor(kindOrStatus, status, context).tone;
}

export function statusLabel(kindOrStatus, status = "", context = {}) {
  if (status === "") return statusDescriptor("run", kindOrStatus).label;
  return statusDescriptor(kindOrStatus, status, context).label;
}

export function statusDescription(kindOrStatus, status = "", context = {}) {
  if (status === "") return statusDescriptor("run", kindOrStatus).description;
  return statusDescriptor(kindOrStatus, status, context).description;
}

export const STATUS_CATALOG = STATUS;

/**
 * Returns a short convergence label for display in run cards and detail panels.
 * Returns an empty string when the run has not converged.
 */
export function convergenceLabel(run) {
  if (!run?.convergence_detected) return "";
  const iter = run.convergence_iteration;
  const pct  = run.convergence_improvement_pct;
  if (iter != null && pct != null) return `Converged iter ${iter} (Δ${pct}%)`;
  if (iter != null) return `Converged iter ${iter}`;
  return "Converged";
}

/**
 * Returns the best video status label for a run, preferring the directly-synced
 * video_status field when present, then falling back to artifact-derived state.
 */
export function resolvedVideoStatus(run, artifactVideoState = "missing") {
  const direct = normalizeStatus(run?.video_status);
  if (direct && direct !== "unknown" && STATUS.video[direct]) return direct;
  return artifactVideoState;
}
