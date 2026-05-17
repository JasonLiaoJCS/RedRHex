(function (global) {
  const STATUS = {
    run: {
      queued: ["queued", "status-queued", "Waiting for mother to start this training request."],
      running: ["running", "status-running", "Training is currently running on mother."],
      stopping: ["stopping", "status-running", "A stop request is in progress."],
      completed: ["completed", "status-completed", "Training finished successfully."],
      failed: ["failed", "status-failed", "Training exited with an error."],
      interrupted: ["interrupted", "status-interrupted", "Training stopped before completion."],
      cancelled: ["cancelled", "status-interrupted", "The queued request was cancelled."],
      deleting: ["deleting", "status-unknown", "Mother is deleting this history item."],
      deleted: ["deleted", "status-unknown", "This history item was removed."],
      unknown: ["unknown", "status-unknown", "Mother has not reported a final status yet."],
    },
    machine: {
      ready: ["ready", "status-completed", "Mother is online and accepting jobs."],
      busy: ["busy", "status-running", "Mother is online but an Isaac/GPU action is running."],
      paused: ["paused", "status-interrupted", "Mother is online but remote launch is paused."],
      offline: ["offline", "status-failed", "Mother heartbeat is stale."],
      missing: ["missing", "status-failed", "No matching mother machine was found."],
      unknown: ["unknown", "status-unknown", "Machine state is unknown."],
    },
    job: {
      queued: ["queued", "status-queued", "The request is waiting for the worker."],
      claimed: ["claimed", "status-running", "The worker has claimed the request."],
      running: ["running", "status-running", "The worker is executing the request."],
      launched: ["launched", "status-running", "Training has launched; waiting for the run record to sync."],
      completed: ["completed", "status-completed", "The request completed successfully."],
      failed: ["failed", "status-failed", "The request failed."],
      cancelled: ["cancelled", "status-interrupted", "The request was cancelled."],
      unknown: ["unknown", "status-unknown", "The request status is unknown."],
    },
  };

  function normalizeStatus(status) {
    return String(status || "unknown").trim().toLowerCase() || "unknown";
  }

  function jobDisplayStatus(job) {
    const status = normalizeStatus(job && job.status);
    if (status === "completed" && String((job && job.type) || "") === "start_training") return "launched";
    return status;
  }

  function descriptor(kind, status, context) {
    const normalizedKind = STATUS[kind] ? kind : "run";
    const normalizedStatus = normalizedKind === "job" && context && context.job
      ? jobDisplayStatus(context.job)
      : normalizeStatus(status);
    const entry = STATUS[normalizedKind][normalizedStatus] || STATUS[normalizedKind].unknown;
    return {
      kind: normalizedKind,
      status: normalizedStatus,
      label: entry[0],
      className: entry[1],
      description: entry[2],
    };
  }

  global.RedRhexStatus = {
    descriptor,
    jobDisplayStatus,
    label: (kind, status, context) => descriptor(kind, status, context).label,
    className: (kind, status, context) => descriptor(kind, status, context).className,
    description: (kind, status, context) => descriptor(kind, status, context).description,
  };
})(window);
