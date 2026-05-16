type NotificationEventType =
  | "training_converged"
  | "training_completed"
  | "training_failed"
  | "video_ready"
  | "test_notification";

type NotifyRequest = {
  event_type: NotificationEventType;
  event_key?: string;
  run_id?: string;
  machine_id?: string;
  requester_id?: string;
  created_at?: string;
  payload?: Record<string, unknown>;
};

type NotificationSettings = {
  id?: string;
  user_id: string;
  machine_id: string;
  email_enabled: boolean;
  discord_enabled: boolean;
  discord_webhook_url?: string;
  notify_training_converged: boolean;
  notify_training_completed: boolean;
  notify_training_failed: boolean;
  notify_video_ready: boolean;
};

type RunEventRow = {
  id?: string;
  event_key?: string;
  notification_status?: string;
  channel_results?: Record<string, unknown>;
  notified_at?: string;
};

const EVENT_LABELS: Record<NotificationEventType, string> = {
  training_converged: "Training converged",
  training_completed: "Training completed",
  training_failed: "Training failed",
  video_ready: "Video ready",
  test_notification: "Notification test",
};

const EVENT_SETTING_KEYS: Record<NotificationEventType, keyof NotificationSettings | ""> = {
  training_converged: "notify_training_converged",
  training_completed: "notify_training_completed",
  training_failed: "notify_training_failed",
  video_ready: "notify_video_ready",
  test_notification: "",
};

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

function env(name: string): string {
  return Deno.env.get(name) ?? "";
}

function jsonResponse(body: Record<string, unknown>, status = 200): Response {
  return Response.json(body, { status, headers: CORS_HEADERS });
}

function authToken(request: Request): string {
  const header = request.headers.get("Authorization") ?? "";
  return header.replace(/^Bearer\s+/i, "").trim();
}

function serviceHeaders() {
  const serviceKey = env("SUPABASE_SERVICE_ROLE_KEY");
  return {
    apikey: serviceKey,
    Authorization: `Bearer ${serviceKey}`,
    "Content-Type": "application/json",
  };
}

async function rest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const base = env("SUPABASE_URL").replace(/\/$/, "");
  const response = await fetch(`${base}${path}`, {
    ...options,
    headers: {
      ...serviceHeaders(),
      ...(options.headers ?? {}),
    },
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new Error(data?.message ?? data?.error ?? response.statusText);
  }
  return data as T;
}

async function userFromJwt(token: string): Promise<{ id: string; email?: string } | null> {
  if (!token || token === env("SUPABASE_SERVICE_ROLE_KEY")) return null;
  const base = env("SUPABASE_URL").replace(/\/$/, "");
  const response = await fetch(`${base}/auth/v1/user`, {
    headers: {
      apikey: env("SUPABASE_ANON_KEY") || env("SUPABASE_SERVICE_ROLE_KEY"),
      Authorization: `Bearer ${token}`,
    },
  });
  if (!response.ok) return null;
  const user = await response.json();
  return user?.id ? { id: user.id, email: user.email } : null;
}

function isWorkerToken(token: string): boolean {
  if (!token) return false;
  const allowed = [
    env("SUPABASE_SERVICE_ROLE_KEY"),
    env("REDRHEX_SUPABASE_MACHINE_TOKEN"),
    env("REDRHEX_NOTIFICATION_WORKER_TOKEN"),
  ].filter(Boolean);
  return allowed.includes(token);
}

async function requesterEmail(userId: string): Promise<string> {
  try {
    const user = await rest<{ email?: string }>(`/auth/v1/admin/users/${encodeURIComponent(userId)}`);
    if (user?.email) return user.email;
  } catch {
    // Fall back to profiles below.
  }
  const rows = await rest<Array<{ email?: string }>>(
    `/rest/v1/profiles?id=eq.${encodeURIComponent(userId)}&select=email&limit=1`,
  );
  return rows[0]?.email ?? "";
}

async function settingsFor(userId: string, machineId: string): Promise<NotificationSettings | null> {
  const rows = await rest<NotificationSettings[]>(
    `/rest/v1/notification_settings?user_id=eq.${encodeURIComponent(userId)}&machine_id=eq.${encodeURIComponent(machineId)}&select=*&limit=1`,
  );
  return rows[0] ?? null;
}

function eventEnabled(settings: NotificationSettings, eventType: NotificationEventType): boolean {
  if (eventType === "test_notification") return true;
  const key = EVENT_SETTING_KEYS[eventType];
  return key ? Boolean(settings[key]) : false;
}

function eventSubject(event: NotifyRequest): string {
  const payload = event.payload ?? {};
  const label = String(payload.display_name ?? event.run_id ?? "RedRHex");
  return `RedRHex ${EVENT_LABELS[event.event_type]}: ${label}`;
}

function eventLines(event: NotifyRequest): string[] {
  const payload = event.payload ?? {};
  const lines = [
    eventSubject(event),
    "",
    `Run: ${event.run_id ?? "-"}`,
    `Machine: ${event.machine_id ?? "-"}`,
    `Status: ${payload.status ?? "-"}`,
    `Task: ${payload.task ?? "-"}`,
    `Iterations: ${payload.max_iterations ?? "-"}`,
  ];
  if (event.event_type === "training_converged") {
    lines.push(`Converged at iteration: ${payload.iteration ?? "-"}`);
    lines.push(`Improvement: ${payload.improvement_pct ?? "-"}%`);
  }
  if (event.event_type === "video_ready") {
    lines.push(`Video: ${payload.storage_path ?? payload.latest_video ?? "-"}`);
  }
  if (payload.latest_checkpoint) lines.push(`Checkpoint: ${payload.latest_checkpoint}`);
  if (payload.remote_url) lines.push(`Remote: ${payload.remote_url}`);
  return lines;
}

function discordPayload(event: NotifyRequest) {
  const payload = event.payload ?? {};
  const fields = [
    { name: "Run", value: String(payload.display_name ?? event.run_id ?? "-"), inline: false },
    { name: "Machine", value: String(event.machine_id ?? "-"), inline: true },
    { name: "Status", value: String(payload.status ?? "-"), inline: true },
    { name: "Task", value: String(payload.task ?? "-"), inline: true },
  ];
  if (event.event_type === "training_converged") {
    fields.push({ name: "Iteration", value: String(payload.iteration ?? "-"), inline: true });
    fields.push({ name: "Improvement", value: `${payload.improvement_pct ?? "-"}%`, inline: true });
  }
  if (event.event_type === "video_ready") {
    fields.push({ name: "Video", value: String(payload.storage_path ?? payload.latest_video ?? "-"), inline: false });
  }
  if (payload.remote_url) fields.push({ name: "Remote", value: String(payload.remote_url), inline: false });
  return {
    content: eventSubject(event),
    embeds: [{ title: EVENT_LABELS[event.event_type], fields }],
  };
}

async function sendDiscord(webhook: string, event: NotifyRequest) {
  if (!webhook) return { ok: false, skipped: true, reason: "No Discord webhook configured" };
  const response = await fetch(webhook, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(discordPayload(event)),
  });
  return { ok: response.ok, status: response.status };
}

async function sendEmail(to: string, event: NotifyRequest) {
  const resendKey = env("REDRHEX_RESEND_API_KEY");
  const emailFrom = env("REDRHEX_NOTIFICATION_EMAIL_FROM");
  if (!resendKey || !emailFrom) return { ok: false, skipped: true, reason: "Email provider not configured" };
  if (!to) return { ok: false, skipped: true, reason: "Requester email not found" };
  const subject = eventSubject(event);
  const response = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${resendKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: emailFrom,
      to: [to],
      subject,
      text: eventLines(event).join("\n"),
    }),
  });
  return { ok: response.ok, status: response.status, to };
}

async function existingEvent(eventKey: string): Promise<RunEventRow | null> {
  if (!eventKey) return null;
  const rows = await rest<RunEventRow[]>(
    `/rest/v1/run_events?event_key=eq.${encodeURIComponent(eventKey)}&select=id,event_key,notification_status,channel_results,notified_at&limit=1`,
  );
  return rows[0] ?? null;
}

async function recordEvent(event: NotifyRequest, status: string, results: Record<string, unknown>) {
  const eventKey = event.event_key ?? `${event.event_type}:${event.run_id ?? "no-run"}:${Date.now()}`;
  const now = new Date().toISOString();
  const body = {
    event_key: eventKey,
    event_type: event.event_type,
    run_id: event.run_id || null,
    machine_id: event.machine_id || null,
    recipient_id: event.requester_id || null,
    payload: event.payload ?? {},
    notification_status: status,
    channel_results: results,
    notified_at: now,
    discord_sent_at: (results.discord as Record<string, unknown> | undefined)?.ok ? now : null,
    email_sent_at: (results.email as Record<string, unknown> | undefined)?.ok ? now : null,
  };
  await rest(
    `/rest/v1/run_events?on_conflict=event_key`,
    {
      method: "POST",
      headers: { Prefer: "resolution=merge-duplicates,return=representation" },
      body: JSON.stringify(body),
    },
  );
}

Deno.serve(async (request) => {
  if (request.method === "OPTIONS") {
    return new Response("ok", { headers: CORS_HEADERS });
  }
  let event: NotifyRequest;
  try {
    event = (await request.json()) as NotifyRequest;
  } catch {
    return jsonResponse({ ok: false, error: "Invalid JSON body" }, 400);
  }
  if (!event.event_type || !EVENT_LABELS[event.event_type]) {
    return jsonResponse({ ok: false, error: "Unsupported or missing event_type" }, 400);
  }
  if (!event.machine_id) {
    return jsonResponse({ ok: false, error: "Missing machine_id" }, 400);
  }

  const token = authToken(request);
  const user = await userFromJwt(token);
  const isTest = event.event_type === "test_notification";
  if (isTest) {
    if (!user?.id) return jsonResponse({ ok: false, error: "Sign in before sending a test notification" }, 401);
    event.requester_id = user.id;
    event.payload = { ...(event.payload ?? {}), display_name: "Notification test", email: user.email };
  } else if (!isWorkerToken(token)) {
    return jsonResponse({ ok: false, error: "Worker authorization required" }, 401);
  }

  if (!event.requester_id) {
    return jsonResponse({ ok: false, error: "Missing requester_id" }, 400);
  }
  event.event_key = event.event_key ?? (
    isTest
      ? `test_notification:${event.requester_id}:${Date.now()}`
      : `${event.event_type}:${event.run_id ?? "no-run"}:${event.requester_id}`
  );

  const existing = await existingEvent(event.event_key);
  if (existing?.notified_at) {
    return jsonResponse({ ok: true, deduped: true, results: existing.channel_results ?? {}, status: existing.notification_status });
  }

  const settings = await settingsFor(event.requester_id, event.machine_id);
  if (!settings) {
    const results = { email: { skipped: true, reason: "No notification settings" }, discord: { skipped: true, reason: "No notification settings" } };
    await recordEvent(event, "skipped", results);
    return jsonResponse({ ok: true, status: "skipped", results });
  }
  if (!eventEnabled(settings, event.event_type)) {
    const results = { email: { skipped: true, reason: "Event disabled" }, discord: { skipped: true, reason: "Event disabled" } };
    await recordEvent(event, "disabled", results);
    return jsonResponse({ ok: true, status: "disabled", results });
  }

  const results: Record<string, unknown> = {};
  if (settings.discord_enabled) {
    try {
      results.discord = await sendDiscord(settings.discord_webhook_url ?? "", event);
    } catch (error) {
      results.discord = { ok: false, error: String(error) };
    }
  } else {
    results.discord = { skipped: true, reason: "Discord disabled" };
  }

  if (settings.email_enabled) {
    try {
      const testEmail = isTest ? String((event.payload ?? {}).email ?? "") : "";
      results.email = await sendEmail(testEmail || await requesterEmail(event.requester_id), event);
    } catch (error) {
      results.email = { ok: false, error: String(error) };
    }
  } else {
    results.email = { skipped: true, reason: "Email disabled" };
  }

  const attempted = Object.values(results).some((result) => !(result as Record<string, unknown>).skipped);
  const allOk = Object.values(results).every((result) => {
    const item = result as Record<string, unknown>;
    return item.skipped || item.ok;
  });
  const status = attempted ? (allOk ? "sent" : "partial") : "no_channels";
  await recordEvent(event, status, results);
  return jsonResponse({ ok: allOk, status, results });
});
