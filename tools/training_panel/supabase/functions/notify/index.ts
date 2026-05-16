type RunEvent = {
  id: string;
  event_type: string;
  run_id: string;
  payload: Record<string, unknown>;
};

function discordPayload(event: RunEvent) {
  const title = event.event_type === "training_completed" ? "Training completed" : "Training failed";
  return {
    content: `${title}: ${event.run_id}`,
    embeds: [
      {
        title,
        fields: [
          { name: "Run", value: event.run_id, inline: true },
          { name: "Task", value: String(event.payload.task ?? "-"), inline: true },
          { name: "Iterations", value: String(event.payload.max_iterations ?? "-"), inline: true },
          { name: "Checkpoint", value: String(event.payload.latest_checkpoint ?? "-"), inline: false },
          { name: "Remote", value: String(event.payload.remote_url ?? "-"), inline: false },
        ],
      },
    ],
  };
}

Deno.serve(async (request) => {
  const event = (await request.json()) as RunEvent;
  const discordWebhook = Deno.env.get("REDRHEX_DISCORD_WEBHOOK_URL");
  const resendKey = Deno.env.get("REDRHEX_RESEND_API_KEY");
  const emailTo = Deno.env.get("REDRHEX_NOTIFICATION_EMAIL_TO");
  const results: Record<string, unknown> = {};

  if (discordWebhook) {
    const response = await fetch(discordWebhook, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(discordPayload(event)),
    });
    results.discord = { ok: response.ok, status: response.status };
  }

  if (resendKey && emailTo) {
    const subject = `RedRHex ${event.event_type}: ${event.run_id}`;
    const response = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${resendKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from: "RedRHex Training <training@updates.redrhex.local>",
        to: [emailTo],
        subject,
        text: `${subject}\n\n${JSON.stringify(event.payload, null, 2)}`,
      }),
    });
    results.email = { ok: response.ok, status: response.status };
  }

  return Response.json({ ok: true, results });
});
