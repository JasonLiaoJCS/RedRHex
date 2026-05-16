class LocalPanelApi {
  async request(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    return data;
  }
}

class SupabaseRemoteApi {
  constructor({ url, anonKey, accessToken }) {
    this.url = String(url || "").replace(/\/$/, "");
    this.anonKey = anonKey;
    this.accessToken = accessToken;
  }

  headers() {
    return {
      apikey: this.anonKey,
      Authorization: `Bearer ${this.accessToken}`,
      "Content-Type": "application/json",
    };
  }

  async rest(path, options = {}) {
    const response = await fetch(`${this.url}/rest/v1/${path.replace(/^\//, "")}`, {
      headers: this.headers(),
      ...options,
    });
    const text = await response.text();
    const data = text ? JSON.parse(text) : null;
    if (!response.ok) throw new Error(data?.message || response.statusText);
    return data;
  }

  async listRuns() {
    return this.rest("runs?select=*&order=created_at.desc");
  }

  async createJob(type, payload, machineId = null) {
    return this.rest("jobs", {
      method: "POST",
      body: JSON.stringify({ type, payload, machine_id: machineId }),
      headers: { ...this.headers(), Prefer: "return=representation" },
    });
  }
}

window.RedRHexApiAdapters = { LocalPanelApi, SupabaseRemoteApi };
