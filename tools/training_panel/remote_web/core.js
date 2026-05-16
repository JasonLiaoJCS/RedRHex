import { HEARTBEAT_STALE_MS } from "./config.js";

export const BUILT_IN_REWARD_PRESETS = [
  {
    id: "baseline",
    name: "Baseline",
    description: "The current default reward configuration. Good starting point for comparison runs.",
    values: {},
    built_in: true,
  },
  {
    id: "speed-focus",
    name: "Speed Focus",
    description: "Emphasises forward velocity and tracking. Faster but may be less stable.",
    values: {
      rew_scale_forward_vel: 5.0,
      rew_scale_vel_tracking: 6.0,
      rew_scale_ang_vel_tracking: 3.5,
      rew_scale_orientation: -0.1,
      rew_scale_base_height: -0.1,
    },
    built_in: true,
  },
  {
    id: "stability-focus",
    name: "Stability Focus",
    description: "Strongly penalises tilting and height deviation for early stability.",
    values: {
      rew_scale_forward_vel: 1.5,
      rew_scale_vel_tracking: 2.0,
      rew_scale_orientation: -0.6,
      rew_scale_base_height: -0.6,
      rew_scale_lin_vel_z: -0.3,
      rew_scale_alive: 0.3,
    },
    built_in: true,
  },
];

export const REWARD_FIELDS = [
  {
    name: "Locomotion Goals",
    fields: [
      ["rew_scale_forward_vel", "Forward Velocity", "Rewards movement in the commanded direction."],
      ["rew_scale_vel_tracking", "Linear Tracking", "Rewards matching the commanded XY velocity."],
      ["rew_scale_ang_vel_tracking", "Turn Tracking", "Rewards matching commanded yaw rate."],
      ["rew_scale_vel_tracking2", "Aux Tracking", "Secondary velocity tracking term."],
      ["rew_scale_direction_align", "Direction Alignment", "Rewards movement aligned with command direction."],
    ],
  },
  {
    name: "Rotation And Legs",
    fields: [
      ["rew_scale_rotation_direction", "In-place Rotation", "Extra reward for correct in-place turns."],
      ["rew_scale_smooth_rotation", "Smooth Rotation", "Rewards smooth rotation without abrupt changes."],
      ["rew_scale_rotation_dir", "Leg Direction", "Rewards legs rotating in the correct direction."],
      ["rew_scale_all_legs", "All Legs Active", "Encourages all legs to participate."],
      ["rew_scale_min_leg_vel", "Minimum Leg Speed", "Keeps the slowest leg moving."],
      ["rew_scale_mean_leg_vel", "Mean Leg Speed", "Rewards average leg rotation speed."],
    ],
  },
  {
    name: "Stability",
    fields: [
      ["rew_scale_orientation", "Body Tilt Penalty", "Penalizes body tilt away from upright."],
      ["rew_scale_base_height", "Height Penalty", "Penalizes body height deviation."],
      ["rew_scale_lin_vel_z", "Bounce Penalty", "Penalizes vertical bouncing."],
      ["rew_scale_ang_vel_xy", "Roll/Pitch Penalty", "Penalizes roll and pitch wobble."],
      ["rew_scale_collision", "Body Collision", "Penalizes body-ground collisions."],
    ],
  },
  {
    name: "Gait And Control",
    fields: [
      ["rew_scale_gait_coherence", "Tripod Coherence", "Rewards sync within tripod groups."],
      ["rew_scale_gait_phase_offset", "Tripod Antiphase", "Rewards classic alternating tripod phase."],
      ["rew_scale_continuous_support", "Ground Contact", "Rewards continuous support."],
      ["rew_scale_abad_action", "ABAD Motion", "Rewards ABAD response to commands."],
      ["rew_scale_abad_stability", "ABAD Symmetry", "Rewards useful ABAD symmetry/asymmetry."],
      ["rew_scale_alive", "Alive Bonus", "Small bonus while the robot is alive."],
      ["rew_scale_action_rate", "Action Smoothness", "Penalizes rapid command changes."],
      ["rew_scale_drive_acc", "Drive Accel Penalty", "Penalizes sudden drive acceleration."],
    ],
  },
];

export const BUILT_IN_TERRAIN_PRESETS = [
  {
    id: "baseline",
    name: "Baseline",
    description: "The current terrain defaults from redrhex_env_cfg.py.",
    values: {},
    built_in: true,
  },
  {
    id: "flat-debug",
    name: "Flat Debug",
    description: "For quick debugging on a plane with terrain curriculum disabled.",
    values: {
      "terrain.terrain_type": "plane",
      "terrain.max_init_terrain_level": 0,
      terrain_curriculum_enable: false,
      terrain_curriculum_levels: [0.0],
    },
    built_in: true,
  },
  {
    id: "mild-mixed",
    name: "Mild Mixed",
    description: "A gentle rough/wave/stairs/boxes mix for early terrain training.",
    values: {
      "terrain.terrain_type": "generator",
      "terrain.max_init_terrain_level": 1,
      "terrain.terrain_generator.difficulty_range": [0.0, 0.1],
      terrain_curriculum_enable: true,
      terrain_curriculum_levels: [0.0, 0.05, 0.1, 0.16, 0.24],
      "terrain.terrain_generator.sub_terrains.random_rough.noise_range": [0.005, 0.035],
      "terrain.terrain_generator.sub_terrains.wave.amplitude_range": [0.005, 0.035],
      "terrain.terrain_generator.sub_terrains.stairs.step_height_range": [0.01, 0.07],
      "terrain.terrain_generator.sub_terrains.boxes.grid_height_range": [0.01, 0.07],
    },
    built_in: true,
  },
  {
    id: "rough-mixed",
    name: "Rough Mixed",
    description: "A stronger mixed-terrain profile for robustness work.",
    values: {
      "terrain.terrain_type": "generator",
      "terrain.max_init_terrain_level": 2,
      "terrain.terrain_generator.difficulty_range": [0.0, 0.3],
      terrain_curriculum_enable: true,
      terrain_curriculum_levels: [0.0, 0.12, 0.28, 0.45, 0.7],
      "terrain.terrain_generator.sub_terrains.flat.proportion": 0.1,
      "terrain.terrain_generator.sub_terrains.random_rough.proportion": 0.3,
      "terrain.terrain_generator.sub_terrains.wave.proportion": 0.2,
      "terrain.terrain_generator.sub_terrains.stairs.proportion": 0.2,
      "terrain.terrain_generator.sub_terrains.boxes.proportion": 0.2,
      "terrain.terrain_generator.sub_terrains.random_rough.noise_range": [0.02, 0.08],
      "terrain.terrain_generator.sub_terrains.wave.amplitude_range": [0.02, 0.08],
      "terrain.terrain_generator.sub_terrains.stairs.step_height_range": [0.03, 0.15],
      "terrain.terrain_generator.sub_terrains.boxes.grid_height_range": [0.03, 0.15],
    },
    built_in: true,
  },
  {
    id: "stairs-boxes",
    name: "Stairs + Boxes",
    description: "Focused obstacle profile for step and box-grid adaptation.",
    values: {
      "terrain.terrain_type": "generator",
      "terrain.max_init_terrain_level": 2,
      "terrain.terrain_generator.difficulty_range": [0.0, 0.25],
      terrain_curriculum_enable: true,
      terrain_curriculum_levels: [0.0, 0.1, 0.22, 0.38, 0.55],
      "terrain.terrain_generator.sub_terrains.flat.proportion": 0.1,
      "terrain.terrain_generator.sub_terrains.random_rough.proportion": 0.1,
      "terrain.terrain_generator.sub_terrains.wave.proportion": 0.05,
      "terrain.terrain_generator.sub_terrains.stairs.proportion": 0.4,
      "terrain.terrain_generator.sub_terrains.boxes.proportion": 0.35,
      "terrain.terrain_generator.sub_terrains.stairs.step_height_range": [0.02, 0.14],
      "terrain.terrain_generator.sub_terrains.stairs.step_width": 0.25,
      "terrain.terrain_generator.sub_terrains.boxes.grid_width": 0.4,
      "terrain.terrain_generator.sub_terrains.boxes.grid_height_range": [0.02, 0.14],
    },
    built_in: true,
  },
];

export const TERRAIN_DEFAULT_VALUES = {
  "terrain.terrain_type": "generator",
  "terrain.prim_path": "/World/ground",
  "terrain.collision_group": -1,
  "terrain.max_init_terrain_level": 1,
  "terrain.debug_vis": false,
  "terrain.physics_material.friction_combine_mode": "multiply",
  "terrain.physics_material.restitution_combine_mode": "multiply",
  "terrain.physics_material.static_friction": 1.2,
  "terrain.physics_material.dynamic_friction": 1.0,
  terrain_curriculum_enable: true,
  terrain_curriculum_levels: [0.0, 0.08, 0.2, 0.35, 0.55],
  "terrain.terrain_generator.size": [6.0, 6.0],
  "terrain.terrain_generator.border_width": 3.0,
  "terrain.terrain_generator.border_height": 1.0,
  "terrain.terrain_generator.num_rows": 6,
  "terrain.terrain_generator.num_cols": 12,
  "terrain.terrain_generator.curriculum": true,
  "terrain.terrain_generator.color_scheme": "none",
  "terrain.terrain_generator.horizontal_scale": 0.1,
  "terrain.terrain_generator.vertical_scale": 0.005,
  "terrain.terrain_generator.slope_threshold": 0.75,
  "terrain.terrain_generator.difficulty_range": [0.0, 0.15],
  "terrain.terrain_generator.use_cache": false,
  "terrain.terrain_generator.cache_dir": "/tmp/isaaclab/terrains",
  "terrain.terrain_generator.sub_terrains.flat.proportion": 0.2,
  "terrain.terrain_generator.sub_terrains.random_rough.proportion": 0.25,
  "terrain.terrain_generator.sub_terrains.random_rough.noise_range": [0.01, 0.06],
  "terrain.terrain_generator.sub_terrains.random_rough.noise_step": 0.005,
  "terrain.terrain_generator.sub_terrains.random_rough.border_width": 0.25,
  "terrain.terrain_generator.sub_terrains.wave.proportion": 0.15,
  "terrain.terrain_generator.sub_terrains.wave.amplitude_range": [0.01, 0.06],
  "terrain.terrain_generator.sub_terrains.wave.num_waves": 2,
  "terrain.terrain_generator.sub_terrains.wave.border_width": 0.25,
  "terrain.terrain_generator.sub_terrains.stairs.proportion": 0.2,
  "terrain.terrain_generator.sub_terrains.stairs.step_height_range": [0.02, 0.12],
  "terrain.terrain_generator.sub_terrains.stairs.step_width": 0.28,
  "terrain.terrain_generator.sub_terrains.stairs.platform_width": 1.2,
  "terrain.terrain_generator.sub_terrains.stairs.border_width": 0.25,
  "terrain.terrain_generator.sub_terrains.boxes.proportion": 0.2,
  "terrain.terrain_generator.sub_terrains.boxes.grid_width": 0.45,
  "terrain.terrain_generator.sub_terrains.boxes.grid_height_range": [0.02, 0.12],
  "terrain.terrain_generator.sub_terrains.boxes.platform_width": 1.5,
};

export const TERRAIN_FIELDS = [
  {
    name: "Importer",
    fields: [
      { key: "terrain.terrain_type", label: "Terrain Type", type: "choice", choices: ["generator", "plane", "usd"], help: "Isaac terrain source." },
      { key: "terrain.prim_path", label: "Prim Path", type: "string", help: "USD prim path used for the ground." },
      { key: "terrain.collision_group", label: "Collision Group", type: "int", step: 1, help: "Collision group assigned to terrain." },
      { key: "terrain.max_init_terrain_level", label: "Max Init Level", type: "int", step: 1, help: "Highest terrain row level available at reset." },
      { key: "terrain.debug_vis", label: "Debug Origins", type: "bool", help: "Show terrain origin debug visualization." },
    ],
  },
  {
    name: "Physics Material",
    fields: [
      { key: "terrain.physics_material.friction_combine_mode", label: "Friction Combine", type: "choice", choices: ["average", "min", "multiply", "max"], help: "How terrain friction combines with robot materials." },
      { key: "terrain.physics_material.restitution_combine_mode", label: "Restitution Combine", type: "choice", choices: ["average", "min", "multiply", "max"], help: "How bounce combines with robot materials." },
      { key: "terrain.physics_material.static_friction", label: "Static Friction", type: "float", step: 0.01, help: "Static friction coefficient." },
      { key: "terrain.physics_material.dynamic_friction", label: "Dynamic Friction", type: "float", step: 0.01, help: "Dynamic friction coefficient." },
    ],
  },
  {
    name: "Curriculum",
    fields: [
      { key: "terrain_curriculum_enable", label: "Terrain Curriculum", type: "bool", help: "Enable stage-based terrain difficulty updates." },
      { key: "terrain_curriculum_levels", label: "Curriculum Levels", type: "list", step: 0.01, help: "Difficulty level per curriculum stage." },
    ],
  },
  {
    name: "Generator",
    fields: [
      { key: "terrain.terrain_generator.size", label: "Tile Size", type: "range", step: 0.1, help: "Sub-terrain tile width and length in meters." },
      { key: "terrain.terrain_generator.border_width", label: "Border Width", type: "float", step: 0.1, help: "Outer terrain border width in meters." },
      { key: "terrain.terrain_generator.border_height", label: "Border Height", type: "float", step: 0.1, help: "Outer terrain border height in meters." },
      { key: "terrain.terrain_generator.num_rows", label: "Rows", type: "int", step: 1, help: "Terrain difficulty rows." },
      { key: "terrain.terrain_generator.num_cols", label: "Columns", type: "int", step: 1, help: "Terrain variation columns." },
      { key: "terrain.terrain_generator.curriculum", label: "Generator Curriculum", type: "bool", help: "Generate terrain rows in curriculum order." },
      { key: "terrain.terrain_generator.color_scheme", label: "Color Scheme", type: "choice", choices: ["height", "random", "none"], help: "Terrain visual color scheme." },
      { key: "terrain.terrain_generator.horizontal_scale", label: "Horizontal Scale", type: "float", step: 0.001, help: "Height-field XY discretization." },
      { key: "terrain.terrain_generator.vertical_scale", label: "Vertical Scale", type: "float", step: 0.001, help: "Height-field Z discretization." },
      { key: "terrain.terrain_generator.slope_threshold", label: "Slope Threshold", type: "float", step: 0.01, help: "Height-field slope correction threshold." },
      { key: "terrain.terrain_generator.difficulty_range", label: "Difficulty Range", type: "range", step: 0.01, help: "Generated terrain difficulty min and max." },
      { key: "terrain.terrain_generator.use_cache", label: "Use Cache", type: "bool", help: "Reuse generated terrain cache when available." },
      { key: "terrain.terrain_generator.cache_dir", label: "Cache Dir", type: "string", help: "Terrain cache directory." },
    ],
  },
  {
    name: "Flat",
    fields: [
      { key: "terrain.terrain_generator.sub_terrains.flat.proportion", label: "Flat Proportion", type: "float", step: 0.01, help: "Sampling weight for flat terrain." },
    ],
  },
  {
    name: "Random Rough",
    fields: [
      { key: "terrain.terrain_generator.sub_terrains.random_rough.proportion", label: "Rough Proportion", type: "float", step: 0.01, help: "Sampling weight for random rough terrain." },
      { key: "terrain.terrain_generator.sub_terrains.random_rough.noise_range", label: "Noise Range", type: "range", step: 0.001, help: "Min and max rough terrain height noise in meters." },
      { key: "terrain.terrain_generator.sub_terrains.random_rough.noise_step", label: "Noise Step", type: "float", step: 0.001, help: "Minimum height change between samples." },
      { key: "terrain.terrain_generator.sub_terrains.random_rough.border_width", label: "Border Width", type: "float", step: 0.01, help: "Flat border around rough tile." },
    ],
  },
  {
    name: "Wave",
    fields: [
      { key: "terrain.terrain_generator.sub_terrains.wave.proportion", label: "Wave Proportion", type: "float", step: 0.01, help: "Sampling weight for wave terrain." },
      { key: "terrain.terrain_generator.sub_terrains.wave.amplitude_range", label: "Amplitude Range", type: "range", step: 0.001, help: "Min and max wave amplitude in meters." },
      { key: "terrain.terrain_generator.sub_terrains.wave.num_waves", label: "Wave Count", type: "int", step: 1, help: "Number of waves per terrain tile." },
      { key: "terrain.terrain_generator.sub_terrains.wave.border_width", label: "Border Width", type: "float", step: 0.01, help: "Flat border around wave tile." },
    ],
  },
  {
    name: "Stairs",
    fields: [
      { key: "terrain.terrain_generator.sub_terrains.stairs.proportion", label: "Stairs Proportion", type: "float", step: 0.01, help: "Sampling weight for pyramid stairs terrain." },
      { key: "terrain.terrain_generator.sub_terrains.stairs.step_height_range", label: "Step Height Range", type: "range", step: 0.001, help: "Min and max stair height in meters." },
      { key: "terrain.terrain_generator.sub_terrains.stairs.step_width", label: "Step Width", type: "float", step: 0.01, help: "Stair tread width in meters." },
      { key: "terrain.terrain_generator.sub_terrains.stairs.platform_width", label: "Platform Width", type: "float", step: 0.1, help: "Central platform width in meters." },
      { key: "terrain.terrain_generator.sub_terrains.stairs.border_width", label: "Border Width", type: "float", step: 0.01, help: "Flat border around stairs tile." },
    ],
  },
  {
    name: "Boxes",
    fields: [
      { key: "terrain.terrain_generator.sub_terrains.boxes.proportion", label: "Boxes Proportion", type: "float", step: 0.01, help: "Sampling weight for random box grid terrain." },
      { key: "terrain.terrain_generator.sub_terrains.boxes.grid_width", label: "Grid Width", type: "float", step: 0.01, help: "Random grid cell width in meters." },
      { key: "terrain.terrain_generator.sub_terrains.boxes.grid_height_range", label: "Grid Height Range", type: "range", step: 0.001, help: "Min and max random grid height in meters." },
      { key: "terrain.terrain_generator.sub_terrains.boxes.platform_width", label: "Platform Width", type: "float", step: 0.1, help: "Central platform width in meters." },
    ],
  },
];

export const ACTIVE_JOB_STATUSES = new Set(["queued", "claimed", "running"]);
export const GPU_JOB_TYPES = new Set(["start_training", "record_video", "export_onnx"]);
export const ACTIVE_REFRESH_MS = 3_000;
export const IDLE_REFRESH_MS = 5_000;

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export function slugify(value) {
  const slug = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug || `preset-${Date.now()}`;
}

export function canOperate(role) {
  return role === "operator" || role === "admin";
}

export function canEditPreset(role) {
  return canOperate(role);
}

export function canEditRun(role) {
  return canOperate(role);
}

export function isMachineFresh(machine, now = Date.now()) {
  if (!machine?.heartbeat_at) return false;
  const heartbeat = Date.parse(machine.heartbeat_at);
  return Number.isFinite(heartbeat) && now - heartbeat <= HEARTBEAT_STALE_MS;
}

export function machineState(machine, now = Date.now()) {
  if (!machine) return "missing";
  if (!isMachineFresh(machine, now)) return "offline";
  if (machine.accept_jobs) return machine.gpu_locked ? "busy" : "ready";
  return "paused";
}

export function statusTone(status) {
  const normalized = String(status || "").toLowerCase();
  if (["completed", "ready", "online", "accepting", "success"].includes(normalized)) return "good";
  if (["running", "claimed", "queued", "busy"].includes(normalized)) return "info";
  if (["failed", "cancelled", "offline", "missing"].includes(normalized)) return "bad";
  return "muted";
}

export function hasActiveRemoteWork(snapshot = {}) {
  const jobs = Array.isArray(snapshot.jobs) ? snapshot.jobs : [];
  const machine = snapshot.targetMachine || snapshot.machine || null;
  return Boolean(machine?.gpu_locked) || jobs.some((job) => ACTIVE_JOB_STATUSES.has(String(job.status || "").toLowerCase()));
}

export function refreshDelayForSnapshot(snapshot = {}) {
  return hasActiveRemoteWork(snapshot) ? ACTIVE_REFRESH_MS : IDLE_REFRESH_MS;
}

export function jobQueueLabel(job = {}, machine = null) {
  const status = String(job.status || "").toLowerCase();
  const type = String(job.type || "");
  if (status === "queued") {
    if (!machine) return "waiting for worker";
    if (!isMachineFresh(machine)) return "waiting for worker heartbeat";
    if (!machine.accept_jobs) return "worker paused";
    if (machine.gpu_locked && GPU_JOB_TYPES.has(type)) return "waiting for GPU";
    return "waiting for worker";
  }
  if (status === "claimed") return "claimed by worker";
  if (status === "running") return "running";
  if (status === "failed") return job.error || "failed";
  if (status === "completed") return "completed";
  return status || "unknown";
}

export function latestVideoArtifact(runId, artifacts = []) {
  const matches = artifacts
    .filter((artifact) => artifact.run_id === runId && artifact.kind === "video" && artifact.storage_path)
    .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
  return matches[0] || null;
}

export function hasAnyVideoRecord(run = {}, artifacts = []) {
  return Boolean(run.latest_video) || artifacts.some((artifact) => artifact.run_id === run.id && artifact.kind === "video");
}

export function videoStateForRun(run = {}, artifacts = []) {
  const artifact = latestVideoArtifact(run.id, artifacts);
  if (artifact) return { state: "ready", artifact };
  if (hasAnyVideoRecord(run, artifacts)) return { state: "uploading", artifact: null };
  if (run.latest_checkpoint) return { state: "recordable", artifact: null };
  return { state: "missing", artifact: null };
}

export function checkpointIteration(path = "") {
  const match = String(path || "").match(/model_(\d+)\.pt(?:$|[?#])/);
  return match ? Number(match[1]) : null;
}

export function checkpointArtifactsForRun(run = {}, artifacts = []) {
  const byPath = new Map();
  if (run.latest_checkpoint) {
    byPath.set(String(run.latest_checkpoint), {
      run_id: run.id,
      kind: "checkpoint",
      local_path: String(run.latest_checkpoint),
      path: String(run.latest_checkpoint),
      iteration: checkpointIteration(run.latest_checkpoint),
    });
  }
  artifacts
    .filter((artifact) => artifact.run_id === run.id && artifact.kind === "checkpoint")
    .forEach((artifact) => {
      const path = String(artifact.local_path || artifact.path || "");
      if (!path) return;
      byPath.set(path, { ...artifact, iteration: checkpointIteration(path) });
    });
  return [...byPath.values()].sort((left, right) => {
    const leftIteration = Number.isFinite(left.iteration) ? left.iteration : -1;
    const rightIteration = Number.isFinite(right.iteration) ? right.iteration : -1;
    if (leftIteration !== rightIteration) return rightIteration - leftIteration;
    return String(right.created_at || "").localeCompare(String(left.created_at || ""));
  });
}

export function checkpointOptionsForRun(run = {}, artifacts = []) {
  return checkpointArtifactsForRun(run, artifacts).map((artifact) => {
    const path = String(artifact.local_path || artifact.path || "");
    const iteration = checkpointIteration(path);
    const label = Number.isFinite(iteration) ? `Iteration ${iteration}` : path.split("/").pop() || "Checkpoint";
    return { path, iteration, label };
  });
}

export function videoArtifactForCheckpoint(run = {}, artifacts = [], checkpoint = "") {
  const iteration = checkpointIteration(checkpoint);
  const videos = artifacts
    .filter((artifact) => artifact.run_id === run.id && artifact.kind === "video" && artifact.storage_path)
    .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
  if (Number.isFinite(iteration)) {
    const pattern = new RegExp(`model_${iteration}(?:\\D|$)`);
    const exact = videos.find((artifact) => pattern.test(`${artifact.storage_path || ""} ${artifact.local_path || ""}`));
    if (exact) return exact;
  }
  if (checkpoint && run.latest_checkpoint && String(checkpoint) === String(run.latest_checkpoint)) {
    return videos[0] || null;
  }
  return null;
}

export function hasVideoRecordForCheckpoint(run = {}, artifacts = [], checkpoint = "") {
  if (videoArtifactForCheckpoint(run, artifacts, checkpoint)) return true;
  const iteration = checkpointIteration(checkpoint);
  if (!Number.isFinite(iteration)) return hasAnyVideoRecord(run, artifacts);
  const pattern = new RegExp(`model_${iteration}(?:\\D|$)`);
  return Boolean(run.latest_video && pattern.test(String(run.latest_video))) || artifacts.some((artifact) => (
    artifact.run_id === run.id
    && artifact.kind === "video"
    && pattern.test(`${artifact.local_path || ""} ${artifact.storage_path || ""}`)
  ));
}

export function videoStateForCheckpoint(run = {}, artifacts = [], checkpoint = "") {
  const selectedCheckpoint = checkpoint || run.latest_checkpoint || checkpointOptionsForRun(run, artifacts)[0]?.path || "";
  const artifact = videoArtifactForCheckpoint(run, artifacts, selectedCheckpoint);
  if (artifact) return { state: "ready", artifact, checkpoint: selectedCheckpoint };
  if (hasVideoRecordForCheckpoint(run, artifacts, selectedCheckpoint)) return { state: "uploading", artifact: null, checkpoint: selectedCheckpoint };
  if (selectedCheckpoint) return { state: "recordable", artifact: null, checkpoint: selectedCheckpoint };
  return { state: "missing", artifact: null, checkpoint: "" };
}

export function shouldReplaceVideoPanel({
  currentState = "",
  currentStorage = "",
  nextState = "",
  nextStorage = "",
  isPlaying = false,
} = {}) {
  if (isPlaying && currentStorage && currentStorage === nextStorage) return false;
  return currentState !== nextState || currentStorage !== nextStorage;
}

export function buildRunMetadataPatch({ displayName = "", folder = "", notes = "", now = new Date() } = {}) {
  const nameValue = String(displayName || "").trim();
  const folderValue = String(folder || "").trim();
  return {
    display_name: nameValue || null,
    folder: folderValue || null,
    notes: String(notes || ""),
    updated_at: now.toISOString(),
  };
}

export function friendlyErrorMessage(error) {
  const message = error?.message || String(error || "");
  if (/schema cache|could not find.*schema|could not find the table/i.test(message)) {
    return `${message} Apply the latest schema.sql in Supabase, then reload the PostgREST schema cache.`;
  }
  if (/storage|bucket|signed.*url|object/i.test(message)) {
    return `${message} Check the private redrhex-videos bucket and storage policies.`;
  }
  return message;
}

export function formatRelativeTime(iso, now = Date.now()) {
  if (!iso) return "unknown";
  const then = Date.parse(iso);
  if (!Number.isFinite(then)) return String(iso);
  const seconds = Math.max(0, Math.floor((now - then) / 1000));
  if (seconds < 45) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function buildTrainingJob({ machineId, params, preset, terrainPreset, role, userId }) {
  const rewardValues = preset?.values && typeof preset.values === "object" ? preset.values : {};
  const terrainValues = terrainPreset?.values && typeof terrainPreset.values === "object" ? terrainPreset.values : {};
  return {
    machine_id: machineId || null,
    type: "start_training",
    actor_id: userId || null,
    actor_role: role || "viewer",
    payload: {
      ...params,
      headless: true,
      reward_preset_id: preset?.id || "baseline",
      reward_overrides: rewardValues,
      terrain_preset_id: terrainPreset?.id || "baseline",
      terrain_overrides: terrainValues,
    },
  };
}

function objectValues(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function runParams(run = {}) {
  return objectValues(run.params);
}

function presetValuesById(presets = [], presetId = "") {
  const preset = presets.find((item) => String(item?.id || "") === String(presetId || ""));
  return objectValues(preset?.values);
}

export function isFinishedTweakRun(run = {}) {
  const status = String(run.status || "").toLowerCase();
  return status !== "running" && status !== "stopping";
}

export function rewardValuesForTweak(run = {}, presets = []) {
  const params = runParams(run);
  const overrides = objectValues(params.reward_overrides);
  if (Object.keys(overrides).length) return { ...overrides };
  const runOverrides = objectValues(run.reward_overrides);
  if (Object.keys(runOverrides).length) return { ...runOverrides };
  return { ...presetValuesById(presets, params.reward_preset_id || run.reward_preset_id) };
}

export function terrainValuesForTweak(run = {}, terrainPresets = []) {
  const params = runParams(run);
  const overrides = objectValues(params.terrain_overrides);
  if (Object.keys(overrides).length) return { ...overrides };
  const runOverrides = objectValues(run.terrain_overrides);
  if (Object.keys(runOverrides).length) return { ...runOverrides };
  return { ...presetValuesById(terrainPresets, params.terrain_preset_id || run.terrain_preset_id) };
}

export function canBuildTweakFromRun(run = {}, presets = []) {
  if (!isFinishedTweakRun(run)) return false;
  const params = runParams(run);
  return Object.keys(params).length > 0 || Object.keys(rewardValuesForTweak(run, presets)).length > 0;
}

export function latestFinishedTweakRun(runs = [], presets = []) {
  return [...runs]
    .sort((a, b) => String(b.created_at || b.updated_at || "").localeCompare(String(a.created_at || a.updated_at || "")))
    .find((run) => canBuildTweakFromRun(run, presets)) || null;
}

export function buildTweakDraftFromRun(run = {}, { presets = [], terrainPresets = [] } = {}) {
  if (!canBuildTweakFromRun(run, presets)) throw new Error("Run does not have usable tweak data.");
  const params = runParams(run);
  const sourceId = String(run.id || "");
  const sourceLabel = String(run.display_name || run.id || "run");
  const rewardValues = rewardValuesForTweak(run, presets);
  const terrainPresetId = String(params.terrain_preset_id || run.terrain_preset_id || "baseline");
  const terrainValues = terrainValuesForTweak(run, terrainPresets);
  const draftId = `tweak-${slugify(sourceId || sourceLabel)}`;
  return {
    source_run: {
      id: sourceId,
      display_name: run.display_name || "",
      status: run.status || "",
      created_at: run.created_at || "",
      updated_at: run.updated_at || "",
      log_dir: run.log_dir || "",
    },
    training_params: {
      task: params.task || "Template-Redrhex-Direct-v0",
      num_envs: Number(params.num_envs || 4),
      max_iterations: Number(params.max_iterations || 8),
      device: params.device || "cuda:0",
      headless: params.headless !== false,
      seed: params.seed ?? null,
      resume: false,
      checkpoint: "",
      reward_preset_id: draftId,
      reward_overrides: rewardValues,
      terrain_preset_id: terrainPresetId,
      terrain_overrides: terrainValues,
      tweak_source_run_id: sourceId,
      tweak_source_label: sourceLabel,
    },
    reward_preset: {
      id: draftId,
      name: `Tweak from ${sourceLabel}`,
      description: `Unsaved reward draft copied from ${sourceLabel}.`,
      values: rewardValues,
      built_in: false,
      draft: true,
      source_run_id: sourceId,
      source_label: sourceLabel,
    },
    terrain_preset_id: terrainPresetId,
    terrain_overrides: terrainValues,
    message: `Loaded tweak draft from ${sourceLabel}.`,
  };
}

export function buildActionJob({ machineId, type, runId, role, userId, payload = {} }) {
  return {
    machine_id: machineId || null,
    type,
    actor_id: userId || null,
    actor_role: role || "viewer",
    payload: { run_id: runId, ...payload },
  };
}

export function normalizePreset(raw) {
  return {
    id: String(raw?.id || ""),
    name: String(raw?.name || raw?.id || "Preset"),
    description: String(raw?.description || ""),
    values: raw?.values && typeof raw.values === "object" ? raw.values : {},
    built_in: Boolean(raw?.built_in),
    updated_at: raw?.updated_at || raw?.created_at || "",
  };
}

export function normalizeTerrainPreset(raw) {
  return normalizePreset(raw);
}
