-- RedRHex Training Panel V3.4 Supabase schema
-- Apply in the Supabase SQL editor, then configure Row Level Security policies
-- for your team's auth provider.

do $$
begin
  create type public.redrhex_role as enum ('viewer', 'operator', 'admin');
exception when duplicate_object then null;
end $$;

do $$
begin
  create type public.redrhex_job_status as enum ('queued', 'claimed', 'running', 'completed', 'failed', 'cancelled');
exception when duplicate_object then null;
end $$;

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  display_name text,
  role public.redrhex_role not null default 'viewer',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.machines (
  machine_id text primary key,
  online boolean not null default false,
  accept_jobs boolean not null default false,
  panel_version text,
  repo_root text,
  rsl_rl_log_root text,
  active_job_id uuid,
  queue_depth integer not null default 0,
  gpu_locked boolean not null default false,
  tunnel_host text,
  heartbeat_at timestamptz,
  last_sync_at timestamptz,
  last_sync_duration_ms integer,
  last_sync_error text not null default '',
  last_sync_summary jsonb not null default '{}',
  updated_at timestamptz not null default now()
);

alter table public.machines add column if not exists last_sync_at timestamptz;
alter table public.machines add column if not exists last_sync_duration_ms integer;
alter table public.machines add column if not exists last_sync_error text not null default '';
alter table public.machines add column if not exists last_sync_summary jsonb not null default '{}';

create table if not exists public.runs (
  id text primary key,
  machine_id text references public.machines(machine_id),
  status text,
  display_name text,
  notes text,
  folder text,
  params jsonb not null default '{}',
  log_dir text,
  latest_checkpoint text,
  latest_video text,
  onnx_path text,
  created_by uuid references auth.users(id),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.runs add column if not exists notes text;
alter table public.runs add column if not exists folder text;
alter table public.runs add column if not exists created_by uuid references auth.users(id);
alter table public.runs add column if not exists convergence_detected boolean;
alter table public.runs add column if not exists convergence_iteration integer;
alter table public.runs add column if not exists convergence_improvement_pct numeric;
alter table public.runs add column if not exists video_status text;
alter table public.runs add column if not exists returncode integer;

create table if not exists public.run_deletions (
  machine_id text not null references public.machines(machine_id) on delete cascade,
  id text not null,
  log_dir text,
  log_dir_name text,
  deleted_by uuid references auth.users(id),
  deleted_at timestamptz not null default now(),
  metadata jsonb not null default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (machine_id, id)
);

alter table public.run_deletions add column if not exists log_dir text;
alter table public.run_deletions add column if not exists log_dir_name text;
alter table public.run_deletions add column if not exists deleted_by uuid references auth.users(id);
alter table public.run_deletions add column if not exists metadata jsonb not null default '{}';

create table if not exists public.reward_presets (
  id text primary key,
  name text not null,
  description text not null default '',
  values jsonb not null default '{}',
  built_in boolean not null default false,
  created_by uuid references auth.users(id),
  updated_by uuid references auth.users(id),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

insert into public.reward_presets (id, name, description, values, built_in)
values
  ('baseline', 'Baseline', 'The current default reward configuration. Good starting point for comparison runs.', '{}'::jsonb, true),
  ('speed-focus', 'Speed Focus', 'Emphasises forward velocity and tracking. Faster but may be less stable.', '{"rew_scale_forward_vel":5.0,"rew_scale_vel_tracking":6.0,"rew_scale_ang_vel_tracking":3.5,"rew_scale_orientation":-0.1,"rew_scale_base_height":-0.1}'::jsonb, true),
  ('stability-focus', 'Stability Focus', 'Strongly penalises tilting and height deviation for early stability.', '{"rew_scale_forward_vel":1.5,"rew_scale_vel_tracking":2.0,"rew_scale_orientation":-0.6,"rew_scale_base_height":-0.6,"rew_scale_lin_vel_z":-0.3,"rew_scale_alive":0.3}'::jsonb, true)
on conflict (id) do update
set name = excluded.name,
    description = excluded.description,
    values = excluded.values,
    built_in = true,
    updated_at = now();

create table if not exists public.terrain_presets (
  id text primary key,
  name text not null,
  description text not null default '',
  values jsonb not null default '{}',
  built_in boolean not null default false,
  created_by uuid references auth.users(id),
  updated_by uuid references auth.users(id),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

insert into public.terrain_presets (id, name, description, values, built_in)
values
  ('baseline', 'Baseline', 'The current terrain defaults from redrhex_env_cfg.py.', '{}'::jsonb, true),
  ('flat-debug', 'Flat Debug', 'For quick debugging on a plane with terrain curriculum disabled.', '{"terrain.terrain_type":"plane","terrain.max_init_terrain_level":0,"terrain_curriculum_enable":false,"terrain_curriculum_levels":[0.0]}'::jsonb, true),
  ('mild-mixed', 'Mild Mixed', 'A gentle rough/wave/stairs/boxes mix for early terrain training.', '{"terrain.terrain_type":"generator","terrain.max_init_terrain_level":1,"terrain.terrain_generator.difficulty_range":[0.0,0.10],"terrain_curriculum_enable":true,"terrain_curriculum_levels":[0.0,0.05,0.10,0.16,0.24],"terrain.terrain_generator.sub_terrains.random_rough.noise_range":[0.005,0.035],"terrain.terrain_generator.sub_terrains.wave.amplitude_range":[0.005,0.035],"terrain.terrain_generator.sub_terrains.stairs.step_height_range":[0.01,0.07],"terrain.terrain_generator.sub_terrains.boxes.grid_height_range":[0.01,0.07]}'::jsonb, true),
  ('rough-mixed', 'Rough Mixed', 'A stronger mixed-terrain profile for robustness work.', '{"terrain.terrain_type":"generator","terrain.max_init_terrain_level":2,"terrain.terrain_generator.difficulty_range":[0.0,0.30],"terrain_curriculum_enable":true,"terrain_curriculum_levels":[0.0,0.12,0.28,0.45,0.70],"terrain.terrain_generator.sub_terrains.flat.proportion":0.10,"terrain.terrain_generator.sub_terrains.random_rough.proportion":0.30,"terrain.terrain_generator.sub_terrains.wave.proportion":0.20,"terrain.terrain_generator.sub_terrains.stairs.proportion":0.20,"terrain.terrain_generator.sub_terrains.boxes.proportion":0.20,"terrain.terrain_generator.sub_terrains.random_rough.noise_range":[0.02,0.08],"terrain.terrain_generator.sub_terrains.wave.amplitude_range":[0.02,0.08],"terrain.terrain_generator.sub_terrains.stairs.step_height_range":[0.03,0.15],"terrain.terrain_generator.sub_terrains.boxes.grid_height_range":[0.03,0.15]}'::jsonb, true),
  ('stairs-boxes', 'Stairs + Boxes', 'Focused obstacle profile for step and box-grid adaptation.', '{"terrain.terrain_type":"generator","terrain.max_init_terrain_level":2,"terrain.terrain_generator.difficulty_range":[0.0,0.25],"terrain_curriculum_enable":true,"terrain_curriculum_levels":[0.0,0.10,0.22,0.38,0.55],"terrain.terrain_generator.sub_terrains.flat.proportion":0.10,"terrain.terrain_generator.sub_terrains.random_rough.proportion":0.10,"terrain.terrain_generator.sub_terrains.wave.proportion":0.05,"terrain.terrain_generator.sub_terrains.stairs.proportion":0.40,"terrain.terrain_generator.sub_terrains.boxes.proportion":0.35,"terrain.terrain_generator.sub_terrains.stairs.step_height_range":[0.02,0.14],"terrain.terrain_generator.sub_terrains.stairs.step_width":0.25,"terrain.terrain_generator.sub_terrains.boxes.grid_width":0.40,"terrain.terrain_generator.sub_terrains.boxes.grid_height_range":[0.02,0.14]}'::jsonb, true)
on conflict (id) do update
set name = excluded.name,
    description = excluded.description,
    values = excluded.values,
    built_in = true,
    updated_at = now();

create table if not exists public.jobs (
  id uuid primary key default gen_random_uuid(),
  machine_id text references public.machines(machine_id),
  type text not null,
  status public.redrhex_job_status not null default 'queued',
  payload jsonb not null default '{}',
  result jsonb not null default '{}',
  error text,
  actor_id uuid references auth.users(id),
  actor_role public.redrhex_role not null default 'viewer',
  claimed_by text,
  claimed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.run_events (
  id uuid primary key default gen_random_uuid(),
  run_id text references public.runs(id) on delete cascade,
  machine_id text references public.machines(machine_id),
  event_type text not null,
  payload jsonb not null default '{}',
  discord_sent_at timestamptz,
  recipient_id uuid references auth.users(id),
  event_key text,
  notification_status text not null default 'pending',
  channel_results jsonb not null default '{}',
  notified_at timestamptz,
  created_at timestamptz not null default now()
  -- No unique constraint on (run_id, event_type): a run may complete multiple times
  -- after resume, so multiple events of the same type must be allowed.
);

create table if not exists public.team_activity_events (
  id uuid primary key default gen_random_uuid(),
  machine_id text references public.machines(machine_id),
  actor_id uuid references auth.users(id),
  actor_name text,
  actor_role public.redrhex_role,
  event_type text not null,
  category text not null default 'system',
  outcome text not null default 'info',
  run_id text references public.runs(id) on delete set null,
  job_id uuid references public.jobs(id) on delete set null,
  points integer not null default 0,
  metadata jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create table if not exists public.artifacts (
  id uuid primary key default gen_random_uuid(),
  run_id text references public.runs(id) on delete cascade,
  machine_id text references public.machines(machine_id),
  kind text not null,
  local_path text,
  storage_path text,
  public_url text,
  bytes bigint,
  created_at timestamptz not null default now(),
  unique (run_id, kind, local_path)
);

alter table public.artifacts add column if not exists storage_path text;
alter table public.artifacts add column if not exists public_url text;
alter table public.artifacts add column if not exists bytes bigint;

alter table public.run_events drop constraint if exists run_events_run_id_event_type_key;
alter table public.run_events add column if not exists recipient_id uuid references auth.users(id);
alter table public.run_events add column if not exists event_key text;
alter table public.run_events add column if not exists notification_status text not null default 'pending';
alter table public.run_events add column if not exists channel_results jsonb not null default '{}';
alter table public.run_events add column if not exists notified_at timestamptz;
alter table public.run_events drop column if exists email_sent_at;

-- proxy_sessions: reserved scaffold for future TensorBoard/play proxy auth.
-- Not yet used by any application code.
create table if not exists public.proxy_sessions (
  id uuid primary key default gen_random_uuid(),
  run_id text references public.runs(id) on delete cascade,
  machine_id text references public.machines(machine_id),
  user_id uuid references auth.users(id),
  service_type text not null,
  token_hash text not null,
  expires_at timestamptz not null,
  created_at timestamptz not null default now()
);

create table if not exists public.notification_settings (
  id uuid primary key default gen_random_uuid(),
  machine_id text references public.machines(machine_id),
  user_id uuid references auth.users(id) on delete cascade,
  discord_enabled boolean not null default false,
  discord_webhook_url text not null default '',
  notify_training_converged boolean not null default true,
  notify_training_completed boolean not null default true,
  notify_training_failed boolean not null default true,
  notify_video_ready boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.notification_settings add column if not exists user_id uuid references auth.users(id) on delete cascade;
alter table public.notification_settings add column if not exists discord_webhook_url text not null default '';
alter table public.notification_settings add column if not exists notify_training_converged boolean not null default true;
alter table public.notification_settings add column if not exists notify_training_completed boolean not null default true;
alter table public.notification_settings add column if not exists notify_training_failed boolean not null default true;
alter table public.notification_settings add column if not exists notify_video_ready boolean not null default true;
alter table public.notification_settings drop column if exists email_enabled;
alter table public.notification_settings drop column if exists email_recipients;

create or replace function public.set_redrhex_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists set_runs_updated_at on public.runs;
create trigger set_runs_updated_at
  before update on public.runs
  for each row execute function public.set_redrhex_updated_at();

drop trigger if exists set_jobs_updated_at on public.jobs;
create trigger set_jobs_updated_at
  before update on public.jobs
  for each row execute function public.set_redrhex_updated_at();

drop trigger if exists set_reward_presets_updated_at on public.reward_presets;
create trigger set_reward_presets_updated_at
  before update on public.reward_presets
  for each row execute function public.set_redrhex_updated_at();

drop trigger if exists set_terrain_presets_updated_at on public.terrain_presets;
create trigger set_terrain_presets_updated_at
  before update on public.terrain_presets
  for each row execute function public.set_redrhex_updated_at();

drop trigger if exists set_machines_updated_at on public.machines;
create trigger set_machines_updated_at
  before update on public.machines
  for each row execute function public.set_redrhex_updated_at();

drop trigger if exists set_run_deletions_updated_at on public.run_deletions;
create trigger set_run_deletions_updated_at
  before update on public.run_deletions
  for each row execute function public.set_redrhex_updated_at();

drop trigger if exists set_notification_settings_updated_at on public.notification_settings;
create trigger set_notification_settings_updated_at
  before update on public.notification_settings
  for each row execute function public.set_redrhex_updated_at();

create or replace function public.claim_next_job_for_machine(p_machine_id text, p_gpu_locked boolean default false)
returns setof public.jobs
language plpgsql
security definer
as $$
begin
  return query
  with next_job as (
    select id
    from public.jobs
    where status = 'queued'
      and (machine_id = p_machine_id or machine_id is null)
      and (not p_gpu_locked or type in ('stop_process', 'tensorboard'))
    order by
      case when type in ('stop_process', 'tensorboard') then 0 else 1 end,
      created_at asc
    for update skip locked
    limit 1
  )
  update public.jobs j
  set status = 'claimed',
      claimed_by = p_machine_id,
      claimed_at = now(),
      updated_at = now()
  from next_job
  where j.id = next_job.id
  returning j.*;
end;
$$;

alter table public.profiles enable row level security;
alter table public.machines enable row level security;
alter table public.runs enable row level security;
alter table public.run_deletions enable row level security;
alter table public.jobs enable row level security;
alter table public.run_events enable row level security;
alter table public.team_activity_events enable row level security;
alter table public.artifacts enable row level security;
alter table public.proxy_sessions enable row level security;
alter table public.notification_settings enable row level security;
alter table public.reward_presets enable row level security;
alter table public.terrain_presets enable row level security;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('redrhex-videos', 'redrhex-videos', false, 1073741824, array['video/mp4', 'image/png'])
on conflict (id) do update
set public = false,
    file_size_limit = excluded.file_size_limit,
    allowed_mime_types = excluded.allowed_mime_types;

drop policy if exists "profiles readable by authenticated users" on public.profiles;
drop policy if exists "runs readable by authenticated users" on public.runs;
drop policy if exists "run deletions readable by authenticated users" on public.run_deletions;
drop policy if exists "operators can update remote run metadata" on public.runs;
drop policy if exists "artifacts readable by authenticated users" on public.artifacts;
drop policy if exists "machines readable by authenticated users" on public.machines;
drop policy if exists "jobs readable by authenticated users" on public.jobs;
drop policy if exists "operators can create jobs" on public.jobs;
drop policy if exists "reward presets readable by authenticated users" on public.reward_presets;
drop policy if exists "operators can create reward presets" on public.reward_presets;
drop policy if exists "operators can update custom reward presets" on public.reward_presets;
drop policy if exists "operators can delete custom reward presets" on public.reward_presets;
drop policy if exists "terrain presets readable by authenticated users" on public.terrain_presets;
drop policy if exists "operators can create terrain presets" on public.terrain_presets;
drop policy if exists "operators can update custom terrain presets" on public.terrain_presets;
drop policy if exists "operators can delete custom terrain presets" on public.terrain_presets;
drop policy if exists "run_events readable by authenticated users" on public.run_events;
drop policy if exists "machine can insert run events" on public.run_events;
drop policy if exists "machine can update own run events" on public.run_events;
drop policy if exists "team activity readable by authenticated users" on public.team_activity_events;
drop policy if exists "machine can insert team activity" on public.team_activity_events;
drop policy if exists "users can read own notification settings" on public.notification_settings;
drop policy if exists "users can create own notification settings" on public.notification_settings;
drop policy if exists "users can update own notification settings" on public.notification_settings;
drop policy if exists "users can delete own notification settings" on public.notification_settings;
drop policy if exists "machine can upsert own row" on public.machines;
drop policy if exists "machine can upsert own runs" on public.runs;
drop policy if exists "machine can upsert own run deletions" on public.run_deletions;
drop policy if exists "machine can upsert own artifacts" on public.artifacts;
drop policy if exists "authenticated users can read redrhex videos" on storage.objects;

create policy "profiles readable by authenticated users" on public.profiles
  for select to authenticated using (true);

create policy "runs readable by authenticated users" on public.runs
  for select to authenticated using (true);

create policy "run deletions readable by authenticated users" on public.run_deletions
  for select to authenticated using (true);

create policy "operators can update remote run metadata" on public.runs
  for update to authenticated
  using (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid()
        and p.role in ('operator', 'admin')
    )
  )
  with check (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid()
        and p.role in ('operator', 'admin')
    )
  );

create policy "artifacts readable by authenticated users" on public.artifacts
  for select to authenticated using (true);

create policy "machines readable by authenticated users" on public.machines
  for select to authenticated using (true);

create policy "jobs readable by authenticated users" on public.jobs
  for select to authenticated using (true);

create policy "operators can create jobs" on public.jobs
  for insert to authenticated
  with check (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid()
        and p.role in ('operator', 'admin')
    )
  );

create policy "reward presets readable by authenticated users" on public.reward_presets
  for select to authenticated using (true);

create policy "operators can create reward presets" on public.reward_presets
  for insert to authenticated
  with check (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid()
        and p.role in ('operator', 'admin')
    )
  );

create policy "operators can update custom reward presets" on public.reward_presets
  for update to authenticated
  using (
    built_in = false
    and exists (
      select 1 from public.profiles p
      where p.id = auth.uid()
        and p.role in ('operator', 'admin')
    )
  )
  with check (
    built_in = false
    and exists (
      select 1 from public.profiles p
      where p.id = auth.uid()
        and p.role in ('operator', 'admin')
    )
  );

create policy "operators can delete custom reward presets" on public.reward_presets
  for delete to authenticated
  using (
    built_in = false
    and exists (
      select 1 from public.profiles p
      where p.id = auth.uid()
        and p.role in ('operator', 'admin')
    )
  );

create policy "terrain presets readable by authenticated users" on public.terrain_presets
  for select to authenticated using (true);

create policy "operators can create terrain presets" on public.terrain_presets
  for insert to authenticated
  with check (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid()
        and p.role in ('operator', 'admin')
    )
  );

create policy "operators can update custom terrain presets" on public.terrain_presets
  for update to authenticated
  using (
    built_in = false
    and exists (
      select 1 from public.profiles p
      where p.id = auth.uid()
        and p.role in ('operator', 'admin')
    )
  )
  with check (
    built_in = false
    and exists (
      select 1 from public.profiles p
      where p.id = auth.uid()
        and p.role in ('operator', 'admin')
    )
  );

create policy "operators can delete custom terrain presets" on public.terrain_presets
  for delete to authenticated
  using (
    built_in = false
    and exists (
      select 1 from public.profiles p
      where p.id = auth.uid()
        and p.role in ('operator', 'admin')
    )
  );

-- run_events: missing SELECT policy (RLS enabled above but no policy = deny all).
create policy "run_events readable by authenticated users" on public.run_events
  for select to authenticated using (true);

create policy "machine can insert run events" on public.run_events
  for insert with check (machine_id = (auth.jwt() ->> 'sub'));

create policy "machine can update own run events" on public.run_events
  for update using (machine_id = (auth.jwt() ->> 'sub'))
  with check (machine_id = (auth.jwt() ->> 'sub'));

create policy "team activity readable by authenticated users" on public.team_activity_events
  for select to authenticated using (true);

create policy "machine can insert team activity" on public.team_activity_events
  for insert with check (machine_id = (auth.jwt() ->> 'sub'));

create policy "users can read own notification settings" on public.notification_settings
  for select to authenticated using (user_id = auth.uid());

create policy "users can create own notification settings" on public.notification_settings
  for insert to authenticated with check (user_id = auth.uid());

create policy "users can update own notification settings" on public.notification_settings
  for update to authenticated using (user_id = auth.uid())
  with check (user_id = auth.uid());

create policy "users can delete own notification settings" on public.notification_settings
  for delete to authenticated using (user_id = auth.uid());

-- Worker write policies.
-- The remote worker authenticates with machine_token as the bearer JWT.
-- If machine_token is a Supabase service-role key it bypasses RLS entirely;
-- if it is a regular user JWT these policies permit the worker to upsert its own rows.
-- The sub claim of the machine JWT is expected to equal the machine_id string.
create policy "machine can upsert own row" on public.machines
  for all using (machine_id = (auth.jwt() ->> 'sub'))
  with check (machine_id = (auth.jwt() ->> 'sub'));

create policy "machine can upsert own runs" on public.runs
  for all using (machine_id = (auth.jwt() ->> 'sub'))
  with check (machine_id = (auth.jwt() ->> 'sub'));

create policy "machine can upsert own run deletions" on public.run_deletions
  for all using (machine_id = (auth.jwt() ->> 'sub'))
  with check (machine_id = (auth.jwt() ->> 'sub'));

create policy "machine can upsert own artifacts" on public.artifacts
  for all using (machine_id = (auth.jwt() ->> 'sub'))
  with check (machine_id = (auth.jwt() ->> 'sub'));

create policy "authenticated users can read redrhex videos" on storage.objects
  for select to authenticated
  using (bucket_id = 'redrhex-videos');

-- Indexes on FK columns (Postgres does not auto-create these).
create index if not exists idx_runs_machine_id      on public.runs(machine_id);
create index if not exists idx_run_deletions_deleted_at on public.run_deletions(machine_id, deleted_at desc);
create index if not exists idx_run_deletions_log_dir_name on public.run_deletions(machine_id, log_dir_name);
create index if not exists idx_jobs_machine_id      on public.jobs(machine_id);
create index if not exists idx_jobs_status          on public.jobs(status);
create index if not exists idx_artifacts_run_id     on public.artifacts(run_id);
create index if not exists idx_run_events_run_id    on public.run_events(run_id);
create unique index if not exists idx_run_events_event_key_unique on public.run_events(event_key) where event_key is not null;
create index if not exists idx_run_events_recipient_id on public.run_events(recipient_id);
create index if not exists idx_reward_presets_builtin on public.reward_presets(built_in);
create unique index if not exists idx_notification_settings_user_machine on public.notification_settings(user_id, machine_id);
create index if not exists idx_team_activity_created_at on public.team_activity_events(created_at desc);
create index if not exists idx_team_activity_actor_id   on public.team_activity_events(actor_id);
create index if not exists idx_team_activity_machine_id on public.team_activity_events(machine_id);
create index if not exists idx_team_activity_run_id     on public.team_activity_events(run_id);
create index if not exists idx_team_activity_job_id     on public.team_activity_events(job_id);

do $$
begin
  alter publication supabase_realtime add table public.runs;
exception when duplicate_object or undefined_object then null;
end $$;

do $$
begin
  alter publication supabase_realtime add table public.jobs;
exception when duplicate_object or undefined_object then null;
end $$;

do $$
begin
  alter publication supabase_realtime add table public.artifacts;
exception when duplicate_object or undefined_object then null;
end $$;

do $$
begin
  alter publication supabase_realtime add table public.machines;
exception when duplicate_object or undefined_object then null;
end $$;

do $$
begin
  alter publication supabase_realtime add table public.run_deletions;
exception when duplicate_object or undefined_object then null;
end $$;

create or replace view public.team_activity_member_7d as
select
  actor_id,
  coalesce(actor_name, 'Unknown member') as actor_name,
  actor_role,
  count(*) as events,
  sum(points) as points,
  count(*) filter (where category = 'training') as training_events,
  count(*) filter (where outcome = 'completed') as completed_events,
  count(*) filter (where outcome in ('failed', 'interrupted')) as failed_events
from public.team_activity_events
where created_at >= now() - interval '7 days'
group by actor_id, coalesce(actor_name, 'Unknown member'), actor_role;

create or replace view public.team_activity_experiment_7d as
select
  category,
  event_type,
  outcome,
  count(*) as events,
  sum(points) as points
from public.team_activity_events
where created_at >= now() - interval '7 days'
group by category, event_type, outcome;
create index if not exists idx_terrain_presets_builtin on public.terrain_presets(built_in);
