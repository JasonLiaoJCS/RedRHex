-- RedRHex Training Panel V2.1 Supabase schema
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
  updated_at timestamptz not null default now()
);

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
  email_sent_at timestamptz,
  created_at timestamptz not null default now()
  -- No unique constraint on (run_id, event_type): a run may complete multiple times
  -- after resume, so multiple events of the same type must be allowed.
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

-- notification_settings: reserved for per-machine notification preferences.
-- TODO: the notify Edge Function currently reads Discord/email config from env vars;
--       wire it to query this table so preferences can be managed per machine.
create table if not exists public.notification_settings (
  id uuid primary key default gen_random_uuid(),
  machine_id text references public.machines(machine_id),
  discord_enabled boolean not null default false,
  email_enabled boolean not null default false,
  email_recipients text[] not null default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

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

drop trigger if exists set_machines_updated_at on public.machines;
create trigger set_machines_updated_at
  before update on public.machines
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
      and (not p_gpu_locked or type = 'stop_process')
    order by
      case when type = 'stop_process' then 0 else 1 end,
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
alter table public.jobs enable row level security;
alter table public.run_events enable row level security;
alter table public.artifacts enable row level security;
alter table public.proxy_sessions enable row level security;
alter table public.notification_settings enable row level security;
alter table public.reward_presets enable row level security;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('redrhex-videos', 'redrhex-videos', false, 1073741824, array['video/mp4'])
on conflict (id) do update
set public = false,
    file_size_limit = excluded.file_size_limit,
    allowed_mime_types = excluded.allowed_mime_types;

drop policy if exists "profiles readable by authenticated users" on public.profiles;
drop policy if exists "runs readable by authenticated users" on public.runs;
drop policy if exists "operators can update remote run metadata" on public.runs;
drop policy if exists "artifacts readable by authenticated users" on public.artifacts;
drop policy if exists "machines readable by authenticated users" on public.machines;
drop policy if exists "jobs readable by authenticated users" on public.jobs;
drop policy if exists "operators can create jobs" on public.jobs;
drop policy if exists "reward presets readable by authenticated users" on public.reward_presets;
drop policy if exists "operators can create reward presets" on public.reward_presets;
drop policy if exists "operators can update custom reward presets" on public.reward_presets;
drop policy if exists "run_events readable by authenticated users" on public.run_events;
drop policy if exists "machine can upsert own row" on public.machines;
drop policy if exists "machine can upsert own runs" on public.runs;
drop policy if exists "machine can upsert own artifacts" on public.artifacts;
drop policy if exists "authenticated users can read redrhex videos" on storage.objects;

create policy "profiles readable by authenticated users" on public.profiles
  for select to authenticated using (true);

create policy "runs readable by authenticated users" on public.runs
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

-- run_events: missing SELECT policy (RLS enabled above but no policy = deny all).
create policy "run_events readable by authenticated users" on public.run_events
  for select to authenticated using (true);

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

create policy "machine can upsert own artifacts" on public.artifacts
  for all using (machine_id = (auth.jwt() ->> 'sub'))
  with check (machine_id = (auth.jwt() ->> 'sub'));

create policy "authenticated users can read redrhex videos" on storage.objects
  for select to authenticated
  using (bucket_id = 'redrhex-videos');

-- Indexes on FK columns (Postgres does not auto-create these).
create index if not exists idx_runs_machine_id      on public.runs(machine_id);
create index if not exists idx_jobs_machine_id      on public.jobs(machine_id);
create index if not exists idx_jobs_status          on public.jobs(status);
create index if not exists idx_artifacts_run_id     on public.artifacts(run_id);
create index if not exists idx_run_events_run_id    on public.run_events(run_id);
create index if not exists idx_reward_presets_builtin on public.reward_presets(built_in);
