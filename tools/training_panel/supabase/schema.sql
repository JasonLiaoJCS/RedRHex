-- RedRHex Training Panel V2.0 Supabase schema
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
  params jsonb not null default '{}',
  log_dir text,
  latest_checkpoint text,
  latest_video text,
  onnx_path text,
  created_by uuid references auth.users(id),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

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
  created_at timestamptz not null default now(),
  unique (run_id, event_type)
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
  discord_enabled boolean not null default false,
  email_enabled boolean not null default false,
  email_recipients text[] not null default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create or replace function public.claim_next_job_for_machine(p_machine_id text)
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
    order by created_at asc
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

create policy "profiles readable by authenticated users" on public.profiles
  for select to authenticated using (true);

create policy "runs readable by authenticated users" on public.runs
  for select to authenticated using (true);

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
