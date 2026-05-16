#!/usr/bin/env bash
set -euo pipefail

PROJECT_REF="${PROJECT_REF:-tqvopodmsprhujyagaan}"
MACHINE_ID="${MACHINE_ID:-biorolapc2-ubuntu}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PANEL_ROOT="$REPO_ROOT/tools/training_panel"

echo "RedRHex notify Edge Function deploy"
echo "Project ref: $PROJECT_REF"
echo "Panel root:  $PANEL_ROOT"
echo

if ! command -v npx >/dev/null 2>&1; then
  echo "npx is required. Install Node/npm first."
  exit 1
fi

if [[ -z "${SUPABASE_ACCESS_TOKEN:-}" ]]; then
  echo "SUPABASE_ACCESS_TOKEN is not set."
  echo "Create one at https://supabase.com/dashboard/account/tokens"
  if [[ ! -t 0 ]]; then
    echo "Then run: export SUPABASE_ACCESS_TOKEN=\"paste-token-here\""
    echo "And rerun: tools/training_panel/supabase/deploy_notify.sh"
    exit 1
  fi
  read -r -s -p "Paste Supabase access token: " SUPABASE_ACCESS_TOKEN
  export SUPABASE_ACCESS_TOKEN
  echo
fi

echo "Checking Supabase access..."
npx --yes supabase projects list >/dev/null
echo "Supabase access OK."
echo

read -r -s -p "Resend API key (leave blank to keep existing secret): " RESEND_KEY
echo
read -r -p "Verified sender, e.g. RedRHex Training <training@yourdomain.com> (blank to keep existing): " EMAIL_FROM

SECRET_ARGS=()
if [[ -n "$RESEND_KEY" ]]; then
  SECRET_ARGS+=("REDRHEX_RESEND_API_KEY=$RESEND_KEY")
fi
if [[ -n "$EMAIL_FROM" ]]; then
  SECRET_ARGS+=("REDRHEX_NOTIFICATION_EMAIL_FROM=$EMAIL_FROM")
fi

read -r -s -p "Machine token (blank if worker uses service-role key or secret already exists): " MACHINE_TOKEN
echo
if [[ -n "$MACHINE_TOKEN" ]]; then
  SECRET_ARGS+=("REDRHEX_SUPABASE_MACHINE_TOKEN=$MACHINE_TOKEN")
fi

if [[ ${#SECRET_ARGS[@]} -gt 0 ]]; then
  echo "Setting Edge Function secrets..."
  npx --yes supabase secrets set "${SECRET_ARGS[@]}" --project-ref "$PROJECT_REF"
else
  echo "No secrets changed."
fi
echo

echo "Deploying notify function..."
npx --yes supabase functions deploy notify \
  --project-ref "$PROJECT_REF" \
  --workdir "$PANEL_ROOT" \
  --use-api \
  --no-verify-jwt
echo

echo "Verifying function endpoint..."
VERIFY_OUTPUT="$(
  curl -i -sS -X POST \
    "https://${PROJECT_REF}.supabase.co/functions/v1/notify" \
    -H "Content-Type: application/json" \
    --data "{\"event_type\":\"test_notification\",\"machine_id\":\"${MACHINE_ID}\"}"
)"
printf '%s\n' "$VERIFY_OUTPUT" | sed -n '1,40p'

if printf '%s' "$VERIFY_OUTPUT" | grep -q "Requested function was not found"; then
  echo
  echo "Deploy verification failed: function still returns 404."
  exit 1
fi

echo
echo "Done. A 401 saying 'Sign in before sending a test notification' is expected for this curl check."
echo "Now open the child Connection page, enable Email, Save Notifications, then Send Test."
