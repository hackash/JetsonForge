#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_FOLDER="${DOWNLOAD_FOLDER:?DOWNLOAD_FOLDER env var is required}"
JP_VERSION="${JP_VERSION:?JP_VERSION env var is required}"
JETSON_TARGET="${JETSON_TARGET:?JETSON_TARGET env var is required}"
USE_ARCHIVED="${USE_ARCHIVED:-auto}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
TEAMS_WEBHOOK_URL="${TEAMS_WEBHOOK_URL:-}"
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"
AUTH_TIMEOUT="${AUTH_TIMEOUT:-300}"

echo "::group::Downloading JetPack $JP_VERSION"

echo "Downloading JetPack $JP_VERSION for $JETSON_TARGET..."
echo "Download location: $DOWNLOAD_FOLDER"
echo ""

mkdir -p "$DOWNLOAD_FOLDER"

# ------------------------------------------------------------------
# Determine --archived-versions flag
# ------------------------------------------------------------------
ARCHIVED_FLAG=""

if [[ "$USE_ARCHIVED" == "true" ]]; then
  echo "Using --archived-versions flag (explicitly enabled)"
  ARCHIVED_FLAG="--archived-versions"
elif [[ "$USE_ARCHIVED" == "false" ]]; then
  echo "Not using --archived-versions flag (explicitly disabled)"
else
  JP_MAJOR=$(echo "$JP_VERSION" | grep -oP '^[0-9]+' || echo "0")
  if [[ "$JP_MAJOR" -lt 6 ]] 2>/dev/null; then
    echo "Detected older JetPack version ($JP_VERSION < 6.0), using --archived-versions (auto)"
    ARCHIVED_FLAG="--archived-versions"
  else
    echo "Detected current JetPack version ($JP_VERSION >= 6.0), no --archived-versions needed (auto)"
  fi
fi

DL_LOG="$(mktemp)"
DL_STATUS="$(mktemp)"
trap "rm -f '$DL_LOG' '$DL_STATUS'" EXIT

html_escape() {
  python3 -c 'import html, sys; print(html.escape(sys.stdin.read()))'
}

extract_login_url() {
  grep -Eo 'https://static-login\.nvidia\.com/service/default/pin\?user_code=[0-9]+' "$DL_LOG" 2>/dev/null \
    | tail -1 || true
}

extract_qr_block() {
  awk '
    /Scan QR code:/ {
      capture=1
      next
    }

    /SDK Manager will start once login is completed/ {
      exit
    }

    capture {
      print
    }
  ' "$DL_LOG" 2>/dev/null || true
}

auth_screen_ready() {
  grep -q "SDK Manager will start once login is completed" "$DL_LOG" 2>/dev/null
}

send_telegram_auth() {
  local login_url="$1"
  local qr_block="$2"

  if [[ -z "$TELEGRAM_BOT_TOKEN" || -z "$TELEGRAM_CHAT_ID" ]]; then
    echo "[INFO] Telegram credentials not set, skipping Telegram notification"
    return 0
  fi

  local escaped_qr
  escaped_qr="$(printf '%s' "$qr_block" | html_escape)"

  local message
  message="🔐 <b>NVIDIA SDK Manager Authentication Required</b>

Open this link:

${login_url}

Or scan this QR code:

<pre>${escaped_qr}</pre>

SDK Manager will continue automatically after login."

  curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=${message}" \
    --data-urlencode "parse_mode=HTML" \
    --data-urlencode "disable_web_page_preview=false" \
    >/dev/null || true
}

send_slack_auth() {
  local login_url="$1"

  if [[ -z "$SLACK_WEBHOOK_URL" ]]; then
    return 0
  fi

  curl -s -X POST -H 'Content-Type: application/json' \
    --data "{\"text\":\"🔐 NVIDIA SDK Manager Authentication Required: ${login_url}\"}" \
    "$SLACK_WEBHOOK_URL" >/dev/null || true
}

send_teams_auth() {
  local login_url="$1"

  if [[ -z "$TEAMS_WEBHOOK_URL" ]]; then
    return 0
  fi

  curl -s -X POST -H 'Content-Type: application/json' \
    --data "{\"@type\":\"MessageCard\",\"@context\":\"https://schema.org/extensions\",\"themeColor\":\"76B900\",\"title\":\"NVIDIA SDK Manager Authentication Required\",\"text\":\"Authenticate here: ${login_url}\"}" \
    "$TEAMS_WEBHOOK_URL" >/dev/null || true
}

echo ""
echo "Executing SDK Manager download and monitoring authentication output..."

SDKM_CMD=(
  sdkmanager
  --cli
  --action downloadonly
  --download-folder "$DOWNLOAD_FOLDER"
  --login-type devzone
  --exit-on-finish
  --license accept
  --product Jetson
  --version "$JP_VERSION"
  --target-os Linux
  --host
  --check-for-updates false
  --query non-interactive
  --target "$JETSON_TARGET"
)

if [[ -n "$ARCHIVED_FLAG" ]]; then
  SDKM_CMD+=("$ARCHIVED_FLAG")
fi

(
  set +e
  set +o pipefail

  # Important:
  # `script` gives SDK Manager a pseudo-terminal.
  # This helps preserve the QR/login screen exactly as SDK Manager prints it.
  if command -v script >/dev/null 2>&1; then
    CMD_STR="$(printf ' %q' "${SDKM_CMD[@]}")"
    script -q -e -c "$CMD_STR" /dev/null 2>&1 | tee "$DL_LOG"
    echo "${PIPESTATUS[0]}" > "$DL_STATUS"
  else
    "${SDKM_CMD[@]}" 2>&1 | tee "$DL_LOG"
    echo "${PIPESTATUS[0]}" > "$DL_STATUS"
  fi
) &

DL_BG_PID=$!

NOTIFIED=false
ELAPSED=0
MAX_WAIT=7200

while kill -0 "$DL_BG_PID" 2>/dev/null; do
  if [[ "$NOTIFIED" == "false" ]]; then
    LOGIN_URL="$(extract_login_url)"

    if [[ -n "$LOGIN_URL" ]] && auth_screen_ready; then
      QR_BLOCK="$(extract_qr_block)"

      NOTIFIED=true

      echo ""
      echo "=========================================="
      echo "AUTHENTICATION REQUIRED"
      echo "Please open this URL in your browser:"
      echo "  $LOGIN_URL"
      echo ""
      echo "QR code captured from SDK Manager output."
      echo "=========================================="
      echo ""

      send_telegram_auth "$LOGIN_URL" "$QR_BLOCK"
      send_slack_auth "$LOGIN_URL"
      send_teams_auth "$LOGIN_URL"
    fi
  fi

  sleep 5
  ELAPSED=$((ELAPSED + 5))

  if [[ "$ELAPSED" -ge "$MAX_WAIT" ]]; then
    echo "::error::Download timed out after ${MAX_WAIT}s"
    kill "$DL_BG_PID" 2>/dev/null || true
    exit 1
  fi
done

wait "$DL_BG_PID" || true
DL_EXIT="$(cat "$DL_STATUS" 2>/dev/null || echo "1")"

if [[ "$DL_EXIT" != "0" ]]; then
  echo "::error::SDK Manager download failed with exit code $DL_EXIT"
  tail -120 "$DL_LOG" || true
  exit "$DL_EXIT"
fi

# Important safety check:
# SDK Manager may exit 0 in some bad cases, so verify download folder is not empty.
if ! find "$DOWNLOAD_FOLDER" -type f -size +1k | grep -q .; then
  echo "::error::SDK Manager exited with code 0, but no downloaded files were found."
  echo ""
  echo "Last SDK Manager output:"
  tail -120 "$DL_LOG" || true
  exit 1
fi

echo ""
echo "Download complete. Files saved to: $DOWNLOAD_FOLDER"
ls -lh "$DOWNLOAD_FOLDER"

echo "::endgroup::"