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
echo "ℹ️  For version compatibility, see: https://github.com/hackash/JetsonForge/blob/main/JETPACK-VERSIONS.md"
echo ""

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

echo ""
echo "Executing SDK Manager download (monitoring output for authentication prompts)..."

# ------------------------------------------------------------------
# Run sdkmanager in background, tee output to a log file so we can
# detect a login URL and send notifications while the process runs.
# set +e / +o pipefail inside the subshell ensures we always capture
# the real exit code regardless of sdkmanager's exit status.
# ------------------------------------------------------------------
DL_LOG=$(mktemp)
DL_STATUS=$(mktemp)
trap "rm -f '$DL_LOG' '$DL_STATUS'" EXIT

(
  set +e
  set +o pipefail
  sdkmanager --cli \
    --action downloadonly \
    --download-folder "$DOWNLOAD_FOLDER" \
    --login-type devzone \
    --exit-on-finish \
    --license accept \
    --product Jetson \
    --version "$JP_VERSION" \
    --target-os Linux \
    --host \
    --check-for-updates false \
    --query non-interactive \
    --target "$JETSON_TARGET" \
    $ARCHIVED_FLAG 2>&1 | tee "$DL_LOG"
  echo "${PIPESTATUS[0]}" > "$DL_STATUS"
) &
DL_BG_PID=$!

NOTIFIED=false
ELAPSED=0
MAX_WAIT=7200   # 2-hour ceiling for large JetPack downloads

while kill -0 "$DL_BG_PID" 2>/dev/null; do
  if [[ "$NOTIFIED" == "false" ]]; then
    LOGIN_URL=$(grep -oP \
      'https://static\.nvidia\.com/sdk-manager/login\.html\?code=[^[:space:]"]+' \
      "$DL_LOG" 2>/dev/null | head -1 || true)
    # Fallback pattern in case the URL format changes
    [[ -z "$LOGIN_URL" ]] && LOGIN_URL=$(grep -oP \
      'https://[^[:space:]]*login[^[:space:]]*\?code=[^[:space:]"]+' \
      "$DL_LOG" 2>/dev/null | head -1 || true)

    if [[ -n "$LOGIN_URL" ]]; then
      NOTIFIED=true
      echo ""
      echo "=========================================="
      echo "AUTHENTICATION REQUIRED"
      echo "Please open this URL in your browser:"
      echo "  $LOGIN_URL"
      echo "=========================================="

      # Telegram
      if [[ -n "$TELEGRAM_BOT_TOKEN" ]] && [[ -n "$TELEGRAM_CHAT_ID" ]]; then
        TGRAM_MSG="🔐 NVIDIA SDK Manager authentication required for JetsonForge download.

Please authenticate to continue:
${LOGIN_URL}

Timeout: ${AUTH_TIMEOUT}s"
        ENCODED=$(echo -n "$TGRAM_MSG" | jq -sRr @uri)
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
          -d "chat_id=${TELEGRAM_CHAT_ID}&text=${ENCODED}&disable_web_page_preview=false" \
          > /dev/null || true
      fi

      # Slack
      if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        curl -s -X POST -H 'Content-Type: application/json' \
          --data "{\"blocks\":[{\"type\":\"section\",\"text\":{\"type\":\"mrkdwn\",\"text\":\"🔐 *NVIDIA SDK Manager Authentication Required*\nJetsonForge download is waiting. <${LOGIN_URL}|🔗 Authenticate here>\"}}]}" \
          "$SLACK_WEBHOOK_URL" > /dev/null || true
      fi

      # Teams
      if [[ -n "$TEAMS_WEBHOOK_URL" ]]; then
        curl -s -X POST -H 'Content-Type: application/json' \
          --data "{\"@type\":\"MessageCard\",\"@context\":\"https://schema.org/extensions\",\"themeColor\":\"76B900\",\"title\":\"🔐 NVIDIA SDK Manager Authentication Required\",\"text\":\"JetsonForge download is waiting. [Authenticate here](${LOGIN_URL})\"}" \
          "$TEAMS_WEBHOOK_URL" > /dev/null || true
      fi
    fi
  fi

  sleep 10
  ELAPSED=$((ELAPSED + 10))
  if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    echo "::error::Download timed out after ${MAX_WAIT}s"
    kill "$DL_BG_PID" 2>/dev/null || true
    exit 1
  fi
done

wait "$DL_BG_PID" || true
DL_EXIT=$(cat "$DL_STATUS" 2>/dev/null || echo "1")

if [[ "$DL_EXIT" != "0" ]]; then
  echo "::error::SDK Manager download failed with exit code $DL_EXIT"
  exit "$DL_EXIT"
fi

echo ""
echo "Download complete. Files saved to: $DOWNLOAD_FOLDER"
ls -lh "$DOWNLOAD_FOLDER"

echo "::endgroup::"
