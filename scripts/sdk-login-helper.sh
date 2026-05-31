#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SDK Manager Authentication Helper for GitHub Actions
# ============================================================
# This script handles headless authentication for NVIDIA SDK Manager
# and sends notification webhooks to Slack and/or Microsoft Teams.
#
# Environment Variables:
#   SLACK_WEBHOOK_URL  - Optional Slack incoming webhook URL
#   TEAMS_WEBHOOK_URL  - Optional Microsoft Teams webhook URL
#   AUTH_TIMEOUT       - Maximum wait time in seconds (default: 300)
# ============================================================

AUTH_TIMEOUT="${AUTH_TIMEOUT:-300}"
CHECK_INTERVAL=10
NVSDKM_DIR="${HOME}/.nvsdkm"
CONFIG_DIR="${HOME}/.config/sdkmanager"

# Color output for terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_header() {
    echo ""
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD}$*${NC}"
    echo -e "${BOLD}========================================${NC}"
    echo ""
}

# Check if SDK Manager is already authenticated
check_authentication() {
    # Check for session files in typical locations
    if [[ -d "$NVSDKM_DIR" ]] && [[ -n "$(ls -A "$NVSDKM_DIR" 2>/dev/null || true)" ]]; then
        log_info "Found SDK Manager session directory: $NVSDKM_DIR"
        return 0
    fi
    
    if [[ -d "$CONFIG_DIR" ]] && [[ -n "$(ls -A "$CONFIG_DIR" 2>/dev/null || true)" ]]; then
        log_info "Found SDK Manager config directory: $CONFIG_DIR"
        return 0
    fi
    
    # Try a simple query command to verify authentication
    if sdkmanager --query --cli 2>&1 | grep -qi "login\|authentication\|credentials"; then
        log_warning "SDK Manager requires authentication"
        return 1
    fi
    
    log_success "SDK Manager is already authenticated"
    return 0
}

# Extract login URL from SDK Manager output
extract_login_url() {
    local output_file="$1"
    local login_url=""
    
    # Look for the authentication URL pattern
    login_url=$(grep -oP 'https://static\.nvidia\.com/sdk-manager/login\.html\?code=[^[:space:]"]+' "$output_file" | head -n1 || true)
    
    if [[ -z "$login_url" ]]; then
        # Try alternative pattern
        login_url=$(grep -oP 'https://[^[:space:]]*login[^[:space:]]*code=[^[:space:]"]+' "$output_file" | head -n1 || true)
    fi
    
    echo "$login_url"
}

# Send Slack notification
send_slack_notification() {
    local login_url="$1"
    local webhook_url="${SLACK_WEBHOOK_URL:-}"
    
    if [[ -z "$webhook_url" ]]; then
        log_info "SLACK_WEBHOOK_URL not set, skipping Slack notification"
        return 0
    fi
    
    log_info "Sending authentication URL to Slack..."
    
    local payload
    payload=$(cat <<EOF
{
  "blocks": [
    {
      "type": "header",
      "text": {
        "type": "plain_text",
        "text": "🔐 NVIDIA SDK Manager Authentication Required"
      }
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "Your JetsonForge GitHub Action build is waiting for authentication.\n\n*Please click the link below to authenticate:*"
      }
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "<${login_url}|🔗 Authenticate SDK Manager>"
      }
    },
    {
      "type": "context",
      "elements": [
        {
          "type": "mrkdwn",
          "text": "⏰ Timeout: ${AUTH_TIMEOUT}s | 🤖 Action: JetsonForge"
        }
      ]
    }
  ]
}
EOF
)
    
    if curl -X POST -H 'Content-Type: application/json' \
        --data "$payload" \
        --silent --show-error \
        "$webhook_url" > /dev/null 2>&1; then
        log_success "Slack notification sent successfully"
    else
        log_warning "Failed to send Slack notification"
    fi
}

# Send Microsoft Teams notification
send_teams_notification() {
    local login_url="$1"
    local webhook_url="${TEAMS_WEBHOOK_URL:-}"
    
    if [[ -z "$webhook_url" ]]; then
        log_info "TEAMS_WEBHOOK_URL not set, skipping Teams notification"
        return 0
    fi
    
    log_info "Sending authentication URL to Microsoft Teams..."
    
    local payload
    payload=$(cat <<EOF
{
  "@type": "MessageCard",
  "@context": "https://schema.org/extensions",
  "summary": "NVIDIA SDK Manager Authentication Required",
  "themeColor": "76B900",
  "title": "🔐 NVIDIA SDK Manager Authentication Required",
  "sections": [
    {
      "activityTitle": "JetsonForge GitHub Action",
      "activitySubtitle": "Authentication needed to continue build",
      "facts": [
        {
          "name": "Timeout:",
          "value": "${AUTH_TIMEOUT} seconds"
        },
        {
          "name": "Action:",
          "value": "JetsonForge Cross-Compilation"
        }
      ],
      "text": "Your GitHub Action build is waiting for NVIDIA Developer authentication. Please click the button below to authenticate."
    }
  ],
  "potentialAction": [
    {
      "@type": "OpenUri",
      "name": "🔗 Authenticate Now",
      "targets": [
        {
          "os": "default",
          "uri": "${login_url}"
        }
      ]
    }
  ]
}
EOF
)
    
    if curl -X POST -H 'Content-Type: application/json' \
        --data "$payload" \
        --silent --show-error \
        "$webhook_url" > /dev/null 2>&1; then
        log_success "Microsoft Teams notification sent successfully"
    else
        log_warning "Failed to send Teams notification"
    fi
}

# Wait for authentication to complete
wait_for_authentication() {
    local elapsed=0
    local spinner_chars="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    local spinner_idx=0
    
    log_info "Waiting for authentication (timeout: ${AUTH_TIMEOUT}s)..."
    
    while [[ $elapsed -lt $AUTH_TIMEOUT ]]; do
        # Check if authenticated
        if check_authentication; then
            log_success "Authentication completed successfully!"
            return 0
        fi
        
        # Display spinner
        spinner_idx=$(( (spinner_idx + 1) % ${#spinner_chars} ))
        printf "\r${BLUE}[${spinner_chars:$spinner_idx:1}]${NC} Waiting for authentication... (${elapsed}s / ${AUTH_TIMEOUT}s)"
        
        sleep "$CHECK_INTERVAL"
        elapsed=$((elapsed + CHECK_INTERVAL))
    done
    
    echo "" # New line after spinner
    log_error "Authentication timeout reached (${AUTH_TIMEOUT}s)"
    return 1
}

# Main execution
main() {
    log_header "NVIDIA SDK Manager Authentication"
    
    # Check if already authenticated
    if check_authentication; then
        log_success "SDK Manager is already authenticated. No action needed."
        exit 0
    fi
    
    log_info "SDK Manager authentication required"
    log_info "Starting headless login process..."
    
    # Create temporary file for SDK Manager output
    local output_file
    output_file=$(mktemp)
    trap "rm -f '$output_file'" EXIT
    
    # Start SDK Manager login in background and capture output
    log_info "Launching SDK Manager CLI login..."
    (
        sdkmanager --cli --login-type devzone 2>&1 | tee "$output_file" &
        echo $! > "${output_file}.pid"
    ) &
    
    # Give it a moment to start and generate the URL
    sleep 5
    
    # Extract the login URL
    local login_url
    login_url=$(extract_login_url "$output_file")
    
    if [[ -z "$login_url" ]]; then
        log_error "Failed to extract authentication URL from SDK Manager output"
        log_info "SDK Manager output:"
        cat "$output_file"
        exit 1
    fi
    
    # Display the URL prominently
    log_header "AUTHENTICATION REQUIRED"
    echo -e "${BOLD}${GREEN}Please open this URL in your browser:${NC}"
    echo ""
    echo -e "  ${BLUE}${BOLD}${login_url}${NC}"
    echo ""
    echo -e "${YELLOW}This URL has also been logged to the GitHub Actions output.${NC}"
    echo ""
    
    # Send webhook notifications
    send_slack_notification "$login_url"
    send_teams_notification "$login_url"
    
    log_header "WAITING FOR AUTHENTICATION"
    
    # Wait for user to complete authentication
    if wait_for_authentication; then
        log_success "SDK Manager authentication successful!"
        
        # Clean up background process if still running
        if [[ -f "${output_file}.pid" ]]; then
            local pid
            pid=$(cat "${output_file}.pid")
            if ps -p "$pid" > /dev/null 2>&1; then
                kill "$pid" 2>/dev/null || true
            fi
        fi
        
        exit 0
    else
        log_error "Authentication failed or timed out"
        exit 1
    fi
}

# Run main function
main "$@"
