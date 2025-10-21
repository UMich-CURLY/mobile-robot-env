#!/usr/bin/env bash
set -euo pipefail

# Installer for Go1 ORB startup at boot
# Usage:
#   ./setup_startup.sh install [username]
#   ./setup_startup.sh uninstall [username]
# If username is omitted, current user is used.

ACTION="${1:-install}"
TARGET_USER="${2:-$(id -un)}"

SERVICE_TEMPLATE_PATH="$(cd "$(dirname "$0")" && pwd)/go1_orb_startup@.service"
SYSTEMD_DIR="/etc/systemd/system"
INSTANCE_NAME="go1_orb_startup@${TARGET_USER}.service"

require_root() {
  if [[ $(id -u) -ne 0 ]]; then
    echo "This action requires root. Re-run with: sudo $0 $ACTION $TARGET_USER"
    exit 1;
  fi
}

validate() {
  if [[ ! -f "$SERVICE_TEMPLATE_PATH" ]]; then
    echo "Service file not found: $SERVICE_TEMPLATE_PATH"
    exit 1
  fi
  if [[ ! -f "/home/${TARGET_USER}/mobile-robot-env/unitree_go1_deploy/run_nuc_orb.sh" ]]; then
    echo "Missing run script at /home/${TARGET_USER}/mobile-robot-env/unitree_go1_deploy/run_nuc_orb.sh"
    exit 1
  fi
}

install_service() {
  require_root
  validate

  echo "Installing systemd service for user: ${TARGET_USER}"
  install -m 0644 "$SERVICE_TEMPLATE_PATH" "${SYSTEMD_DIR}/go1_orb_startup@.service"
  systemctl daemon-reload
  systemctl enable --now "$INSTANCE_NAME"
  systemctl is-enabled "$INSTANCE_NAME" && echo "Enabled: $INSTANCE_NAME"
  systemctl status "$INSTANCE_NAME" --no-pager -l || true
}

uninstall_service() {
  require_root

  echo "Disabling and removing service for user: ${TARGET_USER}"
  systemctl disable --now "$INSTANCE_NAME" || true
  systemctl daemon-reload
  # Only remove the template if no other instances exist
  if ! systemctl list-units --all | grep -q "go1_orb_startup@"; then
    rm -f "${SYSTEMD_DIR}/go1_orb_startup@.service"
    systemctl daemon-reload
  fi
  echo "Done."
}

case "$ACTION" in
  install)
    install_service
    ;;
  uninstall)
    uninstall_service
    ;;
  *)
    echo "Unknown action: $ACTION"
    echo "Usage: $0 {install|uninstall} [username]"
    exit 1
    ;;
esac