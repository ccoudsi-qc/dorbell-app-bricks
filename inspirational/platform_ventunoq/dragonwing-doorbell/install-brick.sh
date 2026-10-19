#!/usr/bin/env bash
#
# install-brick.sh — register the app-local arduino:face_recognition brick into
# App Lab's STOCK data dir, so the app runs like any stock example: the App Lab
# UI's Stop/Run works and the kiosk auto-opens the webview on the board screen.
#
# WHY THIS EXISTS (vs run_applab.sh)
# ----------------------------------
# run_applab.sh launches the app from a private data dir WITHOUT touching stock —
# great for isolation, but the App Lab UI and the on-screen kiosk never see the
# app's custom brick, so the UI "Run" button fails with:
#     [ERROR] Brick "arduino:face_recognition" Not Found
# and the browser is never auto-opened.
#
# The UI, the `arduino-app-cli` daemon (port 8800) and the kiosk all read the
# STOCK data dir (/var/lib/arduino-app-cli). There is no CLI flag or env var to
# register an extra brick with the daemon — so to get UI parity the brick must be
# ADDED to the stock registry. This script does that ADDITIVELY and REVERSIBLY:
# it backs up the one stock file it appends to, never modifies a stock brick, and
# prints the exact uninstall commands. Verified on-board 2026-10-19.
#
# WHAT IT DOES
#   1. Runs the SAME provisioner the daemon uses (arduino-bricks-list-modules in
#      the python-apps-base image) against the app, to generate the brick's
#      registry entry + compose — no hand-authored YAML.
#   2. Backs up the stock bricks-list.yaml.
#   3. Appends ONLY the arduino:face_recognition entry to the stock registry.
#   4. Copies ONLY compose/arduino/face_recognition/ into the stock compose tree.
#   5. Restarts the daemon so it reloads the registry (the step that's easy to
#      miss: a long-running daemon caches the brick list in memory at startup).
#
# USAGE
#   bash install-brick.sh            # install (idempotent; re-run is a no-op)
#   bash install-brick.sh --uninstall  # restore stock from the latest backup
#
# ENV OVERRIDES
#   APP_DIR     app folder (default: this script's dir / the dragonwing-doorbell app)
#   STOCK_DIR   stock data dir  (default: /var/lib/arduino-app-cli)
#   BASE_IMAGE  provisioner img (default: ghcr.io/arduino/app-bricks/python-apps-base:0.12.0)
#   BRICK_ID    brick to install (default: arduino:face_recognition)
#   BRICK_DIR   compose subdir name (default: face_recognition)
set -euo pipefail

BRICK_ID="${BRICK_ID:-arduino:face_recognition}"
BRICK_DIR="${BRICK_DIR:-face_recognition}"
STOCK_DIR="${STOCK_DIR:-/var/lib/arduino-app-cli}"
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/arduino/app-bricks/python-apps-base:0.12.0}"
BACKUP_ROOT="${BACKUP_ROOT:-$HOME/.doorbell-stock-backup}"

# Default APP_DIR: prefer an app.yaml next to this script; else the repo app path.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SELF_DIR/app.yaml" ]; then
    APP_DIR="${APP_DIR:-$SELF_DIR}"
else
    APP_DIR="${APP_DIR:-$SELF_DIR/inspirational/platform_ventunoq/dragonwing-doorbell}"
fi

log()  { printf '\033[1;36m[install-brick]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[install-brick] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Restart the daemon so it reloads the registry. THIS STEP IS REQUIRED: the
# long-running daemon caches the brick list in memory at startup, so a disk edit
# alone leaves the UI reporting the old "Brick not found" (verified on-board
# 2026-10-19). Try without sudo first; fall back to sudo (prompts in a terminal).
# If it can't restart automatically (e.g. run non-interactively), don't fail the
# whole install — the files are already in place — just tell the user the one
# command to finish.
restart_daemon() {
    log "restarting arduino-app-cli daemon (reloads the brick registry — required)"
    if systemctl restart arduino-app-cli.service 2>/dev/null \
       || sudo systemctl restart arduino-app-cli.service; then
        sleep 4
        [ "$(systemctl is-active arduino-app-cli.service)" = active ] \
            || die "daemon did not come back active after restart"
        log "daemon active — registry reloaded"
    else
        printf '\033[1;33m[install-brick] NOTE:\033[0m could not restart the daemon automatically.\n' >&2
        printf '  The brick files are installed, but the App Lab UI will keep showing the OLD\n' >&2
        printf '  registry until you restart the daemon yourself:\n\n' >&2
        printf '    sudo systemctl restart arduino-app-cli.service\n\n' >&2
    fi
}

VER="$(ls -1 "$STOCK_DIR/assets" | head -1)"
[ -n "$VER" ] || die "could not determine assets version under $STOCK_DIR/assets"
STOCK_ASSETS="$STOCK_DIR/assets/$VER"
REG="$STOCK_ASSETS/bricks-list.yaml"

# ── --uninstall ───────────────────────────────────────────────────────────────
if [ "${1:-}" = "--uninstall" ]; then
    latest="$(ls -1dt "$BACKUP_ROOT"/*/ 2>/dev/null | head -1 || true)"
    [ -n "$latest" ] && [ -f "$latest/bricks-list.yaml.orig" ] \
        || die "no backup under $BACKUP_ROOT to restore from"
    log "restoring stock registry from $latest"
    cp -a "$latest/bricks-list.yaml.orig" "$REG"
    rm -rf "$STOCK_ASSETS/compose/arduino/$BRICK_DIR"
    restart_daemon
    log "uninstalled $BRICK_ID from stock (removed compose/arduino/$BRICK_DIR)"
    exit 0
fi

# ── preflight ────────────────────────────────────────────────────────────────
[ -f "$APP_DIR/app.yaml" ] || die "no app.yaml under APP_DIR=$APP_DIR"
[ -f "$REG" ] || die "stock registry not found at $REG (is App Lab installed?)"
command -v docker >/dev/null || die "docker not on PATH"

# ── idempotency guard ──────────────────────────────────────────────────────────
if grep -q "$BRICK_ID" "$REG" 2>/dev/null; then
    log "$BRICK_ID already present in the stock registry — nothing to install."
    log "(re-run with --uninstall to remove it, or restart the daemon if the UI is stale.)"
    restart_daemon
    exit 0
fi

# ── 1. provision the brick's registry entry + compose (same scanner the daemon uses) ──
log "scanning app for bricks (provisioner: $BASE_IMAGE)"
tmp_out="$(mktemp -d)"; chmod 777 "$tmp_out"
trap 'rm -rf "$tmp_out"' EXIT

docker run --rm --entrypoint arduino-bricks-list-modules \
    -v "$APP_DIR":/app -v "$tmp_out":/out \
    "$BASE_IMAGE" -o /out/bricks-list.yaml >/dev/null
grep -q "$BRICK_ID" "$tmp_out/bricks-list.yaml" \
    || die "scanner did not discover $BRICK_ID — check the brick's import path/brick_config.yaml"

mkdir -p "$tmp_out/prov"; chmod 777 "$tmp_out/prov"
docker run --rm --entrypoint arduino-bricks-list-modules \
    -v "$APP_DIR":/app -v "$tmp_out/prov":/out \
    "$BASE_IMAGE" --provision-compose -o /out >/dev/null
[ -d "$tmp_out/prov/compose/arduino/$BRICK_DIR" ] \
    || die "provisioner did not emit compose/arduino/$BRICK_DIR"

# ── 2. back up the one stock file we touch ──────────────────────────────────────
stamp="$(date +%Y%m%d-%H%M%S)"
bk="$BACKUP_ROOT/$stamp"; mkdir -p "$bk"
cp -a "$REG" "$bk/bricks-list.yaml.orig"
log "backed up stock registry → $bk/bricks-list.yaml.orig"

# ── 3. append ONLY our brick's entry (extract it from the generated full list) ──
#     From the brick's '- id:' line to just before the next '- id:' (or EOF).
awk -v id="- id: $BRICK_ID" '
    $0==id {f=1}
    f && /^- id:/ && $0!=id {exit}
    f {print}' "$tmp_out/bricks-list.yaml" > "$tmp_out/brick-block.yaml"
[ -s "$tmp_out/brick-block.yaml" ] || die "failed to extract the $BRICK_ID block"

# ensure stock ends in a newline, then append
[ -n "$(tail -c1 "$REG")" ] && printf '\n' >> "$REG"
cat "$tmp_out/brick-block.yaml" >> "$REG"
log "appended $BRICK_ID → stock registry now $(grep -c '^- id:' "$REG") bricks"

# ── 4. copy ONLY our compose subdir (additive; new dir) ─────────────────────────
cp -a "$tmp_out/prov/compose/arduino/$BRICK_DIR" "$STOCK_ASSETS/compose/arduino/$BRICK_DIR"
log "installed compose/arduino/$BRICK_DIR ($(ls "$STOCK_ASSETS/compose/arduino/$BRICK_DIR" | tr '\n' ' '))"

rm -rf "$tmp_out"; trap - EXIT

# ── 5. reload the daemon ────────────────────────────────────────────────────────
restart_daemon

cat <<EOF

$(printf '\033[1;32m')✓ $BRICK_ID installed into App Lab.$(printf '\033[0m')
  The App Lab UI's Stop/Run now works for this app and the kiosk auto-opens the webview.

  To uninstall (restore pristine stock):
    bash $(basename "${BASH_SOURCE[0]}") --uninstall
  or manually:
    cp $bk/bricks-list.yaml.orig $REG
    rm -rf $STOCK_ASSETS/compose/arduino/$BRICK_DIR
    sudo systemctl restart arduino-app-cli.service
EOF
