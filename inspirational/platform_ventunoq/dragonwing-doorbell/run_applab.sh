#!/usr/bin/env bash
#
# run_applab.sh — launch the Dragonwing Doorbell App Lab app on a Ventuno Q.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# `arduino-app-cli app start <path>` validates every brick in app.yaml against a
# STATIC registry at <data-dir>/assets/<ver>/bricks-list.yaml. Our custom
# `arduino:face_recognition` brick lives INSIDE the app
# (python/arduino/app_bricks/face_recognition/) and is therefore absent from the
# stock registry — so a plain `app start` fails with:
#
#     [ERROR] Brick "arduino:face_recognition" Not Found
#
# The fix (proven on-board 2026-10-18) is to point the CLI at a PRIVATE data dir
# via ARDUINO_APP_CLI__DATA_DIR whose registry = stock + our brick, WITHOUT
# editing a single stock file (the Ventuno Q isolation constraint). This script
# builds that private data dir reproducibly and launches the app.
#
# It never hand-edits YAML: the SAME provisioner the daemon uses
# (`arduino-bricks-list-modules`, shipped in the python-apps-base image) scans
# the app's import path, discovers the app-local brick, and regenerates BOTH:
#   * bricks-list.yaml  — all stock bricks + arduino:face_recognition  (plain mode)
#   * compose/arduino/* — every brick's compose incl. our runner  (--provision-compose)
# Everything else in the data dir is a symlink to the untouched stock tree.
#
# USAGE
#   bash run_applab.sh                 # build private data dir + app start
#   bash run_applab.sh --rebuild       # force-rebuild the data dir, then start
#   bash run_applab.sh --setup-only    # build the data dir, do NOT start the app
#   bash run_applab.sh --stop          # stop the running app (via the private dir)
#
# ENV OVERRIDES
#   APP_DIR     app folder (default: this script's dir)
#   DATA_DIR    private data dir   (default: ~/.doorbell-datadir)
#   STOCK_DIR   stock data dir     (default: /var/lib/arduino-app-cli)
#   BASE_IMAGE  provisioner image  (default: ghcr.io/arduino/app-bricks/python-apps-base:0.12.0)
#
set -euo pipefail

APP_DIR="${APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_DIR="${DATA_DIR:-$HOME/.doorbell-datadir}"
STOCK_DIR="${STOCK_DIR:-/var/lib/arduino-app-cli}"
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/arduino/app-bricks/python-apps-base:0.12.0}"

log() { printf '\033[1;36m[run_applab]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[run_applab] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ── preflight ────────────────────────────────────────────────────────────────
[ -f "$APP_DIR/app.yaml" ] || die "no app.yaml under APP_DIR=$APP_DIR"
[ -d "$APP_DIR/python/arduino/app_bricks/face_recognition" ] \
    || die "face_recognition brick not found under $APP_DIR/python/arduino/app_bricks/"
[ -d "$STOCK_DIR/assets" ] || die "stock data dir not found at $STOCK_DIR (is arduino-app-cli installed?)"
command -v arduino-app-cli >/dev/null || die "arduino-app-cli not on PATH"
command -v docker          >/dev/null || die "docker not on PATH"

# Assets are versioned (e.g. 0.12.0). Discover the version from the stock tree so
# this script keeps working across base-image bumps.
VER="$(ls -1 "$STOCK_DIR/assets" | head -1)"
[ -n "$VER" ] || die "could not determine assets version under $STOCK_DIR/assets"
STOCK_ASSETS="$STOCK_DIR/assets/$VER"
PRIV_ASSETS="$DATA_DIR/assets/$VER"

# ── --stop ──────────────────────────────────────────────────────────────────
if [ "${1:-}" = "--stop" ]; then
    log "stopping app via private data dir"
    ARDUINO_APP_CLI__DATA_DIR="$DATA_DIR" arduino-app-cli app stop "$APP_DIR" || true
    exit 0
fi

REBUILD=0; SETUP_ONLY=0
for a in "$@"; do
    case "$a" in
        --rebuild)    REBUILD=1 ;;
        --setup-only) SETUP_ONLY=1 ;;
        *) die "unknown argument: $a (use --rebuild | --setup-only | --stop)" ;;
    esac
done

# ── build the private data dir ────────────────────────────────────────────────
# Rebuild if asked, or if the registry is missing / lacks our brick.
need_build=1
if [ "$REBUILD" = 0 ] && [ -f "$PRIV_ASSETS/bricks-list.yaml" ] \
   && grep -q "arduino:face_recognition" "$PRIV_ASSETS/bricks-list.yaml" 2>/dev/null; then
    need_build=0
    log "private data dir already provisioned at $DATA_DIR (use --rebuild to refresh)"
fi

if [ "$need_build" = 1 ]; then
    log "building private data dir at $DATA_DIR (stock=$STOCK_DIR, ver=$VER)"
    rm -rf "$DATA_DIR"
    mkdir -p "$PRIV_ASSETS"

    # 1. Top level: symlink every stock entry EXCEPT `assets` (we build our own).
    for entry in "$STOCK_DIR"/* "$STOCK_DIR"/.[!.]*; do
        [ -e "$entry" ] || continue
        name="$(basename "$entry")"
        [ "$name" = "assets" ] && continue
        ln -s "$entry" "$DATA_DIR/$name"
    done

    # 2. assets/<ver>: symlink every stock entry EXCEPT bricks-list.yaml + compose
    #    (both regenerated below from the live provisioner so the app-local brick
    #    is included).
    for entry in "$STOCK_ASSETS"/*; do
        [ -e "$entry" ] || continue
        name="$(basename "$entry")"
        case "$name" in
            bricks-list.yaml|compose) continue ;;
        esac
        ln -s "$entry" "$PRIV_ASSETS/$name"
    done

    # 3. Regenerate the registry + compose tree with the SAME scanner the daemon
    #    uses. The app is mounted at /app so /app/python is on the import path and
    #    the app-local brick is discovered. Container runs as the invoking uid, so
    #    output is owned by us (no sudo, no root-owned files).
    log "scanning app for bricks (provisioner: $BASE_IMAGE)"
    tmp_out="$(mktemp -d)"; chmod 777 "$tmp_out"
    trap 'rm -rf "$tmp_out"' EXIT

    # 3a. bricks-list.yaml — plain mode emits the COMPLETE list (stock + FR).
    # --model-output "" disables arduino-bricks-list-modules' default copy to
    # /app/.cache/models-list.yaml: that dir is never created under the
    # bind-mounted $APP_DIR, so the tool crashes with FileNotFoundError. We
    # don't need the models file here — only bricks-list.yaml and compose/.
    docker run --rm --entrypoint arduino-bricks-list-modules \
        -v "$APP_DIR":/app -v "$tmp_out":/out \
        "$BASE_IMAGE" --model-output "" -o /out/bricks-list.yaml >/dev/null
    grep -q "arduino:face_recognition" "$tmp_out/bricks-list.yaml" \
        || die "scanner did not discover arduino:face_recognition — check the brick's import path/brick_config.yaml"
    cp "$tmp_out/bricks-list.yaml" "$PRIV_ASSETS/bricks-list.yaml"

    # 3b. compose/ — --provision-compose emits every brick's compose incl. our
    #     runner, so the daemon merges face-recognition-runner into app-compose.yaml.
    rm -rf "$tmp_out/prov"; mkdir -p "$tmp_out/prov"; chmod 777 "$tmp_out/prov"
    docker run --rm --entrypoint arduino-bricks-list-modules \
        -v "$APP_DIR":/app -v "$tmp_out/prov":/out \
        "$BASE_IMAGE" --provision-compose -o /out >/dev/null
    [ -f "$tmp_out/prov/compose/arduino/face_recognition/brick_compose.ventunoq.yaml" ] \
        || die "provisioner did not emit the face_recognition ventunoq compose"
    cp -r "$tmp_out/prov/compose" "$PRIV_ASSETS/compose"

    rm -rf "$tmp_out"; trap - EXIT
    log "private registry ready: $(grep -c '^- id:' "$PRIV_ASSETS/bricks-list.yaml") bricks (incl. arduino:face_recognition)"
fi

if [ "$SETUP_ONLY" = 1 ]; then
    log "--setup-only: data dir ready, not starting the app"
    log "to start manually:  ARDUINO_APP_CLI__DATA_DIR=$DATA_DIR arduino-app-cli app start $APP_DIR"
    exit 0
fi

# ── launch ────────────────────────────────────────────────────────────────────
# NOTE: `app start` is state-changing — it stops any running app and rebuilds the
# compose. The runner image (face-recognition-runner:0.1.0) must already be built
# (see the repo-root runner/README.md) or the compose pull fails.
log "starting app  (ARDUINO_APP_CLI__DATA_DIR=$DATA_DIR)"
log "webview will be at  http://<board>:7000  (App Lab opens it automatically)"
exec env ARDUINO_APP_CLI__DATA_DIR="$DATA_DIR" arduino-app-cli app start "$APP_DIR"
