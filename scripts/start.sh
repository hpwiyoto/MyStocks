#!/usr/bin/env bash
# One-command bring-up for the whole stack (MySQL + app + scheduler),
# including recovering from the Codespaces idle-timeout killing the whole
# VM (and with it, dockerd itself -- not just the containers). Meant to be
# run as the bare `start` command in a fresh terminal; see the `start`
# shell function installed into ~/.bashrc by .devcontainer/devcontainer.json
# postStartCommand, which just cd's here and calls this script.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

echo "== MyStocks: starting =="

# --- 1. Docker daemon itself may be down (Codespaces idle-timeout kills
# the whole VM, not just individual processes) -- `docker ps` fails
# instantly if dockerd isn't running, so this check is cheap. ---
if ! docker ps >/dev/null 2>&1; then
    echo "-> dockerd not running, starting it..."
    sudo dockerd >/tmp/dockerd.log 2>&1 &
    for _ in $(seq 1 30); do
        docker ps >/dev/null 2>&1 && break
        sleep 1
    done
    if ! docker ps >/dev/null 2>&1; then
        echo "!! dockerd still not responding after 30s -- check /tmp/dockerd.log"
        exit 1
    fi
    echo "-> dockerd ready"
else
    echo "-> dockerd already running"
fi

# --- 2. Bring up (or recreate, for anything that changed) all services.
# restart:unless-stopped means most of the time this is a no-op confirming
# things are already up, not an actual (re)start. ---
echo "-> docker compose up -d"
docker compose up -d

# --- 3. Wait for MySQL's own healthcheck, not just "container running" --
# the app container can be Up while still failing to connect if MySQL
# hasn't finished initializing yet. ---
echo -n "-> waiting for MySQL to be healthy"
for _ in $(seq 1 60); do
    status="$(docker inspect --format '{{.State.Health.Status}}' mystocks-mysql-1 2>/dev/null || echo "unknown")"
    [ "$status" = "healthy" ] && break
    echo -n "."
    sleep 2
done
echo ""
if [ "$status" != "healthy" ]; then
    echo "!! MySQL didn't report healthy in time (last status: $status) -- check: docker logs mystocks-mysql-1"
fi

# --- 4. Confirm the app is actually answering, not just "container up". ---
echo -n "-> waiting for Streamlit to answer on :8501"
for _ in $(seq 1 30); do
    code="$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8501 -m 5 2>/dev/null || echo "000")"
    [ "$code" = "200" ] && break
    echo -n "."
    sleep 2
done
echo ""

echo ""
docker ps --format '  {{.Names}}: {{.Status}}'
echo ""
if [ "$code" = "200" ]; then
    echo "== Ready (local: http://localhost:8501) =="
    if [ -n "${CODESPACE_NAME:-}" ]; then
        echo "== Codespaces URL: https://${CODESPACE_NAME}-8501.app.github.dev =="
        echo "   (kalau belum bisa dibuka: buka panel PORTS di VS Code/browser, forward port 8501 manual sekali)"
    fi
else
    echo "!! App still not answering (http $code) -- check: docker logs mystocks-app-1"
    exit 1
fi
