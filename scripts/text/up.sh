#!/usr/bin/env bash
# up.sh [--no-router]: the agent server for text mode, with no speech server, speaker, Pi or microphone.
#   router (llama-server, the fine-tuned model) + agent server, both on loopback with a per-checkout session token.
#   --no-router: Gemini answers every turn (no GGUF needed; slower, and the router's fast lookups are off).
# Ports (18xxx, so the voice stack's can run beside it): TEXT_PORT_AGENT=18781 TEXT_PORT_ROUTER=18089
# Variables: ROUTER_MODEL ROUTER_THREADS GGUF_TOOLS AGENT_ARGS   State and logs: out/text/ (token, delegate jobs, logs, pids)
# The agent needs a Gemini key (~/.secrets/google-ai-studio/api-key.txt, read by the code, never printed).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
STATE=$REPO/out/text
PORT_AGENT=${TEXT_PORT_AGENT:-18781}
PORT_ROUTER=${TEXT_PORT_ROUTER:-18089}
ROUTER_MODEL=${ROUTER_MODEL:-router-350m-kd-web4-s0-unverified}
ROUTER_THREADS=${ROUTER_THREADS:-3}
GGUF_TOOLS=${GGUF_TOOLS:-$HOME/work/gguf-tools}
ROUTER=1; [ "${1:-}" = --no-router ] && ROUTER=0
die() { echo "text-up: $*" >&2; exit 1; }
listening() { ss -ltn 2>/dev/null | grep -q ":$1 "; }
# the venue: the University of the Philippines Cebu (a map pin; no IP lookup). Pin another with VOICE_AGENT_HOME_COORDS="lat,lon".
export VOICE_AGENT_HOME_COORDS=${VOICE_AGENT_HOME_COORDS:-10.3225319,123.8988885}
mkdir -p "$STATE/logs" "$STATE/pids" "$STATE/delegate"
cd "$REPO"
listening "$PORT_AGENT" && die "port $PORT_AGENT is in use: make text-down first"
[ "$ROUTER" = 1 ] && listening "$PORT_ROUTER" && die "port $PORT_ROUTER is in use: make text-down first"
# a session token for this checkout: the router, the agent server and the client share it (mode 0600)
[ -s "$STATE/token" ] || (umask 077; head -c 24 /dev/urandom | base64 | tr -d '/+=\n' >"$STATE/token")
LAN_SPEECH_TOKEN=$(cat "$STATE/token"); export LAN_SPEECH_TOKEN

args=(--ipv4 --host 127.0.0.1 --port "$PORT_AGENT" --pi-url "" --delegate-state-dir "$STATE/delegate")
if [ "$ROUTER" = 1 ]; then
  model=$GGUF_TOOLS/models/$ROUTER_MODEL/gguf/router-Q4_K_M.gguf
  [ -f "$model" ] || die "router model not found: $model (ROUTER_MODEL=$ROUTER_MODEL); make text-up NO_ROUTER=1 runs without it"
  LD_LIBRARY_PATH=$GGUF_TOOLS/llama.cpp/build/bin setsid "$GGUF_TOOLS/llama.cpp/build/bin/llama-server" --jinja -c 8192 -np 1 -fa on \
    -t "$ROUTER_THREADS" -tb "$ROUTER_THREADS" -m "$model" --host 127.0.0.1 --port "$PORT_ROUTER" \
    --api-key-file "$STATE/token" >"$STATE/logs/router.log" 2>&1 </dev/null &
  echo $! >"$STATE/pids/router"
  for _ in $(seq 60); do curl -sf "http://127.0.0.1:$PORT_ROUTER/health" >/dev/null && break; sleep 1; done
  curl -sf "http://127.0.0.1:$PORT_ROUTER/health" >/dev/null || die "the router did not come up, see $STATE/logs/router.log"
  echo "router: up (port $PORT_ROUTER)"
  args+=(--hedge 0 --router-url "http://127.0.0.1:$PORT_ROUTER")
else
  args+=(--hedge -1)
fi
# shellcheck disable=SC2086
(cd pipeline && exec setsid uv run python -m voice_pipeline.agent_server "${args[@]}" ${AGENT_ARGS:-}) \
  >"$STATE/logs/agent.log" 2>&1 </dev/null &
echo $! >"$STATE/pids/agent"
for _ in $(seq 120); do grep -q "agent server on" "$STATE/logs/agent.log" 2>/dev/null && break; sleep 1; done
grep -q "agent server on" "$STATE/logs/agent.log" 2>/dev/null || die "the agent server did not come up, see $STATE/logs/agent.log"
grep "agent server on" "$STATE/logs/agent.log"
echo "ready: make text"
