#!/usr/bin/env bash
# down.sh: stop what up.sh started (by PID file; the process groups it made), nothing else.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PIDS=$REPO/out/text/pids
for name in agent router; do
  [ -f "$PIDS/$name" ] || continue
  pid=$(cat "$PIDS/$name")
  if kill -0 "$pid" 2>/dev/null; then kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null; echo "stopped $name ($pid)"; fi
  rm -f "$PIDS/$name"
done
