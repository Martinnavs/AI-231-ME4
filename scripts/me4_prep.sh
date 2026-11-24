#!/usr/bin/env bash
# me4_prep.sh — Pi preparation runbook for ME4 document ingest
# Usage: me4_prep.sh SRC DB [--execute] [--smoke FILE]
# Without --execute: dry-run (prints commands, runs none with side effects).
set -euo pipefail

SRC="${1:?usage: me4_prep.sh SRC DB [--execute] [--smoke FILE]}"
DB="${2:?usage: me4_prep.sh SRC DB [--execute] [--smoke FILE]}"
EXECUTE=0
SMOKE=""

shift 2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --execute) EXECUTE=1 ;;
        --smoke) SMOKE="$2"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

EXEC=":"  # no-op for dry-run
if [[ "$EXECUTE" -eq 1 ]]; then
    EXEC="exec_cmd"
fi

exec_cmd() {
    echo "→ $*"
    "$@"
}

TMPDIR="${TMPDIR:-/tmp}"
LOGFILE="$TMPDIR/me4_prep_temp_$$"
trap '
    # Restore governor and restart services on exit
    if [[ "$EXECUTE" -eq 1 ]]; then
        for gov in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
            echo ondemand > "$gov" 2>/dev/null || true
        done
        systemctl --user start pi-speaker pi-device-mcp 2>/dev/null || true
    fi
    rm -f "$LOGFILE"
' EXIT

# 1. Preflight
echo "=== Preflight ==="
SRC_SIZE=$(du -sm "$SRC" 2>/dev/null | awk '{print $1}')
SRC_SIZE=${SRC_SIZE:-0}
REQUIRED_MB=$(( SRC_SIZE * 3 + 512 ))
FREE_MB=$(df -m "$(dirname "$DB")" | awk 'NR==2{print $4}')
echo "Source size: ${SRC_SIZE} MB, required: ${REQUIRED_MB} MB, free: ${FREE_MB} MB"

if [[ "$EXECUTE" -eq 0 ]]; then
    echo "[DRY-RUN] would check: df, vcgencmd, meminfo, swapon, services"
    echo "[DRY-RUN] df -m \"$(dirname "$DB")\""
    echo "[DRY-RUN] vcgencmd get_throttled"
    echo "[DRY-RUN] grep MemAvailable /proc/meminfo"
    echo "[DRY-RUN] swapon --show"
    echo "[DRY-RUN] systemctl --user status pi-speaker pi-device-mcp"
    echo "[DRY-RUN] grep . /sys/devices/system/cpu/cpufreq/policy*/scaling_governor"
else
    vcgencmd get_throttled 2>/dev/null || echo "vcgencmd: not available"
    grep MemAvailable /proc/meminfo 2>/dev/null || echo "meminfo: not available"
    swapon --show 2>/dev/null || echo "swapon: not available"
    systemctl --user status pi-speaker pi-device-mcp 2>/dev/null || echo "services: not available"
    for gov in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
        echo "  $gov: $(cat "$gov")"
    done 2>/dev/null || echo "governor: not available"
fi

# 2. Stop ME3 services
echo "=== Stop services ==="
echo "${EXEC} systemctl --user stop pi-speaker pi-device-mcp"
$EXEC systemctl --user stop pi-speaker pi-device-mcp 2>/dev/null || true

# 3. Set performance governor
echo "=== Governor ==="
if [[ "$EXECUTE" -eq 0 ]]; then
    echo "[DRY-RUN] would set scaling_governor=performance"
else
    for gov in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
        echo performance > "$gov" 2>/dev/null || true
    done
fi

# 4. Temperature logger (background)
echo "=== Temperature logger ==="
if [[ "$EXECUTE" -eq 0 ]]; then
    echo "[DRY-RUN] would start temperature logger (10s interval)"
else
    (
        while true; do
            TEMP=$(vcgencmd measure_temp 2>/dev/null || echo "temp=N/A")
            echo "$(date -Is) $TEMP" >> "$LOGFILE"
            sleep 10
        done
    ) &
    TEMP_PID=$!
fi

# 5. Run ingest
echo "=== Ingest ==="
INGEST_CMD="python -m voice_agent.rag.ingest $SRC --out $DB --workers auto --deadline-min 50"
if [[ -n "$SMOKE" ]]; then
    INGEST_CMD="$INGEST_CMD --smoke $SMOKE"
fi
echo "${EXEC} $INGEST_CMD"
$EXEC python -m voice_agent.rag.ingest "$SRC" --out "$DB" --workers auto --deadline-min 50
INGEST_RC=$?

# 6. Verify
echo "=== Verify ==="
if [[ "$EXECUTE" -eq 0 ]]; then
    echo "[DRY-RUN] would verify meta.complete=1, files count, smoke searches"
else
    # Kill temp logger
    kill "$TEMP_PID" 2>/dev/null || true

    # Check complete
    COMPLETE=$(python -c "
import sqlite3
conn = sqlite3.connect('$DB')
r = conn.execute(\"SELECT value FROM meta WHERE key='complete'\").fetchone()
print(r[0] if r else '?')
conn.close()
")
    echo "meta.complete=$COMPLETE"

    # File counts
    python -c "
import sqlite3
conn = sqlite3.connect('$DB')
total = conn.execute('SELECT COUNT(*) FROM files').fetchone()[0]
ok = conn.execute(\"SELECT COUNT(*) FROM files WHERE status='ok'\").fetchone()[0]
failed = conn.execute(\"SELECT COUNT(*) FROM files WHERE status='unreadable'\").fetchone()[0]
skipped = conn.execute(\"SELECT COUNT(*) FROM files WHERE status='skipped'\").fetchone()[0]
print(f'files: {total} total, {ok} ok, {failed} failed, {skipped} skipped')
conn.close()
"

    # Smoke searches
    if [[ -n "$SMOKE" ]]; then
        python -c "
import json, sys
sys.path.insert(0, 'src')
from voice_agent.rag.dbindex import DbIndex
idx = DbIndex('$DB')
with open('$SMOKE') as f:
    questions = json.load(f)
smokes = questions[:3]
for q in smokes:
    r = idx.search(q['query'], 4)
    hits = len(r)
    print(f'  smoke \"{q[\"query\"][:40]}\": {hits} hits')
" 2>&1 || echo "smoke search: error"
    fi
fi

if [[ "$EXECUTE" -eq 1 && "$INGEST_RC" -ne 0 ]]; then
    echo "ingest failed with code $INGEST_RC" >&2
    exit "$INGEST_RC"
fi

echo "=== Done ==="
