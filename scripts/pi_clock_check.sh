#!/usr/bin/env bash
# Ticket 268 (part 2): Pi clock sanity, run by the OWNER on the Pi before a demo run.
#
# The agent never runs this (it needs sudo and wifi). By default the script only
# REPORTS - the local clock, the time zone and the NTP state. With --fix it also acts:
# when NTP is not synchronised it sets the clock ONCE from an external HTTPS Date
# header (curl -sI https://www.google.com) via `sudo date -s`, then re-checks.
#
# Exit codes: 0 = synchronised (or the clock was just corrected); 1 = NOT
# synchronised (report mode) or the correction failed; 2 = bad usage.
set -euo pipefail

FIX=0
for arg in "$@"; do
  case "$arg" in
    --fix) FIX=1 ;;
    -h|--help)
      echo "usage: pi_clock_check.sh [--fix]"
      echo ""
      echo "Report the Pi's clock, time zone and NTP state (read-only by default)."
      echo "With --fix: when NTP is not synchronised, set the clock once from an"
      echo "external HTTPS Date header (sudo date -s), then re-check."
      exit 0
      ;;
    *)
      echo "unknown option: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

echo "local clock: $(date '+%Y-%m-%d %H:%M:%S %Z (%z)')"
if [ -r /etc/timezone ]; then
  echo "zone:        $(cat /etc/timezone)"
else
  echo "zone:        $(timedatectl show -p Timezone --value 2>/dev/null | sed 's/^Timezone=//' || echo unknown)"
fi

ntp_synced=""
if command -v timedatectl >/dev/null 2>&1; then
  ntp_line=$(timedatectl show -p NTPSynchronized 2>/dev/null || true)
  ntp_synced=${ntp_line#NTPSynchronized=}
  echo "ntp:         ${ntp_line:-unknown (timedatectl answered nothing)}"
elif command -v chronyc >/dev/null 2>&1; then
  tracking=$(chronyc tracking 2>/dev/null || true)
  if [ -n "$tracking" ]; then
    # chronyc tracking answers only when a reference source is in use
    ntp_synced=yes
    echo "ntp:         chronyc: $(echo "$tracking" | head -n 1)"
  else
    echo "ntp:         unknown (chronyc has no reference source)"
  fi
else
  echo "ntp:         unknown (no timedatectl and no chronyc installed)"
fi

case "$ntp_synced" in
  yes|true|1)
    echo "state:       synchronised"
    exit 0
    ;;
esac

echo "state:       NOT synchronised"
if [ "$FIX" -ne 1 ]; then
  echo ""
  echo "the agent's start-up clock check will flag this run until it is fixed."
  echo "run again with --fix (needs sudo and wifi) to set the clock once from an"
  echo "external HTTPS Date header, or fix NTP itself (systemd-timesyncd/chrony)."
  exit 1
fi

# the external reference: the RFC-1123 Date header of any HTTPS response
date_header=$(curl -sI --max-time 15 https://www.google.com \
  | awk 'tolower($1) == "date:" {print $2, $3, $4, $5, $6; exit}')
if [ -z "${date_header:-}" ]; then
  echo "could not read an external Date header (offline or no wifi); not touching the clock." >&2
  exit 1
fi
echo "reference:   $date_header (HTTPS Date header)"
sudo date -s "$date_header"
echo "clock now:   $(date '+%Y-%m-%d %H:%M:%S %Z (%z)')"

# re-check: report the NTP state again (it may still read 'no' - date -s sets the wall
# clock, it does not start an NTP service; the point is the date is now right)
if command -v timedatectl >/dev/null 2>&1; then
  re_line=$(timedatectl show -p NTPSynchronized 2>/dev/null || true)
  echo "re-check:    ${re_line:-unknown}"
else
  echo "re-check:    ntp state unknown (no timedatectl)"
fi
echo "clock corrected from the external reference; NTP itself is still unsynchronised"
echo "until a time service (systemd-timesyncd/chrony) is running."
exit 0
