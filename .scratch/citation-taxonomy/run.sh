#!/usr/bin/env bash
# 3 axes, agy only, plan mode (search and page reading only); never the skip-permissions flag.
cd /mnt/School/PhD/AI231/AI-231-ME4/.scratch/citation-taxonomy || exit 1
for q in A-freshness B-attribution C-industry; do
  ( timeout 3300 agy --mode plan --model gemini-3.8-flash-high --print-timeout 45m \
      -p "$(cat header-agy.txt common.txt q-$q.txt)" > AGY-$q.md 2> agy-$q.err
    echo "exit $?" > agy-$q.done ) &
done
wait
echo all done
