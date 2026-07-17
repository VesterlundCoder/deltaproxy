#!/bin/bash
# 24-hour 6F5 discovery sweep on 4 cores: 4 independent harvest.py workers with
# distinct seeds, each streaming positives + background to its own files.
# Keeps the Mac awake for the duration with one caffeinate.
set -e
cd "$(dirname "$0")"
OUT=sweep_6f5_24h
mkdir -p "$OUT"

# keep awake ~24.5h (no display sleep), detached
caffeinate -i -t 88200 >/dev/null 2>&1 &
echo "caffeinate pid $!"

for i in 1 2 3 4; do
  nohup python3 harvest.py \
      --hours 24 --nrank 120 --seed $((100 + i)) \
      --save-thresh 0.05 --background-rate 0.001 \
      --out          "$OUT/harvest_w${i}.json" \
      --hits-out     "$OUT/positives_w${i}.jsonl" \
      --background-out "$OUT/background_w${i}.jsonl" \
      > "$OUT/w${i}.log" 2>&1 &
  echo "worker $i pid $!  (seed $((100 + i)))"
done

echo "Launched 4 workers -> $OUT/. Tail with: tail -f $OUT/w1.log"
