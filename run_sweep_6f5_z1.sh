#!/bin/bash
# 24-hour 6F5 discovery sweep on 4 cores, z restricted to the convergent
# window |z| < 1 (|z|>=1 always diverges). Uses the restricted discover.Z_POOL.
# Fresh seeds + fresh OUT dir so the earlier sweep_6f5_24h/ data is preserved
# and there is no duplicate re-saving.
set -e
cd "$(dirname "$0")"
OUT=sweep_6f5_z1_24h
mkdir -p "$OUT"

caffeinate -i -t 88200 >/dev/null 2>&1 &
echo "caffeinate pid $!"

for i in 1 2 3 4; do
  nohup python3 harvest.py \
      --hours 24 --nrank 120 --seed $((300 + i)) \
      --save-thresh 0.05 --background-rate 0.001 \
      --out          "$OUT/harvest_w${i}.json" \
      --hits-out     "$OUT/positives_w${i}.jsonl" \
      --background-out "$OUT/background_w${i}.jsonl" \
      > "$OUT/w${i}.log" 2>&1 &
  echo "6F5-z1 worker $i pid $!  (seed $((300 + i)))"
done

echo "Launched 4 6F5 workers (|z|<1) -> $OUT/. Tail with: tail -f $OUT/w1.log"
