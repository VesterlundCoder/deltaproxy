#!/bin/bash
# 24-hour BLIND 8F7 discovery sweep on 2 cores (broad + uniform modes).
# Null-result-aware: each worker tracks the global max dhat (any sign) so the
# absence/rarity of positive-delta 8F7 trajectories is documented quantitatively.
set -e
cd "$(dirname "$0")"
OUT=sweep_8f7_24h
mkdir -p "$OUT"

caffeinate -i -t 88200 >/dev/null 2>&1 &
echo "caffeinate pid $!"

for i in 1 2; do
  nohup python3 harvest_generic.py \
      --dim 8 --hours 24 --nrank 120 --modes broad,uniform \
      --seed $((200 + i)) --save-thresh 0.05 --background-rate 0.001 \
      --out            "$OUT/harvest_w${i}.json" \
      --hits-out       "$OUT/positives_w${i}.jsonl" \
      --background-out "$OUT/background_w${i}.jsonl" \
      > "$OUT/w${i}.log" 2>&1 &
  echo "8F7 worker $i pid $!  (seed $((200 + i)))"
done

echo "Launched 2 8F7 workers -> $OUT/. Tail with: tail -f $OUT/w1.log"
