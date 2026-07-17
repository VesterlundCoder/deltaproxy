#!/bin/bash
# 24-hour WIDE BLIND 8F7 discovery sweep on 2 cores.
# Covers the full 15-D shift space far more broadly than the box-6 run:
#   modes  uniform,wide   ->  shift ~ U[-9,9]^15  and  U[-14,14]^15
#   random multi-advance directions (rand-dir-prob 0.6, up to 3 roots)
#   z restricted to the convergent window |z| < 1 (|z|>=1 always diverges)
# Still null-result-aware: tracks global max dhat (any sign) + top-dhat list.
# Writes to a NEW dir so the earlier box-6 null data in sweep_8f7_24h/ is kept.
set -e
cd "$(dirname "$0")"
OUT=sweep_8f7_wide_z1_24h
mkdir -p "$OUT"

caffeinate -i -t 88200 >/dev/null 2>&1 &
echo "caffeinate pid $!"

for i in 1 2; do
  nohup python3 harvest_generic.py \
      --dim 8 --hours 24 --nrank 120 --modes uniform,wide \
      --box-uniform 9 --box-wide 14 --rand-dir-prob 0.6 --max-advance 3 \
      --z-max 1.0 --seed $((210 + i)) --save-thresh 0.05 --background-rate 0.001 \
      --out            "$OUT/harvest_w${i}.json" \
      --hits-out       "$OUT/positives_w${i}.jsonl" \
      --background-out "$OUT/background_w${i}.jsonl" \
      > "$OUT/w${i}.log" 2>&1 &
  echo "8F7-wide worker $i pid $!  (seed $((210 + i)))"
done

echo "Launched 2 WIDE 8F7 workers -> $OUT/. Tail with: tail -f $OUT/w1.log"
