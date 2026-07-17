"""
filter_survivors.py
===================
On-LUMI reducer for oversized survivor sets. Streams all per-rank
survivors_p*.jsonl in a run dir, writes only records with r2 >= THRESH to
survivors_top.jsonl, and reports the r2 distribution (reservoir sample) so the
threshold can be tuned to yield a downloadable, high-delta subset for PSLQ.

Usage:
  python3 filter_survivors.py <run_dir> <thresh> [workers]
  e.g. python3 filter_survivors.py \
       /scratch/project_465002669/cmf/runs/4f3_companion_50pct_19782562 0.30 16
"""
import glob
import json
import os
import random
import sys
from multiprocessing import Pool

RUN = sys.argv[1]
TH = float(sys.argv[2])
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
CAP = 20000  # reservoir size per rank


def do(f):
    out = f + ".top"
    n = k = 0
    samp = []
    with open(out, "w") as w:
        for line in open(f):
            n += 1
            try:
                r = json.loads(line).get("r2", None)
            except Exception:
                continue
            if r is None:
                continue
            if len(samp) < CAP:
                samp.append(r)
            elif random.random() < 0.002:
                samp[random.randrange(CAP)] = r
            if r >= TH:
                w.write(line)
                k += 1
    return (n, k, samp)


def main():
    files = sorted(glob.glob(os.path.join(RUN, "survivors_p*.jsonl")))
    if not files:
        print("no survivors_p*.jsonl in", RUN)
        sys.exit(1)
    print(f"[filter] {len(files)} rank files  thresh={TH}  workers={WORKERS}", flush=True)
    with Pool(WORKERS) as pool:
        res = pool.map(do, files)
    N = sum(x[0] for x in res)
    K = sum(x[1] for x in res)
    S = sorted(v for x in res for v in x[2])
    m = len(S)
    q = (lambda p: S[min(int(m * p), m - 1)]) if m else (lambda p: float("nan"))
    top = os.path.join(RUN, "survivors_top.jsonl")
    os.system(f"cat {RUN}/survivors_p*.jsonl.top > {top}")
    os.system(f"rm -f {RUN}/survivors_p*.jsonl.top")
    sz = os.path.getsize(top) if os.path.exists(top) else 0
    print(f"[filter] scanned={N:,}  kept(r2>={TH})={K:,}  ->  {top} ({sz/1e6:.1f} MB)")
    print(f"[filter] r2 sample(n={m:,}): p50={q(.5):.3f} p90={q(.9):.3f} "
          f"p99={q(.99):.3f} p999={q(.999):.4f} max={S[-1] if S else float('nan'):.4f}")


if __name__ == "__main__":
    main()
