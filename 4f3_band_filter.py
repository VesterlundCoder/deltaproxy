"""
4f3_band_filter.py  (runs ON LUMI under srun)
=============================================
Stream the 431M 4F3 survivor shards, keep records in a delta band, DEDUP by the
CMF-defining key (shift, dir, z_num, z_den) keeping the max r2 per key, and write
a compact deduped candidate file. Also prints band/dedup counts.

Usage (on LUMI):
  srun ... python3 4f3_band_filter.py --lo 0.02 --hi 1.0 \
       --dir /scratch/project_465002669/cmf/runs/4f3_50pct_19723871 \
       --out /scratch/project_465002669/cmf/runs/4f3_50pct_19723871/band_0p02_1p0_dedup.jsonl
"""
import argparse, glob, heapq, json, os
from multiprocessing import Pool

# NOTE: bounded-memory. Per worker we keep only a size-K min-heap of the highest
# r2 records (deduped by CMF key), plus scalar counters. The full band is far too
# large to materialise (hundreds of millions), and PSLQ is only feasible on a
# small top slice, so we extract the top-K by delta.


def scan(args):
    fn, lo, hi, K = args
    heap = []                       # min-heap of (r2, uid, key, line)
    inheap = {}                     # key -> r2 currently retained
    uid = 0
    tot = band = 0
    hist = {}                       # coarse r2 histogram (0.05 bins) over band
    with open(fn) as f:
        for line in f:
            i = line.find('"r2": ')
            if i < 0:
                continue
            tot += 1
            try:
                v = float(line[i + 6:].split(",", 1)[0])
            except ValueError:
                continue
            if not (lo < v < hi):
                continue
            band += 1
            b = int(v / 0.05)
            hist[b] = hist.get(b, 0) + 1
            rec = json.loads(line)
            key = (tuple(rec["shift"]), tuple(rec["dir"]),
                   int(rec["z_num"]), int(rec["z_den"]))
            if key in inheap:        # duplicate config -> identical r2, skip
                continue
            if len(heap) < K:
                uid += 1
                heapq.heappush(heap, (v, uid, key, line.rstrip("\n")))
                inheap[key] = v
            elif v > heap[0][0]:
                _, _, oldkey, _ = heapq.heappop(heap)
                inheap.pop(oldkey, None)
                uid += 1
                heapq.heappush(heap, (v, uid, key, line.rstrip("\n")))
                inheap[key] = v
    return tot, band, hist, heap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--hi", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=5000)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "survivors_p*.jsonl")))
    print("scanning %d shards, band %g<r2<%g, top-%d by delta"
          % (len(files), a.lo, a.hi, a.topk), flush=True)
    with Pool(a.procs) as p:
        res = p.map(scan, [(fn, a.lo, a.hi, a.topk) for fn in files])
    tot = sum(r[0] for r in res)
    band = sum(r[1] for r in res)
    hist = {}
    for _, _, h, _ in res:
        for b, c in h.items():
            hist[b] = hist.get(b, 0) + c
    # merge per-worker heaps -> global top-K distinct by r2
    best = {}                       # key -> (r2, line)
    for _, _, _, heap in res:
        for v, _uid, key, line in heap:
            cur = best.get(key)
            if cur is None or v > cur[0]:
                best[key] = (v, line)
    top = sorted(best.items(), key=lambda kv: -kv[1][0])[:a.topk]
    with open(a.out, "w") as g:
        for _key, (_v, line) in top:
            g.write(line + "\n")
    print("total=%d  band=%d  band_frac=%.4f" % (tot, band, band / tot), flush=True)
    print("wrote top-%d distinct-by-config to %s (max r2=%.5f, min r2=%.5f)"
          % (len(top), a.out, top[0][1][0], top[-1][1][0]), flush=True)
    print("band r2 histogram (0.05 bins):", flush=True)
    for b in sorted(hist):
        print("  [%.2f,%.2f): %d" % (b * 0.05, b * 0.05 + 0.05, hist[b]), flush=True)


if __name__ == "__main__":
    main()
