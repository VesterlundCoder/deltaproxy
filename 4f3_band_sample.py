"""
4f3_band_sample.py  (runs ON LUMI under sbatch)
===============================================
Exact-uniform random sample of N records from the delta band across the 4F3
survivor shards, WITHOUT replacement and WITHOUT materialising the 372M-record
band. Method: bottom-k sampling -- assign each band record an independent
Uniform(0,1) key and keep the k records with the smallest keys (a size-k max-heap
per worker on the key; merge; take the global k smallest). This is a provably
uniform simple random sample of size k over the union.

Usage (on LUMI):
  python3 4f3_band_sample.py --dir . --out band_sample_10k.jsonl \
      --lo 0.02 --hi 1.0 --k 10000 --seed 20260705 --procs 32
"""
import argparse, glob, heapq, json, os, random
from multiprocessing import Pool


def scan(args):
    fn, lo, hi, k, seed = args
    rng = random.Random((seed, os.path.basename(fn)).__hash__() & 0xFFFFFFFF)
    heap = []                       # max-heap of (-key, uid, line): keep k smallest keys
    uid = 0
    band = 0
    with open(fn) as f:
        for line in f:
            i = line.find('"r2": ')
            if i < 0:
                continue
            try:
                v = float(line[i + 6:].split(",", 1)[0])
            except ValueError:
                continue
            if not (lo < v < hi):
                continue
            band += 1
            key = rng.random()
            uid += 1
            if len(heap) < k:
                heapq.heappush(heap, (-key, uid, line.rstrip("\n")))
            elif key < -heap[0][0]:
                heapq.heapreplace(heap, (-key, uid, line.rstrip("\n")))
    return band, heap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--hi", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "survivors_p*.jsonl")))
    print("sampling k=%d uniform from band %g<r2<%g over %d shards"
          % (a.k, a.lo, a.hi, len(files)), flush=True)
    with Pool(a.procs) as p:
        res = p.map(scan, [(fn, a.lo, a.hi, a.k, a.seed) for fn in files])
    band = sum(r[0] for r in res)
    # global bottom-k merge on the uniform key
    allkeyed = []
    for _, heap in res:
        for negk, uid, line in heap:
            allkeyed.append((-negk, line))     # (key, line)
    allkeyed.sort(key=lambda t: t[0])
    chosen = allkeyed[:a.k]
    with open(a.out, "w") as g:
        for _key, line in chosen:
            g.write(line + "\n")
    print("band=%d  sampled=%d  -> %s" % (band, len(chosen), a.out), flush=True)


if __name__ == "__main__":
    main()
