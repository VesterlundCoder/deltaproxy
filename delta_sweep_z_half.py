"""
delta_sweep_z_half.py
=====================
1-billion-trajectory delta proxy sweep at z = ±1/2 for 6F5 companion CMFs.

For z=±1/2, all r2 values are negative (uniformly contracting cocycle).
Positive delta corresponds to r2 close to 0 (threshold -0.02).

Pipeline:
  1. 6 parallel numpy proxy workers screen 1B trajectories (r2 > -0.02 filter)
  2. Main process collects survivors, computes exact integer delta
  3. Keeps only exact delta > 0
  4. Runs full PSLQ battery on every positive-delta survivor
  5. Stores all positive-delta survivors with PSLQ results to JSONL

Usage:
  python3 delta_sweep_z_half.py --total 1000000000 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Queue, Process
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmf_generic as cg
from proxy_batch import lyapunov_batch, ratios_batch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_DIR = os.path.join(CODE_DIR, "gpu_proxy")
sys.path.insert(0, GPU_DIR)
from params import gen_params, build_pools

DIM = 6
BOX = 6
NSTEPS = 120
Z_FIXED = np.array([(1, 2), (-1, 2)], dtype=np.int64)  # z = ±1/2
BATCH = 20000  # trajectories per proxy call
R2_THRESH = -0.02  # proxy threshold for positive delta (z=1/2: all r2 < 0)


def sweep_worker(worker_id, seed, gid_start, gid_end, out_queue):
    """One sweep worker: scans gid_start..gid_end, sends r2 > R2_THRESH survivors."""
    nsh = cg.nshift_for(DIM)
    dirs, _ = build_pools(DIM, z_max=1.0)
    zs = Z_FIXED
    ndir, nz = len(dirs), len(zs)
    gid = gid_start
    local_count = 0
    local_survivors = 0
    while gid < gid_end:
        n = min(BATCH, gid_end - gid)
        gids = np.arange(gid, gid + n, dtype=np.uint64)
        shift, di, zi = gen_params(seed, gids, nsh, BOX, ndir, nz)
        dirv = dirs[di].astype(np.float64)
        zn = zs[zi, 0].astype(np.float64)
        zd = zs[zi, 1].astype(np.float64)
        lam = lyapunov_batch(shift.astype(np.float64), dirv, zn, zd, DIM, NSTEPS, np)
        r = ratios_batch(lam, np)
        r2 = r[:, 0]
        l1 = lam[:, 0]
        finite = np.isfinite(r2)
        hit = finite & (r2 > R2_THRESH)
        idx = np.nonzero(hit)[0]
        for j in idx:
            rec = {
                "gid": int(gid + j),
                "r2": float(r2[j]),
                "lam1": float(l1[j]),
                "shift": shift[j].tolist(),
                "dir": dirs[di[j]].tolist(),
                "z_num": int(zs[zi[j], 0]),
                "z_den": int(zs[zi[j], 1]),
                "worker": worker_id,
            }
            out_queue.put(rec)
            local_survivors += 1
        gid += n
        local_count += n
        if local_count % 400000 == 0:
            out_queue.put({"_progress": True, "worker": worker_id,
                           "screened": local_count, "survivors": local_survivors})
    out_queue.put(None)  # signal done


def exact_delta_one(args):
    """Compute exact delta for one survivor (for multiprocessing)."""
    rec, N, dps = args
    import mpmath as mp
    mp.mp.dps = dps
    sys.path.insert(0, CODE_DIR)
    import pslq_companion as pc

    out = {"gid": rec.get("gid"), "r2": rec.get("r2"),
           "lam1": rec.get("lam1"),
           "shift": rec["shift"], "dir": rec["dir"],
           "z_num": rec["z_num"], "z_den": rec["z_den"],
           "worker": rec.get("worker")}
    try:
        d, L, pair = pc.delta_and_limit(
            rec["shift"], rec["dir"],
            rec["z_num"], rec["z_den"], DIM, N
        )
        out["delta_exact"] = d
        out["pair"] = list(pair) if pair else None
        if d is not None and d > 0.0 and L is not None:
            out["L"] = mp.nstr(L, 50)
            out["pslq"] = pc.pslq_identify(L, 100000, 20000)
        else:
            out["L"] = None
            out["pslq"] = []
    except Exception as e:
        out["delta_exact"] = None
        out["pslq"] = []
        out["error"] = str(e)[:140]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total", type=int, default=1_000_000_000,
                    help="total trajectories to screen")
    ap.add_argument("--workers", type=int, default=6, help="proxy sweep workers")
    ap.add_argument("--pslq-workers", type=int, default=4,
                    help="parallel workers for exact delta + PSLQ")
    ap.add_argument("--N", type=int, default=90, help="exact delta depth (2N walk)")
    ap.add_argument("--dps", type=int, default=160)
    ap.add_argument("--out", default="cluster_out/delta_sweep_z_half")
    ap.add_argument("--seed", type=int, default=20260708)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    survivors_file = outdir / "proxy_survivors.jsonl"
    posdelta_file = outdir / "positive_delta.jsonl"
    pslq_file = outdir / "pslq_results.jsonl"
    log_file = outdir / "sweep.log"

    per_worker = args.total // args.workers

    queue = Queue(maxsize=10000)
    procs = []
    for w in range(args.workers):
        gs = w * per_worker
        ge = (w + 1) * per_worker if w < args.workers - 1 else args.total
        seed = args.seed + w
        p = Process(target=sweep_worker, args=(w, seed, gs, ge, queue))
        p.start()
        procs.append(p)

    sf = open(survivors_file, "w")
    pf = open(posdelta_file, "w")
    xf = open(pslq_file, "w")
    logf = open(log_file, "w")

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"START z=±1/2 delta sweep: {args.total:,} trajectories, "
        f"{args.workers} proxy workers, r2_thresh={R2_THRESH}, "
        f"dim={DIM}, box={BOX}, N={args.N}")

    done_workers = 0
    total_screened = 0
    total_survivors = 0
    total_pos_delta = 0
    total_pslq_hits = 0
    all_survivors = []
    t0 = time.time()

    # Phase 1: collect all proxy survivors
    while done_workers < args.workers:
        try:
            rec = queue.get(timeout=10)
        except Exception:
            if done_workers >= args.workers:
                break
            continue

        if rec is None:
            done_workers += 1
            log(f"proxy worker {done_workers}/{args.workers} finished")
            continue

        if rec.get("_progress"):
            total_screened += rec["screened"]
            el = time.time() - t0
            rate = total_screened / el if el > 0 else 0
            log(f"  progress: screened={total_screened:,}  "
                f"survivors={total_survivors:,}  "
                f"rate={rate:,.0f}/s  "
                f"elapsed={el/60:.1f}min")
            continue

        sf.write(json.dumps(rec) + "\n")
        sf.flush()
        total_survivors += 1
        all_survivors.append(rec)

    sf.close()
    el_proxy = time.time() - t0
    log(f"PHASE 1 DONE: {total_survivors:,} proxy survivors in {el_proxy/60:.1f}min")

    # Phase 2: exact delta + PSLQ on all survivors
    log(f"PHASE 2: exact delta + PSLQ on {len(all_survivors):,} survivors "
        f"with {args.pslq_workers} workers")

    from multiprocessing import Pool
    task_args = [(r, args.N, args.dps) for r in all_survivors]
    t1 = time.time()

    with Pool(args.pslq_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(exact_delta_one, task_args, chunksize=8), 1):
            de = result.get("delta_exact")
            if de is not None and de > 0.0:
                pf.write(json.dumps(result) + "\n")
                pf.flush()
                xf.write(json.dumps(result) + "\n")
                xf.flush()
                total_pos_delta += 1
                pslq_hits = result.get("pslq", [])
                if pslq_hits:
                    total_pslq_hits += len(pslq_hits)
                    for h in pslq_hits:
                        if h.get("type") != "rational":
                            log(f"  *** PSLQ HIT: delta={de:.4f} "
                                f"z={result['z_num']}/{result['z_den']} "
                                f"L={result.get('L','?')[:30]} "
                                f"type={h['type']} "
                                f"coeffs={h['coeffs']}")
            if i % 500 == 0:
                el = time.time() - t1
                log(f"  verified {i:,}/{len(all_survivors):,}  "
                    f"pos_delta={total_pos_delta:,}  "
                    f"pslq_hits={total_pslq_hits:,}  "
                    f"({i/el:.1f}/s)")

    pf.close()
    xf.close()
    logf.close()

    elapsed = time.time() - t0
    el_pslq = time.time() - t1
    print(f"\n{'='*70}")
    print(f"DONE: {elapsed/60:.1f} minutes total")
    print(f"  Phase 1 (proxy):  {el_proxy/60:.1f} min  -> {total_survivors:,} survivors")
    print(f"  Phase 2 (PSLQ):   {el_pslq/60:.1f} min  -> {total_pos_delta:,} positive delta")
    print(f"  PSLQ hits:         {total_pslq_hits:,}")
    print(f"  Proxy survivors -> {survivors_file}")
    print(f"  Positive delta  -> {posdelta_file}")
    print(f"  PSLQ results    -> {pslq_file}")
    print(f"  Log             -> {log_file}")


if __name__ == "__main__":
    main()
