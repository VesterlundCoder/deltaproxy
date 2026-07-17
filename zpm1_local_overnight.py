"""
zpm1_local_overnight.py
=======================
Local overnight z=±1 sweep for 6F5 companion CMFs.

Runs 6 parallel numpy sweep workers (different seeds, disjoint gid ranges).
Every 1000 new survivors, runs the full PSLQ battery (20-constant library)
and prints any closed-form hits to stdout + a results file.

Stops after --hours (default 12) or when the total target is reached.

Usage:
  python3 zpm1_local_overnight.py --hours 12 --workers 6 --batch-size 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
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
Z_FIXED = np.array([(1, 1), (-1, 1)], dtype=np.int64)
BAND_LO = -0.1
BAND_HI = 0.1
BATCH = 20000  # trajectories per proxy call


def sweep_worker(worker_id, seed, gid_start, gid_end, out_queue):
    """One sweep worker: scans gid_start..gid_end, sends survivors to queue."""
    nsh = cg.nshift_for(DIM)
    dirs, _ = build_pools(DIM, z_max=1.0)
    zs = Z_FIXED
    ndir, nz = len(dirs), len(zs)
    gid = gid_start
    local_count = 0
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
        hit = finite & (r2 > BAND_LO) & (r2 < BAND_HI)
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
            local_count += 1
        gid += n
    out_queue.put(None)  # signal done


def pslq_batch(survivors, dps=160, N=90, maxcoeff=100000):
    """Run PSLQ on a batch of survivors. Returns list of hit dicts."""
    import mpmath as mp
    mp.mp.dps = dps

    sys.path.insert(0, CODE_DIR)
    import pslq_companion as pc

    hits = []
    for rec in survivors:
        try:
            d, L, pair = pc.delta_and_limit(
                rec["shift"], rec["dir"],
                rec["z_num"], rec["z_den"], DIM, N
            )
            if d is None or L is None:
                continue
            rec["delta_exact"] = d
            if d >= -0.1:  # keep everything in band
                rec["L"] = mp.nstr(L, 40)
                pslq_hits = pc.pslq_identify(L, maxcoeff, 20000)
                if pslq_hits:
                    rec["pslq"] = pslq_hits
                    hits.append(rec)
        except Exception as e:
            rec["delta_exact"] = None
            rec["error"] = str(e)[:100]
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1000,
                    help="run PSLQ every N survivors")
    ap.add_argument("--out", default="cluster_out/zpm1_overnight")
    ap.add_argument("--seed", type=int, default=20260706)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    survivors_file = outdir / "all_survivors.jsonl"
    hits_file = outdir / "pslq_hits.jsonl"
    log_file = outdir / "overnight.log"

    # Each worker gets a large gid range (2^60 / workers) — effectively unbounded
    gid_space = 1 << 60
    per_worker = gid_space // args.workers

    queue = Queue(maxsize=args.batch_size * 4)
    procs = []
    for w in range(args.workers):
        gs = w * per_worker
        ge = (w + 1) * per_worker
        seed = args.seed + w
        p = Process(target=sweep_worker, args=(w, seed, gs, ge, queue))
        p.start()
        procs.append(p)

    deadline = time.time() + args.hours * 3600
    buffer = []
    total_survivors = 0
    total_pslq_batches = 0
    total_hits = 0
    done_workers = 0

    sf = open(survivors_file, "w")
    hf = open(hits_file, "w")
    logf = open(log_file, "w")

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"START z=±1 overnight sweep: {args.workers} workers, "
        f"{args.hours}h, PSLQ every {args.batch_size} survivors, "
        f"band=[{BAND_LO},{BAND_HI}], dim={DIM}, box={BOX}")

    while done_workers < args.workers and time.time() < deadline:
        try:
            rec = queue.get(timeout=5)
        except Exception:
            if time.time() >= deadline:
                break
            continue

        if rec is None:
            done_workers += 1
            log(f"worker {done_workers}/{args.workers} finished its gid range")
            continue

        sf.write(json.dumps(rec) + "\n")
        sf.flush()
        buffer.append(rec)
        total_survivors += 1

        if len(buffer) >= args.batch_size:
            batch = buffer[:]
            buffer = []
            total_pslq_batches += 1
            t0 = time.time()
            log(f"PSLQ batch #{total_pslq_batches}: {len(batch)} survivors "
                f"(total={total_survivors})")
            hits = pslq_batch(batch)
            dt = time.time() - t0
            if hits:
                total_hits += len(hits)
                for h in hits:
                    hf.write(json.dumps(h) + "\n")
                    hf.flush()
                    log(f"  *** HIT: delta={h.get('delta_exact'):.4f} "
                        f"L={h.get('L','?')[:30]} "
                        f"pslq={h.get('pslq','?')}")
                log(f"  -> {len(hits)} hits in {dt:.1f}s "
                    f"(running total: {total_hits})")
            else:
                log(f"  -> 0 hits in {dt:.1f}s (running total: {total_hits})")

    # Final PSLQ on remaining buffer
    if buffer:
        log(f"FINAL PSLQ batch: {len(buffer)} remaining survivors")
        hits = pslq_batch(buffer)
        if hits:
            total_hits += len(hits)
            for h in hits:
                hf.write(json.dumps(h) + "\n")
                hf.flush()
                log(f"  *** HIT: delta={h.get('delta_exact'):.4f} "
                    f"L={h.get('L','?')[:30]} "
                    f"pslq={h.get('pslq','?')}")

    # Terminate workers
    for p in procs:
        p.terminate()
        p.join(timeout=5)

    sf.close()
    hf.close()
    logf.close()

    elapsed = time.time() - (deadline - args.hours * 3600)
    print(f"\n{'='*60}")
    print(f"DONE: {total_survivors} survivors, {total_pslq_batches} PSLQ batches, "
          f"{total_hits} hits in {elapsed/3600:.1f}h")
    print(f"  survivors -> {survivors_file}")
    print(f"  hits      -> {hits_file}")
    print(f"  log       -> {log_file}")


if __name__ == "__main__":
    main()
