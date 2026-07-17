"""
proxy_batch.py
==============
BATCHED spectral proxy -- the GPU-portable form of the delta detector.

The detector is QR-stabilized float Lyapunov on a tiny d x d companion cocycle.
That is embarrassingly parallel across trajectories and needs NO big integers
(unlike the RNS brute force), so it maps 1:1 onto a SIMT GPU: one lane per
trajectory, batched d x d matmul + batched QR for N steps.

This module implements that batched form with numpy (xp=np). The SAME code runs
on GPU by passing xp=cupy (drop-in) -- every op used (arithmetic, einsum,
linalg.qr, sign, log, argmin) exists in cupy and torch. Use it to (a) screen on
CPU in vectorized batches and (b) benchmark the structure to project GPU scale.

Validated to match the scalar engine (cmf_generic.spectral_ratios) to ~1e-12.

Usage:
  python3 proxy_batch.py --dim 6 --batch 20000 --nrank 120 --bench
  python3 proxy_batch.py --dim 8 --batch 20000 --nrank 120 --bench
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def esym_batch(vals, xp):
    """Elementary symmetric polynomials of each row.
    vals: (B, m) -> e: (B, m+1) with e[:,0]=1, e[:,k]=e_k(vals)."""
    B, m = vals.shape
    e = xp.zeros((B, m + 1), dtype=vals.dtype)
    e[:, 0] = 1.0
    for j in range(m):                       # add one root at a time (DP)
        v = vals[:, j:j + 1]                 # (B,1)
        # e[:,k] += v*e[:,k-1] for k=m..1  (descending to avoid overwrite)
        e[:, 1:m + 1] = e[:, 1:m + 1] + v * e[:, 0:m]
    return e


def build_M_batch(n, shift, dirv, z_num, z_den, dim, xp):
    """Batched companion matrices M(n): (B, dim, dim)."""
    B = shift.shape[0]
    f = shift[:, :dim] + n * dirv[:, :dim] + 1.0            # (B,dim)
    g = shift[:, dim:] + n * dirv[:, dim:] + 2.0            # (B,dim-1)
    ef = esym_batch(f, xp)                                  # (B,dim+1)
    eg = esym_batch(g, xp)                                  # (B,dim)
    eg = xp.concatenate([eg, xp.zeros((B, 1), dtype=eg.dtype)], axis=1)  # (B,dim+1)
    hf = -z_num.reshape(-1, 1)                              # (B,1)
    hg = z_den.reshape(-1, 1)
    c = xp.zeros((B, dim), dtype=shift.dtype)
    c[:, 0] = (hf[:, 0]) * ef[:, dim]
    for k in range(1, dim):
        c[:, k] = hg[:, 0] * eg[:, dim - k] + hf[:, 0] * ef[:, dim - k]
    M = xp.zeros((B, dim, dim), dtype=shift.dtype)
    for r in range(1, dim):
        M[:, r, r - 1] = 1.0
    M[:, :, dim - 1] = c
    return M


def lyapunov_batch(shift, dirv, z_num, z_den, dim, N, xp=np, qr_every=1):
    """Batched finite-time Lyapunov spectra. Returns lam: (B, dim) descending.

    qr_every>1 amortizes the QR over several raw matmuls (re-orthonormalize less
    often). qr_every=1 is the safe default (matches the scalar engine)."""
    B = shift.shape[0]
    Q = xp.broadcast_to(xp.eye(dim, dtype=shift.dtype), (B, dim, dim)).copy()
    logs = xp.zeros((B, dim), dtype=shift.dtype)
    for n in range(N):
        M = build_M_batch(n + 1, shift, dirv, z_num, z_den, dim, xp)
        Q = xp.einsum('bij,bjk->bik', M, Q)
        if (n + 1) % qr_every == 0 or n == N - 1:
            Qn, R = xp.linalg.qr(Q)
            d = xp.diagonal(R, axis1=1, axis2=2)           # (B,dim)
            s = xp.sign(d); s = xp.where(s == 0, 1.0, s)
            Q = Qn * s[:, None, :]                          # column sign fix
            logs = logs + xp.log(xp.maximum(xp.abs(d), 1e-300))
    return logs / N


def ratios_batch(lam, xp=np):
    """r_k = -lam_k/lam_1 for k=2..dim. lam:(B,dim) -> (B,dim-1)."""
    l1 = lam[:, 0:1]
    bad = ~xp.isfinite(l1) | (l1 == 0)
    r = -lam[:, 1:] / xp.where(bad, 1.0, l1)
    r = xp.where(bad, xp.nan, r)
    return r


# ── verification + benchmark ──────────────────────────────────────────────
def _verify(dim, xp=np):
    import random
    import cmf_generic as cg
    rng = random.Random(3)
    nsh = cg.nshift_for(dim); ZS = cg.z_pool(z_max=1.0); DIRS = cg.dir_pool(dim)
    shifts, dirs, zn, zd, scal = [], [], [], [], []
    for _ in range(64):
        sh = [rng.randint(-5, 5) for _ in range(nsh)]
        dv = list(rng.choice(DIRS)); a, b = rng.choice(ZS)
        shifts.append(sh); dirs.append(dv); zn.append(a); zd.append(b)
        r, _ = cg.spectral_ratios(sh, dv, a, b, 120, dim)
        scal.append(r)
    lam = lyapunov_batch(xp.asarray(np.array(shifts, float)), xp.asarray(np.array(dirs, float)),
                         xp.asarray(np.array(zn, float)), xp.asarray(np.array(zd, float)),
                         dim, 120, xp)
    rb = ratios_batch(lam, xp)
    rb = np.asarray(rb.get() if hasattr(rb, "get") else rb)
    scal = np.array(scal); err = np.nanmax(np.abs(rb - scal))
    print(f"  [verify dim={dim}] max|batch - scalar| over 64 traj, r_2..r_dim: {err:.2e}")
    return err


def _bench(dim, batch, N, xp=np, qr_every=1):
    import random
    import cmf_generic as cg
    rng = random.Random(5)
    nsh = cg.nshift_for(dim); ZS = cg.z_pool(z_max=1.0); DIRS = cg.dir_pool(dim)
    shift = xp.asarray(np.array([[rng.randint(-6, 6) for _ in range(nsh)] for _ in range(batch)], float))
    dirv = xp.asarray(np.array([list(rng.choice(DIRS)) for _ in range(batch)], float))
    zn = xp.asarray(np.array([rng.choice(ZS)[0] for _ in range(batch)], float))
    zd = xp.asarray(np.array([rng.choice(ZS)[1] for _ in range(batch)], float))
    # warm-up
    lyapunov_batch(shift[:64], dirv[:64], zn[:64], zd[:64], dim, 8, xp, qr_every)
    if xp is not np:
        xp.cuda.Device().synchronize()
    t0 = time.time()
    lam = lyapunov_batch(shift, dirv, zn, zd, dim, N, xp, qr_every)
    r = ratios_batch(lam, xp)
    if xp is not np:
        xp.cuda.Device().synchronize()
    dt = time.time() - t0
    r = np.asarray(r.get() if hasattr(r, "get") else r)
    thr = batch / dt
    print(f"  [bench dim={dim}] B={batch:,} N={N} qr_every={qr_every}: "
          f"{dt:.2f}s -> {thr:,.0f} traj/s  ({thr*3600:,.0f}/core-hour)  "
          f"max r_2={np.nanmax(r[:,0]):+.4f}")
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--nrank", type=int, default=120)
    ap.add_argument("--qr-every", type=int, default=1)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--gpu", action="store_true",
                    help="run on GPU via cupy (cupy-rocm on LUMI/AMD)")
    args = ap.parse_args()
    xp = np
    if args.gpu:
        try:
            import cupy as cp
            xp = cp
            print("batched spectral proxy (xp=cupy GPU)")
        except Exception as e:
            print(f"cupy unavailable ({e}); falling back to numpy")
    else:
        print("batched spectral proxy (xp=numpy; pass --gpu for cupy)")
    if args.verify or not args.bench:
        _verify(args.dim, xp)
    if args.bench:
        _bench(args.dim, args.batch, args.nrank, xp, args.qr_every)


if __name__ == "__main__":
    main()
