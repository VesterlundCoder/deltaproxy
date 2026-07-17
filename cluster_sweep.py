"""
cluster_sweep.py
================
Anchored neighborhood ("cluster") sweep in the COMPANION CMF family.

Rationale (Checkpoint-8 pivot):
  * The known Tauraso zeta(5) CMF is a single-axis recurrence, NOT a commuting
    (conservative) matrix field, so it cannot anchor a multi-axis CMF search, and
    its rational-function structure is not representable by the companion family.
  * In the companion gauge |z|>=1 DIVERGES (no convergent ratio), so z=+-1 is not
    scannable; the accessible analog is near-boundary z (|z|->1-) + the near-zero
    gap that the production pool excluded.
  * Therefore we do "hits cluster near hits" INSIDE the valid companion family:
    anchor on the real top exact-delta survivors we already have, and sweep a
    bounded integer neighborhood of each anchor (shift +- radius, optional dir
    switch, optional z resample from an EXTENDED pool). Keep every delta>=0 point
    for full local PSLQ downstream.

Pipeline:
  anchors (top |delta|) -> K stochastic perturbations each -> batched float64
  companion Lyapunov proxy r2 = -lam2/lam1 -> keep r2 >= thresh -> JSONL for PSLQ.

Usage:
  python3 cluster_sweep.py --anchors verify_6f5_lumi_out/phase3_ladder_pslq.jsonl \
      --dim 6 --n-anchors 1000 --per-anchor 10000 --radius 2 \
      --dir-switch 0.15 --z-mode extended --thresh 0.0 \
      --out cluster_out/survivors.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmf_generic as cg                                  # noqa: E402
from proxy_batch import lyapunov_batch, ratios_batch      # noqa: E402


def torch_proxy(shift, dirv, zn, zd, dim, nsteps, device, dtype):
    """GPU batched companion Lyapunov proxy for EXPLICIT params (not gid-based).
    Batched einsum + manual modified Gram-Schmidt QR (matches proxy_batch /
    proxy_kernel.run_torch; validated to ~1e-12 vs the scalar engine). Returns
    (r2, lam1) as numpy float64 arrays."""
    import torch
    td = torch.float64 if dtype == "float64" else torch.float32
    with torch.no_grad():
        sh = torch.as_tensor(shift.astype(np.float64), device=device, dtype=td)
        dv = torch.as_tensor(dirv.astype(np.float64), device=device, dtype=td)
        zn_t = torch.as_tensor(zn.astype(np.float64), device=device, dtype=td)
        zd_t = torch.as_tensor(zd.astype(np.float64), device=device, dtype=td)
        n = sh.shape[0]
        Q = torch.eye(dim, dtype=td, device=device).expand(n, dim, dim).contiguous()
        logs = torch.zeros(n, dim, dtype=td, device=device)

        def esym(vals):                                    # (n,m)->(n,m+1)
            B, m = vals.shape
            e = torch.zeros(B, m + 1, dtype=vals.dtype, device=vals.device)
            e[:, 0] = 1.0
            for j in range(m):
                e[:, 1:m + 1] = e[:, 1:m + 1] + vals[:, j:j + 1] * e[:, 0:m]
            return e

        for step in range(1, nsteps + 1):
            f = sh[:, :dim] + step * dv[:, :dim] + 1.0
            g = sh[:, dim:] + step * dv[:, dim:] + 2.0
            ef = esym(f)                                   # (n,dim+1)
            eg = esym(g)                                   # (n,dim)
            eg = torch.cat([eg, torch.zeros(n, 1, dtype=td, device=device)], 1)
            c = torch.zeros(n, dim, dtype=td, device=device)
            c[:, 0] = (-zn_t) * ef[:, dim]
            for k in range(1, dim):
                c[:, k] = zd_t * eg[:, dim - k] + (-zn_t) * ef[:, dim - k]
            M = torch.zeros(n, dim, dim, dtype=td, device=device)
            for r in range(1, dim):
                M[:, r, r - 1] = 1.0
            M[:, :, dim - 1] = c
            V = torch.einsum('bij,bjk->bik', M, Q)
            Qn = torch.empty_like(V)
            for j in range(dim):
                v = V[:, :, j].clone()
                for i in range(j):
                    qi = Qn[:, :, i]
                    v = v - (qi * v).sum(1, keepdim=True) * qi
                nrm = torch.clamp(v.norm(dim=1, keepdim=True), min=1e-300)
                logs[:, j] = logs[:, j] + torch.log(nrm.squeeze(1))
                Qn[:, :, j] = v / nrm
            Q = Qn
        lam = logs / nsteps
        l1 = lam[:, 0]
        r2 = torch.where(l1 != 0, -lam[:, 1] / l1, torch.full_like(l1, float("nan")))
    return (r2.detach().cpu().numpy().astype(np.float64),
            l1.detach().cpu().numpy().astype(np.float64))


def load_anchors(path, dim, n_anchors):
    """Read anchor records (phase3 or survivor JSONL); keep top-|delta| unique
    (shift,dir,z). Records must carry shift, dir, z_num, z_den. Delta key is
    best_ladder_delta (phase3) or exact_delta, else r2."""
    nsh = cg.nshift_for(dim)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            sh, dv = r.get("shift"), r.get("dir")
            zn, zd = r.get("z_num"), r.get("z_den")
            if sh is None or dv is None or zn is None or zd is None:
                continue
            if len(sh) != nsh or len(dv) != nsh:
                continue
            d = r.get("best_ladder_delta")
            if d is None:
                d = r.get("exact_delta")
            if d is None:
                d = r.get("r2", 0.0)
            rows.append((float(d if d is not None else 0.0),
                         tuple(int(x) for x in sh), tuple(int(x) for x in dv),
                         int(zn), int(zd)))
    # dedup by (shift,dir,z), keep max delta
    best = {}
    for d, sh, dv, zn, zd in rows:
        k = (sh, dv, zn, zd)
        if k not in best or d > best[k]:
            best[k] = d
    uniq = sorted(([d, *k] for k, d in best.items()), key=lambda x: x[0], reverse=True)
    anchors = uniq[:n_anchors]
    print(f"[anchors] read {len(rows)} rows -> {len(uniq)} unique -> "
          f"top {len(anchors)} anchors (delta {anchors[-1][0]:.4f}..{anchors[0][0]:.4f})")
    return anchors


def gen_neighborhood(anchors, per_anchor, radius, dir_switch, z_mode,
                     dim, ext_zpool, rng):
    """Vectorized generation of perturbed (shift,dir,zn,zd) around each anchor."""
    nsh = cg.nshift_for(dim)
    dir_pool = np.array(cg.dir_pool(dim), dtype=np.int64)   # (ND, nsh)
    ndir = len(dir_pool)
    tot = len(anchors) * per_anchor

    a_shift = np.array([a[1] for a in anchors], dtype=np.int64)   # (A,nsh)
    a_dir = np.array([a[2] for a in anchors], dtype=np.int64)     # (A,nsh)
    a_zn = np.array([a[3] for a in anchors], dtype=np.int64)      # (A,)
    a_zd = np.array([a[4] for a in anchors], dtype=np.int64)

    rep = np.repeat(np.arange(len(anchors)), per_anchor)         # (tot,)
    # shift: anchor + uniform integer jitter in [-radius, radius]
    jit = rng.integers(-radius, radius + 1, size=(tot, nsh), dtype=np.int64)
    shift = a_shift[rep] + jit
    # dir: keep anchor dir, or (prob dir_switch) replace with a random pool dir
    dirv = a_dir[rep].copy()
    swap = rng.random(tot) < dir_switch
    if swap.any():
        pick = rng.integers(0, ndir, size=int(swap.sum()))
        dirv[swap] = dir_pool[pick]
    # z: keep anchor z, or resample from extended pool
    if z_mode == "anchor":
        zn = a_zn[rep].copy(); zd = a_zd[rep].copy()
    else:
        zi = rng.integers(0, len(ext_zpool), size=tot)
        zn = ext_zpool[zi, 0].copy(); zd = ext_zpool[zi, 1].copy()
    return shift, dirv, zn, zd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True, help="phase3/survivor JSONL")
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--n-anchors", type=int, default=1000)
    ap.add_argument("--per-anchor", type=int, default=10000)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--dir-switch", type=float, default=0.15)
    ap.add_argument("--z-mode", choices=["anchor", "extended"], default="extended")
    ap.add_argument("--z-min", type=float, default=0.02)
    ap.add_argument("--z-max", type=float, default=0.999)
    ap.add_argument("--nsteps", type=int, default=120)
    ap.add_argument("--thresh", type=float, default=0.0, help="keep r2 >= thresh")
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ext_zpool = np.array(cg.z_pool(z_max=args.z_max, z_min=args.z_min), dtype=np.int64)
    print(f"[cfg] dim={args.dim} anchors={args.n_anchors} per_anchor={args.per_anchor} "
          f"radius=+-{args.radius} dir_switch={args.dir_switch} z_mode={args.z_mode} "
          f"ext_z={len(ext_zpool)} thresh={args.thresh}")

    anchors = load_anchors(args.anchors, args.dim, args.n_anchors)
    if not anchors:
        print("[error] no anchors loaded"); sys.exit(1)

    shift, dirv, zn, zd = gen_neighborhood(
        anchors, args.per_anchor, args.radius, args.dir_switch,
        args.z_mode, args.dim, ext_zpool, rng)
    tot = shift.shape[0]
    print(f"[gen] {tot:,} candidate trajectories generated")

    kept = 0
    t0 = time.time()
    seen = set()
    with open(args.out, "w") as out:
        for b0 in range(0, tot, args.batch):
            b1 = min(b0 + args.batch, tot)
            if args.backend == "torch":
                r2, lam1 = torch_proxy(shift[b0:b1], dirv[b0:b1], zn[b0:b1],
                                       zd[b0:b1], args.dim, args.nsteps,
                                       args.device, args.dtype)
            else:
                s = shift[b0:b1].astype(np.float64)
                d = dirv[b0:b1].astype(np.float64)
                n = zn[b0:b1].astype(np.float64)
                dd = zd[b0:b1].astype(np.float64)
                lam = lyapunov_batch(s, d, n, dd, args.dim, args.nsteps, np)
                r = ratios_batch(lam, np)
                r2 = r[:, 0]; lam1 = lam[:, 0]
            keep = np.isfinite(r2) & (r2 >= args.thresh)
            idx = np.nonzero(keep)[0]
            for j in idx:
                sh = shift[b0 + j].tolist(); dv = dirv[b0 + j].tolist()
                zk = (int(zn[b0 + j]), int(zd[b0 + j]))
                key = (tuple(sh), tuple(dv), zk)
                if key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps({
                    "shift": sh, "dir": dv,
                    "z_num": zk[0], "z_den": zk[1],
                    "r2": float(r2[j]), "lam1": float(lam1[j])}) + "\n")
                kept += 1
            el = time.time() - t0
            print(f"[proxy] {b1:,}/{tot:,}  kept={kept:,} (unique)  "
                  f"{b1/el:,.0f} traj/s ETA {(tot-b1)/max(b1/el,1):.0f}s", flush=True)

    print(f"[done] scanned {tot:,}  kept {kept:,} unique survivors (r2>={args.thresh}) "
          f"-> {args.out}  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
