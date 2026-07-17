#!/usr/bin/env python3
"""
jstar_estimator.py
===================
Practical estimation of the first visible error mode J* for CMF trajectories.

Three estimation strategies:
  1. Full-ladder screening (blind): keep candidates if any stable r_k > 0
  2. Post-hoc J* identification: given exact delta, find which r_k matches
  3. Singular-vector visibility: approximate visibility determinant from
     QR-decomposed singular vectors

The full-ladder approach is the production screening method: it is conservative
and catches hidden-positive regimes where r_2 is negative but r_3+ is positive.

Usage:
  # Blind screening: full-ladder on a batch
  python3 jstar_estimator.py --mode blind --dim 6 --n 1000 --depth 120 --box 6

  # Post-hoc: identify J* for a known trajectory
  python3 jstar_estimator.py --mode posthoc --dim 6 --depth 120 \
      --shift '[-2,-2,0,0,-2,0,-2,-2,-2,0,-2]' \
      --dir '[0,0,0,0,0,0,0,0,0,1,0]' --z-num 7 --z-den 20

  # Singular-vector visibility
  python3 jstar_estimator.py --mode svvis --dim 6 --depth 120 \
      --shift '[-2,-2,0,0,-2,0,-2,-2,-2,0,-2]' \
      --dir '[0,0,0,0,0,0,0,0,0,1,0]' --z-num 7 --z-den 20
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmf_generic as cg
from spectral_delta import lyapunov_spectrum


def full_ladder(shift, dirv, zn, zd, dim, N):
    """Compute the full ratio ladder r_k = -lambda_k/lambda_1 for k=2..dim.
    Returns (ratios, lambdas, jstar_blind_estimate)."""
    lam = lyapunov_spectrum(
        lambda n: cg.build_M_float(n+1, shift, dirv, zn, zd, dim), dim, N)
    if lam[0] == 0 or not np.isfinite(lam[0]):
        return None, lam, None
    ratios = -lam[1:] / lam[0]  # r_2, r_3, ..., r_dim
    # Blind J* estimate: first non-degenerate ratio (|r_k| < 1.05)
    jstar = None
    for k, r in enumerate(ratios):
        if abs(r) < 1.05 and np.isfinite(r):
            jstar = k + 2  # k=0 -> r_2, k=1 -> r_3, etc.
            break
    return ratios, lam, jstar


def blind_screen(shift, dirv, zn, zd, dim, N, thresh=0.0):
    """Conservative blind screening: keep if ANY stable r_k > thresh.
    Returns (should_keep, best_rk, best_k, all_ratios)."""
    ratios, lam, _ = full_ladder(shift, dirv, zn, zd, dim, N)
    if ratios is None:
        return False, float("nan"), None, None
    best_rk = float("nan")
    best_k = None
    for k, r in enumerate(ratios):
        if abs(r) < 1.05 and np.isfinite(r) and r > thresh:
            if np.isnan(best_rk) or r > best_rk:
                best_rk = float(r)
                best_k = k + 2
    return best_k is not None, best_rk, best_k, ratios


def posthoc_jstar(shift, dirv, zn, zd, dim, N):
    """Given a trajectory, compute exact delta and identify which r_k matches.
    Returns (exact_delta, jstar, ratios, lambdas)."""
    ratios, lam, _ = full_ladder(shift, dirv, zn, zd, dim, N)
    if ratios is None:
        return None, None, None, lam
    # Compute exact delta
    exact_d = cg.independent_delta(shift, dirv, zn, zd, N, dim)
    if exact_d is None:
        return None, None, ratios, lam
    # Find which r_k is closest to exact delta
    best_k = None
    best_diff = float("inf")
    for k, r in enumerate(ratios):
        if abs(r) < 1.05 and np.isfinite(r):
            diff = abs(r - exact_d)
            if diff < best_diff:
                best_diff = diff
                best_k = k + 2
    return exact_d, best_k, ratios, lam


def sv_visibility(shift, dirv, zn, zd, dim, N):
    """Approximate visibility determinant from QR-decomposed singular vectors.
    Computes the cumulative product P_n, then extracts left/right singular vectors
    and evaluates the visibility determinant V_{a,j} for the standard readout
    (phi = last row, psi = last column)."""
    # Build cumulative product with QR tracking
    Q = np.eye(dim)
    logs = np.zeros(dim)
    Q_history = []
    for n in range(N):
        M = cg.build_M_float(n+1, shift, dirv, zn, zd, dim)
        Y = M @ Q
        Q_new, R = np.linalg.qr(Y)
        diag = np.diag(R).copy()
        signs = np.sign(diag); signs[signs == 0] = 1.0
        S = np.diag(signs)
        Q = Q_new @ S
        logs += np.log(np.maximum(np.abs(np.diag(R)), 1e-300))
        if n == N - 1:
            Q_history.append(Q.copy())
    lam = logs / N

    # The final Q matrix columns approximate the Oseledets splitting
    # u_i ~ Q[:, i] (right singular vectors of P_N)
    # Standard readout: phi(v) = v[dim-1] (last component -> numerator)
    #                    psi(v) = v[dim-1] (last component -> denominator)
    # For companion matrices, the readout is: p_n = P_n[last_row, last_col]
    #                                          q_n = P_n[last_row, last_col]
    # Actually the readout is row-pair based. We use the standard:
    # phi = e_{dim-1}^T (selects last row), psi = e_{dim-1}^T
    # For the companion system, phi and psi are different row functionals.
    # We approximate: a = first mode where psi(u_i) != 0
    #                 j* = first visible mode after a

    # Use the Q columns as approximate modes
    modes = Q  # (dim, dim), columns are approximate u_i

    # Standard readout: numerator = row i, denominator = row j
    # For companion: last column entries are the convergents
    # phi(u) = u[dim-1] (last component), psi(u) = u[dim-1]
    # But we need DIFFERENT functionals. Use:
    # phi = projection onto row 0 of P_N, psi = projection onto row 1
    # This is a simplification; the real visibility depends on the readout pair.

    # Simplified: use last-component as denominator, first-component as numerator
    phi = np.zeros(dim); phi[0] = 1.0  # numerator functional
    psi = np.zeros(dim); psi[dim-1] = 1.0  # denominator functional

    # Evaluate on modes
    phi_vals = phi @ modes  # phi(u_i) for each i
    psi_vals = psi @ modes  # psi(u_i) for each i

    # Denominator mode: first i where psi(u_i) is significant
    psi_thresh = 1e-10 * max(np.abs(psi_vals).max(), 1e-300)
    a = None
    for i in range(dim):
        if abs(psi_vals[i]) > psi_thresh:
            a = i
            break
    if a is None:
        a = 0  # fallback

    # Visibility determinant V_{a,j} = phi(u_j)*psi(u_a) - phi(u_a)*psi(u_j)
    # J* = first j > a with V_{a,j} != 0
    jstar = None
    for j in range(a+1, dim):
        V = phi_vals[j] * psi_vals[a] - phi_vals[a] * psi_vals[j]
        vis_thresh = 1e-10 * max(abs(phi_vals[a] * psi_vals[a]), 1e-300)
        if abs(V) > vis_thresh:
            jstar = j + 1  # 1-indexed mode number
            break

    return {
        "lambda": lam.tolist(),
        "ratios": (-lam[1:]/lam[0]).tolist(),
        "denominator_mode_a": a + 1,
        "jstar_sv": jstar,
        "phi_vals": phi_vals.tolist(),
        "psi_vals": psi_vals.tolist(),
        "visibility_determinants": [
            float(phi_vals[j]*psi_vals[a] - phi_vals[a]*psi_vals[j])
            for j in range(dim)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["blind", "posthoc", "svvis"], default="blind")
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--depth", type=int, default=120)
    ap.add_argument("--box", type=int, default=6)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--thresh", type=float, default=0.0)
    ap.add_argument("--shift", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--z-num", type=int, default=None)
    ap.add_argument("--z-den", type=int, default=None)
    args = ap.parse_args()

    if args.mode == "blind":
        # Batch blind screening
        from params import gen_params, build_pools
        nsh = cg.nshift_for(args.dim)
        dirs, zs = build_pools(args.dim, z_max=1.0)
        ndir, nz = len(dirs), len(zs)
        gids = np.arange(0, args.n, dtype=np.uint64)
        shift, di, zi = gen_params(20260626, gids, nsh, args.box, ndir, nz)

        kept = 0
        jstar_dist = {}
        for k in range(args.n):
            sh = shift[k].tolist()
            dv = dirs[di[k]].tolist()
            zn, zd = int(zs[zi[k], 0]), int(zs[zi[k], 1])
            keep, best_r, best_k, ratios = blind_screen(
                sh, dv, zn, zd, args.dim, args.depth, args.thresh)
            if keep:
                kept += 1
                jstar_dist[best_k] = jstar_dist.get(best_k, 0) + 1

        print(f"Blind screening: {kept}/{args.n} kept (thresh={args.thresh})")
        print(f"J* distribution among kept: {jstar_dist}")
        print(f"  (2=standard, 3+=hidden-positive rescue)")

    elif args.mode == "posthoc":
        shift = json.loads(args.shift)
        dirv = json.loads(args.dir)
        exact_d, jstar, ratios, lam = posthoc_jstar(
            shift, dirv, args.z_num, args.z_den, args.dim, args.depth)
        print(f"Exact delta: {exact_d}")
        print(f"Lambda: {lam}")
        print(f"Ratios r_k: {ratios}")
        print(f"Post-hoc J* = {jstar}  (r_{jstar} = {ratios[jstar-2]:.6f})")
        print(f"  Exact delta matches r_{jstar} within {abs(ratios[jstar-2]-exact_d):.6f}")

    elif args.mode == "svvis":
        shift = json.loads(args.shift)
        dirv = json.loads(args.dir)
        result = sv_visibility(
            shift, dirv, args.z_num, args.z_den, args.dim, args.depth)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
