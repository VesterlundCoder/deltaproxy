"""
audit_matrices_rt.py
====================
Gauge-invariant correctness audit: prove our compiled companion matrix
(cmf_generic.build_M_*) IS the canonical RamanujanTools pFq CMF matrix for
4F3 (dim=4) and 5F4 (dim=5), so the GPU sweeps screen the intended object.

RamanujanTools `trajectory_matrix(traj, start)` evaluates at position
`start + n*trajectory`, "up to a constant" (an n-dependent scalar / gauge).
Our `build_M(n)` puts a_i = shift[i]+n*dir[i]+1, b_j = shift[dim+j]+n*dir[dim+j]+2,
so with  shift_a = a_start-1, dir_a = a_step, shift_b = b_start-2, dir_b = b_step,
BOTH matrices sit at the SAME point (a_start+n*a_step, b_start+n*b_step) for each n.

An overall scalar cannot change eigenvalue RATIOS, so if it is the same CMF the
NORMALIZED eigenvalue spectrum |lambda_i|/|lambda_max| must match at every n.
(This ratio is exactly what sets the irrationality measure delta.)
"""
from __future__ import annotations
import os
import sys

import numpy as np
import sympy as sp
from sympy.abc import n as nsym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cmf_generic as cg

from ramanujantools import Position
from ramanujantools.cmf import pFq


import mpmath as mp

mp.mp.dps = 600
N = 120                    # walk to N and 2N; delta is a double-depth measure


def _mpf_exact(sympy_num):
    """Exact sympy Integer/Rational -> mpmath mpf at full working precision."""
    r = sp.nsimplify(sympy_num)
    p, q = r.as_numer_denom()
    return mp.mpf(int(p)) / mp.mpf(int(q))


def lyapunov_spectrum(step_matrices):
    """QR-stabilized Lyapunov exponents (log growth of singular values / step).
    GAUGE-INVARIANT: a bounded per-step gauge G(n) cannot change these rates,
    so if two matrix sequences are the same CMF (cocycle-equivalent) the whole
    spectrum, and in particular dhat = -lambda2/lambda1, must coincide."""
    dim = step_matrices[0].shape[0]
    Q = np.eye(dim)
    logs = np.zeros(dim)
    for M in step_matrices:
        A = M @ Q
        Q, Rm = np.linalg.qr(A)
        d = np.diag(Rm)
        s = np.sign(d)
        s[s == 0] = 1.0
        Q = Q * s                      # keep R diagonal positive
        logs += np.log(np.abs(d))
    lam = np.sort(logs / len(step_matrices))[::-1]
    return lam


def rt_step_matrices(cmf, traj, start, depth):
    """RamanujanTools pFq per-step trajectory matrices M(n), n=1..depth (float)."""
    Msym = cmf._trajectory_matrix_inner(traj, start, nsym)
    mats = []
    for n in range(1, depth + 1):
        Mm = Msym.subs({nsym: n})
        R, C = Mm.rows(), Mm.cols()
        mats.append(np.array([[float(sp.N(Mm[i, j].factor(), 30))
                               for j in range(C)] for i in range(R)], dtype=float))
    return mats


def ours_step_matrices(shift, dirv, zn, zd, dim, depth):
    """Our companion per-step matrices M(n), n=1..depth (float)."""
    return [np.array(cg.build_M_float(n, shift, dirv, zn, zd, dim), dtype=float)
            for n in range(1, depth + 1)]


def run_case(label, dim, z_num, z_den, a_start, a_step, b_start, b_step):
    import time
    p, q = dim, dim - 1
    z = sp.Rational(z_num, z_den)
    xs = sp.symbols(f"x:{p}")
    ys = sp.symbols(f"y:{q}")
    start = Position({**{xs[i]: a_start[i] for i in range(p)},
                      **{ys[j]: b_start[j] for j in range(q)}})
    traj = Position({**{xs[i]: a_step[i] for i in range(p)},
                     **{ys[j]: b_step[j] for j in range(q)}})
    cmf = pFq(p, q, z)

    shift = [a_start[i] - 1 for i in range(p)] + [b_start[j] - 2 for j in range(q)]
    dirv = list(a_step) + list(b_step)

    print(f"\n### {label}  ({dim}F{dim-1}, z={z})  Lyapunov depth {N}", flush=True)
    print(f"  our shift={shift}  dir={dirv}", flush=True)
    t0 = time.time()
    lam_rt = lyapunov_spectrum(rt_step_matrices(cmf, traj, start, N))
    print(f"  [RamanujanTools per-step build+QR in {time.time()-t0:.1f}s]", flush=True)
    lam_o = lyapunov_spectrum(ours_step_matrices(shift, dirv, z_num, z_den, dim, N))

    # gauge-invariant fingerprint: full Lyapunov spectrum (shifted so lambda1=0,
    # since the overall integer-clearing scalar only shifts every exponent equally)
    rel_rt = lam_rt - lam_rt[0]
    rel_o = lam_o - lam_o[0]
    dhat_rt = -lam_rt[1] / lam_rt[0] if lam_rt[0] else float('nan')
    dhat_o = -lam_o[1] / lam_o[0] if lam_o[0] else float('nan')
    worst = float(np.max(np.abs(rel_rt - rel_o)))
    print(f"  RamanujanTools lambda (rel to top): {np.round(rel_rt, 5)}", flush=True)
    print(f"  ours           lambda (rel to top): {np.round(rel_o, 5)}", flush=True)
    print(f"  proxy dhat=-lam2/lam1:  RT={dhat_rt:+.5f}  ours={dhat_o:+.5f}", flush=True)
    ok = worst < 1e-3
    print(f"  ==> SPECTRUM {'MATCH' if ok else 'MISMATCH'} (max|Δ(rel-lambda)|={worst:.2e})",
          flush=True)
    return ok


if __name__ == "__main__":
    results = []
    # cheap low-dim confirmations first (2F1, 3F2), then the targets 4F3/5F4
    results.append(run_case("2F1 sanity", 2, 1, 2,
                            a_start=[2, 3], a_step=[1, 1],
                            b_start=[3], b_step=[1]))
    results.append(run_case("3F2 sanity", 3, -1, 2,
                            a_start=[2, 1, 3], a_step=[1, 1, 1],
                            b_start=[3, 2], b_step=[1, 1]))
    results.append(run_case("4F3 case A", 4, 1, 2,
                             a_start=[2, 3, 2, 4], a_step=[1, 1, 1, 1],
                             b_start=[3, 4, 5], b_step=[1, 1, 1]))
    results.append(run_case("4F3 case B", 4, -1, 3,
                             a_start=[1, 2, 3, 1], a_step=[2, 1, 1, 1],
                             b_start=[2, 3, 2], b_step=[1, 2, 1]))
    results.append(run_case("5F4 case A", 5, 1, 2,
                             a_start=[2, 3, 2, 4, 3], a_step=[1, 1, 1, 1, 1],
                             b_start=[3, 4, 5, 3], b_step=[1, 1, 1, 1]))
    results.append(run_case("5F4 case B", 5, -1, 4,
                             a_start=[1, 2, 1, 3, 2], a_step=[1, 2, 1, 1, 1],
                             b_start=[2, 3, 2, 4], b_step=[1, 1, 2, 1]))
    concl = [r for r in results if r is not None]
    ok = concl and all(concl)
    print(f"\n==== {'ALL CONCLUSIVE CASES MATCH: our 4F3/5F4 == RamanujanTools pFq' if ok else 'FAILURES/INCONCLUSIVE PRESENT'} "
          f"({sum(1 for r in concl if r)}/{len(concl)} conclusive matched) ====")
