"""
verify_against_ramanujantools.py
================================
Correctness audit: prove that our compiled companion walk (cmf_generic) computes
the SAME CMF as the canonical RamanujanTools `pFq(p, q, z)` for 4F3 (dim=4) and
5F4 (dim=5), so the GPU sweeps screen the intended matrices.

Method (gauge-invariant): for aligned (a=numerator, b=denominator, z) parameters
along an all-ones trajectory we compare, at matching depths,
  * the convergent LIMIT value  (must be the same hypergeometric constant), and
  * the irrationality measure DELTA (double-depth, max over row pairs).

Alignment (cmf_generic gauge):  f_i = shift[i] + n*dir[i] + 1  (numerator a_i)
                                g_j = shift[dim+j] + n*dir[dim+j] + 2 (denom b_j)
so  shift[i]=a_start[i]-1, dir[i]=a_step[i]; shift[dim+j]=b_start[j]-2,
    dir[dim+j]=b_step[j];  z = z_num/z_den.
"""
from __future__ import annotations
import os
import sys

import sympy as sp
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cmf_generic as cg

from ramanujantools import Position
from ramanujantools.cmf import pFq

mp.mp.dps = 220


def ours_walk_lastcol(shift, dirv, zn, zd, dim, N):
    """Exact-integer walk; return last-column vectors at depth N and 2N."""
    P = [[1 if i == j else 0 for j in range(dim)] for i in range(dim)]
    snapN = None
    for n in range(1, 2 * N + 1):
        P = cg.matmul_int(P, cg.build_M_int(n, shift, dirv, zn, zd, dim), dim)
        if n == N:
            snapN = [P[i][dim - 1] for i in range(dim)]
    return snapN, [P[i][dim - 1] for i in range(dim)]


def best_delta_and_limit(vN, v2N):
    """Max double-depth delta over row pairs + the limit of the best pair."""
    dim = len(vN)
    best_d, best_L = None, None
    for i in range(dim):
        for j in range(dim):
            if i == j or vN[j] == 0 or v2N[j] == 0:
                continue
            LN = mp.mpf(vN[i]) / mp.mpf(vN[j])
            L2 = mp.mpf(v2N[i]) / mp.mpf(v2N[j])
            err = abs(LN - L2)
            if err == 0:
                continue
            lq = mp.log(abs(mp.mpf(vN[j])))
            if lq == 0:
                continue
            d = float(-(1 + mp.log(err) / lq))
            if best_d is None or d > best_d:
                best_d, best_L = d, L2
    return best_d, best_L


def rt_limit_delta(p, q, z, a_start, a_step, b_start, b_step, N):
    """RamanujanTools pFq: limit value (deep) + delta at depth N."""
    n = sp.Symbol("n")
    xs = sp.symbols(f"x:{p}")
    ys = sp.symbols(f"y:{q}")
    start = Position({**{xs[i]: a_start[i] for i in range(p)},
                      **{ys[j]: b_start[j] for j in range(q)}})
    traj = Position({**{xs[i]: a_step[i] for i in range(p)},
                     **{ys[j]: b_step[j] for j in range(q)}})
    cmf = pFq(p, q, sp.Rational(z.numerator, z.denominator))
    # deep limit as the "true" L, then delta at depth N against it
    L = cmf.trajectory_matrix(traj, start).limit({n: 1}, 2 * N, {n: 1}).as_float()
    lim_N = cmf.trajectory_matrix(traj, start).limit({n: 1}, N, {n: 1})
    d = lim_N.delta(mp.mpf(str(L)))
    return mp.mpf(str(L)), float(d)


def run_case(label, dim, z_num, z_den, a_start, a_step, b_start, b_step, N=140):
    p, q = dim, dim - 1
    z = sp.Rational(z_num, z_den)
    shift = [a_start[i] - 1 for i in range(p)] + [b_start[j] - 2 for j in range(q)]
    dirv = list(a_step) + list(b_step)

    vN, v2N = ours_walk_lastcol(shift, dirv, z_num, z_den, dim, N)
    d_ours, L_ours = best_delta_and_limit(vN, v2N)
    L_rt, d_rt = rt_limit_delta(p, q, z, a_start, a_step, b_start, b_step, N)

    print(f"\n### {label}  ({dim}F{dim-1}, z={z})  N={N}")
    print(f"  shift={shift} dir={dirv}")
    print(f"  ours  : delta={d_ours:+.5f}  L={mp.nstr(L_ours, 22)}")
    print(f"  RTools: delta={d_rt:+.5f}  L={mp.nstr(L_rt, 22)}")
    print(f"  delta match: {abs(d_ours - d_rt) < 0.02}")


if __name__ == "__main__":
    # dim4 / 4F3
    run_case("4F3 case A", 4, 1, 2,
             a_start=[2, 3, 2, 4], a_step=[1, 1, 1, 1],
             b_start=[3, 4, 5], b_step=[1, 1, 1])
    run_case("4F3 case B", 4, -1, 3,
             a_start=[1, 2, 3, 1], a_step=[1, 1, 1, 1],
             b_start=[2, 3, 2], b_step=[1, 1, 1])
    # dim5 / 5F4
    run_case("5F4 case A", 5, 1, 2,
             a_start=[2, 3, 2, 4, 3], a_step=[1, 1, 1, 1, 1],
             b_start=[3, 4, 5, 3], b_step=[1, 1, 1, 1])
    run_case("5F4 case B", 5, -1, 4,
             a_start=[1, 2, 1, 3, 2], a_step=[1, 1, 1, 1, 1],
             b_start=[2, 3, 2, 4], b_step=[1, 1, 1, 1])
