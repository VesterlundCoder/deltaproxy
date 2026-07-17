"""
pfq_extract.py
==============
Extract the EXACT RamanujanTools pFq trajectory-step matrix M(n) symbolically,
with symbolic start (x_i, y_j), symbolic trajectory steps (tx_i, ty_j), symbolic
z and step variable n. This is the universal per-step builder we will translate
into a vectorized numpy/torch kernel for the LUMI GPU sweep, guaranteeing we run
the same object RamanujanTools defines.
"""
import sys, time
import sympy as sp
from sympy.abc import n as nsym
from ramanujantools import Position
from ramanujantools.cmf import pFq


def extract(p, q, symbolic_traj=True):
    z = sp.Symbol("z")
    xs = sp.symbols(f"x:{p}")
    ys = sp.symbols(f"y:{q}")
    if symbolic_traj:
        txs = sp.symbols(f"tx:{p}")
        tys = sp.symbols(f"ty:{q}")
    else:
        txs = [1] * p
        tys = [1] * q
    start = Position({**{xs[i]: xs[i] for i in range(p)},
                      **{ys[j]: ys[j] for j in range(q)}})
    traj = Position({**{xs[i]: txs[i] for i in range(p)},
                     **{ys[j]: tys[j] for j in range(q)}})
    cmf = pFq(p, q, z)
    t0 = time.time()
    M = cmf._trajectory_matrix_inner(traj, start, nsym)
    build = time.time() - t0
    R, C = M.rows(), M.cols()
    print(f"\n===== pFq({p},{q})  symbolic_traj={symbolic_traj}  "
          f"[{R}x{C}, built in {build:.1f}s] =====", flush=True)
    entries = {}
    for i in range(R):
        for j in range(C):
            e = sp.simplify(M[i, j].factor())
            entries[(i, j)] = e
            print(f"  M[{i},{j}] = {e}", flush=True)
    return entries


if __name__ == "__main__":
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    st = (sys.argv[2] == "sym") if len(sys.argv) > 2 else False
    extract(dim, dim - 1, symbolic_traj=st)
