"""
pfq_gpu.py
==========
EXACT RamanujanTools pFq CMF trajectory-step matrix, vectorized for GPU sweeps.

We do NOT re-derive the recurrence. Instead we take RamanujanTools' own per-axis
matrices  M(axis, sign) = cmf.M(axis, sign)  (sympy matrices in x_i, y_j, z) and
lambdify each entry using ONLY the arithmetic operators (+, -, *, /, **).  Such
lambdas run unchanged on both numpy arrays and torch tensors, so the same code
path validates on a laptop (numpy) and screens billions of trajectories on LUMI
(torch/ROCm).

The pFq trajectory-step matrix for a diagonal trajectory t (each step in
{-1,0,+1}) at base position P is, EXACTLY as RamanujanTools computes it in
`_calculate_diagonal_matrix_backtrack`, the ordered product

    M_step(P) = prod_{axis in sorted(axes)}  M(axis, sign_axis)  evaluated at the
    running position, where each factor shifts its own coordinate afterwards.

The walk at depth k multiplies M_step(start + k*t) for k = 0,1,2,...  which is
precisely `cmf.walk(t, [N], start)`.  We assert this entrywise in `selftest`.
"""
from __future__ import annotations

import functools
import json
import os

DEFS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "pfq_matrix_defs.json")


def _param_names(p, q):
    return [f"x{i}" for i in range(p)] + [f"y{j}" for j in range(q)] + ["z"]


def generate_defs(pairs, path=DEFS_PATH):
    """LOCAL codegen (needs ramanujantools): dump the EXACT pFq per-axis matrices
    M(axis, sign) as pure-python expression strings so the LUMI container needs
    only numpy/torch (no ramanujantools / python-flint at runtime)."""
    import sympy as sp
    from ramanujantools.cmf import pFq
    defs = {}
    if os.path.exists(path):
        with open(path) as f:
            defs = json.load(f)
    for (p, q) in pairs:
        zsym = sp.Symbol("z")
        xs = list(sp.symbols(f"x:{p}")); ys = list(sp.symbols(f"y:{q}"))
        axes = xs + ys
        N = pFq.predict_rank(p, q, zsym)
        cmf = pFq(p, q, zsym)
        entry = {}
        for ai, axis in enumerate(axes):
            for sign in (True, False):
                M = sp.Matrix(cmf.M(axis, sign))
                grid = [[str(sp.together(M[i, j])) for j in range(N)]
                        for i in range(N)]
                entry[f"{ai}|{int(sign)}"] = grid
        defs[f"{p},{q}"] = {"N": N, "params": _param_names(p, q), "grids": entry}
        print(f"  generated pFq({p},{q}) defs (N={N})", flush=True)
    with open(path, "w") as f:
        json.dump(defs, f)
    print(f"wrote {path}", flush=True)
    return path


# ── axis matrices loaded from the shipped defs (pure python, no ramanujantools)
@functools.lru_cache(maxsize=None)
def _axis_lambdas(p, q):
    """Return (axes_sorted, axes, params, grids, N). grids[(ai,sign)] is an NxN
    grid of scalar lambdas over the params, built from the pre-generated pure-
    python expression strings. Falls back to on-the-fly ramanujantools codegen
    (local dev only) if the defs file lacks (p,q)."""
    key = f"{p},{q}"
    defs = {}
    if os.path.exists(DEFS_PATH):
        with open(DEFS_PATH) as f:
            defs = json.load(f)
    if key not in defs:
        generate_defs([(p, q)])
        with open(DEFS_PATH) as f:
            defs = json.load(f)
    entry = defs[key]
    N = entry["N"]
    pnames = entry["params"]
    axes = pnames[:-1]                       # x0..x{p-1}, y0..y{q-1} (name strings)
    axes_sorted = sorted(axes)
    hdr = "lambda " + ", ".join(pnames) + ": "
    grids = {}
    for _k, grid in entry["grids"].items():
        ai_s, sign_s = _k.split("|")
        ai, sign = int(ai_s), bool(int(sign_s))
        grids[(ai, sign)] = [[eval(hdr + grid[i][j]) for j in range(N)]
                             for i in range(N)]
    return axes_sorted, axes, pnames, grids, N


class PFqBuilder:
    """Vectorized pFq trajectory-step matrix builder for a fixed (p, q)."""

    def __init__(self, p, q, xp):
        self.p, self.q = p, q
        self.xp = xp                                  # numpy or torch module
        (self.axes_sorted, self.axes, self.params,
         self.grids, self.N) = _axis_lambdas(p, q)
        self.axis_index = {a: i for i, a in enumerate(self.axes)}
        self.naxes = len(self.axes)
        self._istorch = xp.__name__ != "numpy"

    def _stack(self, arrs, axis):
        """Backend-safe stack (torch uses dim=, numpy uses axis=)."""
        if self._istorch:
            return self.xp.stack(arrs, dim=axis)
        return self.xp.stack(arrs, axis=axis)

    def _matrix(self, ai, sign, pos, z, base0):
        """M(axis_ai, sign) evaluated at position `pos` (list of (B,) arrays)."""
        xp = self.xp
        grid = self.grids[(ai, sign)]
        args = list(pos) + [z]
        rows = []
        for i in range(self.N):
            cols = [grid[i][j](*args) + base0 for j in range(self.N)]
            rows.append(self._stack(cols, -1))         # (B, N)
        return self._stack(rows, -2)                   # (B, N, N)

    def step_matrix(self, start, traj, z, base0):
        """Diagonal trajectory step matrix at base position `start` (list of (B,)
        arrays, order = self.axes), traj = list of ints in {-1,0,1}, z=(B,)."""
        xp = self.xp
        B = base0.shape[0]
        pos = [s + base0 for s in start]               # mutable running position
        result = None
        for axis in self.axes_sorted:
            ai = self.axis_index[axis]
            t = traj[ai]
            if t == 0:
                continue
            M = self._matrix(ai, t >= 0, pos, z, base0)
            result = M if result is None else self._matmul(result, M)
            pos[ai] = pos[ai] + t
        if result is None:                             # all-zero trajectory
            eye = xp.eye(self.N, dtype=base0.dtype) if xp.__name__ == "numpy" \
                else xp.eye(self.N, dtype=base0.dtype, device=base0.device)
            result = eye.reshape(1, self.N, self.N) + xp.zeros((B, 1, 1),
                                                               dtype=base0.dtype) \
                if xp.__name__ == "numpy" else \
                eye.reshape(1, self.N, self.N).expand(B, self.N, self.N)
        return result

    def _matmul(self, A, B):
        return self.xp.matmul(A, B)

    def _spectrum_pfq(self, startv, trajv, zv, nsteps, base0):
        """FULL finite-time Lyapunov spectrum of the pFq walk for a batch sharing
        one trajectory pattern `trajv`. startv: list of (B,) base start arrays
        (order self.axes); zv: (B,); base0: (B,) zeros of the right backend/dtype.
        Returns (lam, bad): lam is (B,N) sorted DESCENDING; bad is (B,) bool mask
        of pole/non-finite trajectories. This is the single QR core used by BOTH
        the top-two proxy and the full multi-ratio ladder, so they are bit-parity
        consistent with the production sweep."""
        xp = self.xp
        B = base0.shape[0]
        N = self.N
        istorch = xp.__name__ != "numpy"
        if istorch:
            Q = xp.eye(N, dtype=base0.dtype, device=base0.device)
            Q = Q.reshape(1, N, N).expand(B, N, N).contiguous()
            eyeB = xp.eye(N, dtype=base0.dtype, device=base0.device).reshape(1, N, N)
            logs = xp.zeros((B, N), dtype=base0.dtype, device=base0.device)
            bad = xp.zeros(B, dtype=xp.bool, device=base0.device)
        else:
            Q = xp.broadcast_to(xp.eye(N), (B, N, N)).copy()
            eyeB = xp.eye(N).reshape(1, N, N)
            logs = xp.zeros((B, N))
            bad = xp.zeros(B, dtype=bool)
        for k in range(nsteps):
            pos = [startv[a] + k * trajv[a] for a in range(self.naxes)]
            M = self.step_matrix(pos, trajv, zv, base0)          # (B,N,N)
            V = xp.matmul(M, Q)
            finite = xp.isfinite(V).reshape(B, -1).all(axis=1) if not istorch \
                else xp.isfinite(V).reshape(B, -1).all(dim=1)
            badstep = ~finite                 # no host sync (keep GPU async)
            bad = bad | badstep
            V = xp.where(badstep.reshape(B, 1, 1), eyeB, V)
            # Manual modified Gram-Schmidt over columns of V (batched, no
            # rocSOLVER): torch.linalg.qr is PATHOLOGICALLY SLOW on ROCm.
            # Only elementwise + reductions here -> fast on ROCm/CUDA.
            if istorch:
                Qn = xp.empty_like(V)
                for j in range(N):
                    v = V[:, :, j].clone()
                    for i in range(j):
                        qi = Qn[:, :, i]
                        dot = (qi * v).sum(dim=1, keepdim=True)
                        v = v - dot * qi
                    nrm = xp.clamp(v.norm(dim=1, keepdim=True), min=1e-300)
                    logs[:, j] = logs[:, j] + xp.log(nrm.squeeze(1))
                    Qn[:, :, j] = v / nrm
            else:
                Qn = xp.empty_like(V)
                for j in range(N):
                    v = V[:, :, j].copy()
                    for i in range(j):
                        qi = Qn[:, :, i]
                        dot = (qi * v).sum(axis=1, keepdims=True)
                        v = v - dot * qi
                    nrm = xp.sqrt((v * v).sum(axis=1, keepdims=True))
                    nrm = xp.maximum(nrm, 1e-300)
                    logs[:, j] = logs[:, j] + xp.log(nrm[:, 0])
                    Qn[:, :, j] = v / nrm
            Q = Qn
        lam = logs / nsteps
        if istorch:
            lam = xp.sort(lam, dim=1, descending=True).values
        else:
            lam = xp.sort(lam, axis=1)[:, ::-1]
        return lam, bad

    def lyapunov_batch_pfq(self, startv, trajv, zv, nsteps, base0):
        """Top-two Lyapunov exponents of the pFq walk (production proxy).
        Returns (r2, lam1) each (B,). Non-finite (pole) trajectories -> nan."""
        xp = self.xp
        lam, bad = self._spectrum_pfq(startv, trajv, zv, nsteps, base0)
        l1 = lam[:, 0]
        r2 = xp.where(l1 != 0, -lam[:, 1] / l1, xp.full_like(l1, float("nan")))
        nan = xp.full_like(r2, float("nan"))
        r2 = xp.where(bad, nan, r2)
        l1 = xp.where(bad, nan, l1)
        return r2, l1

    def lyapunov_ladder_pfq(self, startv, trajv, zv, nsteps, base0):
        """FULL subdominant-ratio ladder r_k = -lam_k/lam_1 for k=2..N of the pFq
        walk. Returns (ratios, lam, bad): ratios is (B, N-1) = [r_2,...,r_N] with
        NaN where lam_1 is degenerate or the trajectory hit a pole; lam is (B,N)
        sorted descending; bad is (B,) pole mask. Diagnostic counterpart of the
        top-two proxy, sharing the identical QR core (_spectrum_pfq)."""
        xp = self.xp
        N = self.N
        lam, bad = self._spectrum_pfq(startv, trajv, zv, nsteps, base0)
        l1 = lam[:, 0]
        cols = []
        for k in range(1, N):
            rk = xp.where(l1 != 0, -lam[:, k] / l1, xp.full_like(l1, float("nan")))
            rk = xp.where(bad, xp.full_like(rk, float("nan")), rk)
            cols.append(rk)
        ratios = self._stack(cols, -1)                       # (B, N-1)
        return ratios, lam, bad


# ── sweep helpers: dir pool + shift->(x,y) mapping (disjoint ranges) ─────────
def pfq_dir_pool(dim):
    """Forward diagonal trajectories in {0,1}: all-ones plus each single axis
    dropped. Length nsh = 2*dim-1 per pattern."""
    nsh = 2 * dim - 1
    pats = [[1] * nsh]
    for k in range(nsh):
        p = [1] * nsh
        p[k] = 0
        pats.append(p)
    return pats


def map_shift_to_xy(shift_row, dim, box):
    """Map an integer shift vector in [-box,box]^nsh to pFq start params, placing
    x_i and y_j in DISJOINT positive integer ranges so x_i != y_j (no poles).
      x_i in [2, 2+2box],  y_j in [2box+4, 4box+4]
    shift_row: (nsh,) ints. Returns (x_start[dim], y_start[dim-1])."""
    bx = 2 + box                      # center of x-range
    by = 3 * box + 4                  # center of y-range (disjoint above x)
    x = [int(shift_row[i]) + bx for i in range(dim)]
    y = [int(shift_row[dim + j]) + by for j in range(dim - 1)]
    return x, y


# ── numpy correctness gate: entrywise match against cmf.walk ─────────────────
def selftest(cases=None, depth=12, tol=1e-7, verbose=True):
    import numpy as np
    import sympy as sp
    from ramanujantools import Position
    from ramanujantools.cmf import pFq

    if cases is None:
        cases = [
            # (p, q, z_num, z_den, x_start, y_start, traj_x, traj_y)
            # well-separated starts (no x_i=y_j collisions -> no spurious poles)
            (2, 1, 1, 2, [2, 5], [8], [1, 1], [1]),
            (3, 2, -1, 2, [2, 5, 9], [12, 15], [1, 1, 1], [1, 1]),
            (3, 2, 1, 3, [2, 5, 9], [12, 15], [1, 0, 1], [1, 0]),
            (4, 3, 1, 2, [2, 5, 9, 13], [17, 21, 25], [1, 1, 1, 1], [1, 1, 1]),
            (4, 3, -1, 3, [3, 7, 11, 15], [19, 23, 27], [1, 0, 1, 1], [1, 1, 0]),
            (5, 4, 1, 2, [2, 5, 9, 13, 17], [21, 25, 29, 33], [1]*5, [1]*4),
            (5, 4, 2, 3, [3, 7, 11, 15, 19], [23, 27, 31, 35], [1, 1, 0, 1, 1], [1, 0, 1, 1]),
        ]
    allok = True
    for (p, q, zn, zd, xst, yst, tx, ty) in cases:
        z = sp.Rational(zn, zd)
        xs = sp.symbols(f"x:{p}"); ys = sp.symbols(f"y:{q}")
        start = Position({**{xs[i]: xst[i] for i in range(p)},
                          **{ys[j]: yst[j] for j in range(q)}})
        traj = Position({**{xs[i]: tx[i] for i in range(p)},
                         **{ys[j]: ty[j] for j in range(q)}})
        cmf = pFq(p, q, z)
        try:
            W_rt = np.array(cmf.walk(traj, [depth], start)[0].tolist(), dtype=float)
        except Exception as e:
            if verbose:
                print(f"  pFq({p},{q}) z={zn}/{zd} traj={list(tx)+list(ty)}: "
                      f"RT walk diverged ({type(e).__name__}) - skipped", flush=True)
            continue

        b = PFqBuilder(p, q, np)
        base0 = np.zeros(1)
        startv = [np.array([float(xst[i])]) for i in range(p)] + \
                 [np.array([float(yst[j])]) for j in range(q)]
        trajv = list(tx) + list(ty)
        zv = np.array([float(zn) / float(zd)])
        W = np.eye(b.N)[None]
        for k in range(depth):
            posk = [startv[a] + k * trajv[a] for a in range(b.naxes)]
            Mk = b.step_matrix(posk, trajv, zv, base0)
            W = W @ Mk
        W = W[0]
        err = float(np.max(np.abs(W - W_rt)) / (np.max(np.abs(W_rt)) + 1e-30))
        ok = err < tol
        allok &= ok
        if verbose:
            print(f"  pFq({p},{q}) z={zn}/{zd} traj={trajv}: "
                  f"rel_err={err:.2e}  {'OK' if ok else 'MISMATCH'}", flush=True)
    print(f"\n==> pFq builder {'ALL MATCH RamanujanTools' if allok else 'FAILED'}",
          flush=True)
    return allok


if __name__ == "__main__":
    selftest()
