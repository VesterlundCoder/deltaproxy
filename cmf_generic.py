"""
Dimension-generic CMF machinery for hypergeometric pF(p-1) companion fields.

The repository uses the right-product convention

    P_N = M_1 M_2 ... M_N.

Exact arithmetic therefore updates ``P = P @ M_n``.  The QR proxy in
``spectral_delta.lyapunov_spectrum`` analyses the same product by propagating
``M_n.T`` because P_N and P_N.T have identical singular values.
"""
from __future__ import annotations

import math
from fractions import Fraction

import mpmath as mp
import numpy as np

from spectral_delta import lyapunov_spectrum


def nshift_for(dim: int) -> int:
    return 2 * dim - 1


def _esym_float(vals):
    n = len(vals)
    e = np.zeros(n + 1, dtype=np.float64)
    e[0] = 1.0
    for v in vals:
        for k in range(n, 0, -1):
            e[k] += v * e[k - 1]
    return e


def _esym_mp(vals):
    n = len(vals)
    e = [mp.mpf(0)] * (n + 1)
    e[0] = mp.mpf(1)
    for v in vals:
        for k in range(n, 0, -1):
            e[k] += v * e[k - 1]
    return e


def _esym_int(vals):
    poly = [1]
    for r in vals:
        new = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new[i] += c
            new[i + 1] += c * r
        poly = new
    return poly


def _roots(n, shift, dirv, dim):
    f = [shift[i] + n * dirv[i] + 1 for i in range(dim)]
    g = [shift[dim + j] + n * dirv[dim + j] + 2 for j in range(dim - 1)]
    return f, g


def build_M_float(n, shift, dirv, z_num, z_den, dim):
    f, g = _roots(n, shift, dirv, dim)
    ef = _esym_float(f)
    eg = np.concatenate([_esym_float(g), [0.0]])
    c = np.zeros(dim, dtype=np.float64)
    c[0] = -float(z_num) * ef[dim]
    for k in range(1, dim):
        c[k] = float(z_den) * eg[dim - k] - float(z_num) * ef[dim - k]
    M = np.zeros((dim, dim), dtype=np.float64)
    for rr in range(1, dim):
        M[rr, rr - 1] = 1.0
    M[:, dim - 1] = c
    return M


def build_M_mp(n, shift, dirv, z_num, z_den, dim):
    f, g = _roots(n, shift, dirv, dim)
    ef = _esym_mp([mp.mpf(x) for x in f])
    eg = _esym_mp([mp.mpf(x) for x in g]) + [mp.mpf(0)]
    c = [mp.mpf(0)] * dim
    c[0] = -mp.mpf(z_num) * ef[dim]
    for k in range(1, dim):
        c[k] = mp.mpf(z_den) * eg[dim - k] - mp.mpf(z_num) * ef[dim - k]
    M = mp.zeros(dim)
    for rr in range(1, dim):
        M[rr, rr - 1] = 1
    for rr in range(dim):
        M[rr, dim - 1] = c[rr]
    return M


def build_M_int(n, shift, dirv, z_num, z_den, dim):
    f, g = _roots(n, shift, dirv, dim)
    ef = _esym_int(f)
    eg = _esym_int(g) + [0]
    c = [0] * dim
    c[0] = -int(z_num) * ef[dim]
    for k in range(1, dim):
        c[k] = int(z_den) * eg[dim - k] - int(z_num) * ef[dim - k]
    M = [[0] * dim for _ in range(dim)]
    for rr in range(1, dim):
        M[rr][rr - 1] = 1
    for rr in range(dim):
        M[rr][dim - 1] = c[rr]
    return M


def matmul_int(A, B, dim):
    C = [[0] * dim for _ in range(dim)]
    for i in range(dim):
        for k in range(dim):
            a = A[i][k]
            if a:
                for j in range(dim):
                    C[i][j] += a * B[k][j]
    return C


def log_bigint(n):
    n = abs(int(n))
    if n == 0:
        return float("-inf")
    b = n.bit_length()
    shift = max(0, b - 53)
    return math.log(n >> shift) + shift * math.log(2.0)


def numerator_zero_steps(shift, dirv, dim, n_start=1, n_end=None):
    """Return numerator-root zero events f_i(n)=0 in the inspected depth range."""
    events = []
    for i in range(dim):
        s = int(shift[i]) + 1
        d = int(dirv[i])
        if d == 0:
            if s == 0:
                events.append((i, "static", None))
            continue
        num = -s
        if num % d != 0:
            continue
        n = num // d
        if n >= n_start and (n_end is None or n <= n_end):
            events.append((i, "dynamic", int(n)))
    return events


def is_degenerate(shift, dirv, dim, n_start=1, n_end=None):
    return bool(numerator_zero_steps(shift, dirv, dim, n_start, n_end))


def spectral_dhat(shift, dirv, z_num, z_den, N, dim):
    lam = lyapunov_spectrum(
        lambda n: build_M_float(n + 1, shift, dirv, z_num, z_den, dim),
        dim, N, product_order="right")
    if lam[0] == 0 or not np.isfinite(lam[0]):
        return float("nan"), lam
    return float(-lam[1] / lam[0]), lam


def spectral_ratios(shift, dirv, z_num, z_den, N, dim):
    lam = lyapunov_spectrum(
        lambda n: build_M_float(n + 1, shift, dirv, z_num, z_den, dim),
        dim, N, product_order="right")
    if lam[0] == 0 or not np.isfinite(lam[0]):
        return np.full(dim - 1, np.nan), lam
    return -lam[1:] / lam[0], lam


def spectral_dhat_robust(shift, dirv, z_num, z_den, N, dim,
                          degen_hi=1.05, min_thresh=None):
    ratios, lam = spectral_ratios(shift, dirv, z_num, z_den, N, dim)
    if ratios.size == 0 or not np.isfinite(ratios[0]):
        return float("nan"), 0, lam, ratios
    r2 = ratios[0]
    meaningful = abs(r2) < degen_hi and (min_thresh is None or r2 >= min_thresh)
    if meaningful:
        return float(r2), 2, lam, ratios
    for k, rk in enumerate(ratios[1:], start=3):
        if np.isfinite(rk) and abs(rk) < degen_hi:
            return float(rk), k, lam, ratios
    return float(r2), 2, lam, ratios


def _pair_delta_from_entries(pn, qn, p2, q2):
    if qn == 0 or q2 == 0:
        return None
    cross = pn * q2 - p2 * qn
    if cross == 0:
        return None
    log_err = log_bigint(cross) - log_bigint(qn) - log_bigint(q2)
    log_q = log_bigint(qn)
    if not math.isfinite(log_q) or log_q == 0:
        return None
    return -(1.0 + log_err / log_q)


def pair_deltas_int(shift, dirv, z_num, z_den, N, dim):
    """Exact double-depth deltas for every fixed ordered row pair."""
    P = [[1 if i == j else 0 for j in range(dim)] for i in range(dim)]
    snapN = None
    for n in range(1, 2 * N + 1):
        P = matmul_int(P, build_M_int(n, shift, dirv, z_num, z_den, dim), dim)
        if n == N:
            snapN = [row[:] for row in P]
    last = dim - 1
    out = {}
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            d = _pair_delta_from_entries(
                snapN[i][last], snapN[j][last], P[i][last], P[j][last])
            if d is not None:
                out[(i, j)] = d
    return out


def independent_delta(shift, dirv, z_num, z_den, N, dim,
                      row_pair=None, return_pair=False):
    """Pure-integer double-depth delta for a fixed pair or delta_O^max."""
    values = pair_deltas_int(shift, dirv, z_num, z_den, N, dim)
    if row_pair is not None:
        pair = tuple(row_pair)
        value = values.get(pair)
        return (value, pair) if return_pair else value
    if not values:
        return (None, None) if return_pair else None
    pair, value = max(values.items(), key=lambda item: item[1])
    return (value, pair) if return_pair else value


def stable_max_pair(shift, dirv, z_num, z_den, depths, dim):
    records = []
    for N in depths:
        value, pair = independent_delta(
            shift, dirv, z_num, z_den, N, dim, return_pair=True)
        records.append({"N": int(N), "delta_max": value, "pair": pair})
    pairs = [r["pair"] for r in records if r["pair"] is not None]
    stable = bool(pairs) and all(p == pairs[0] for p in pairs)
    return records, stable


def arith_delta(shift, dirv, z_num, z_den, N, dim, lam1_hint=None,
                row_pair=None):
    if lam1_hint is None or not np.isfinite(lam1_hint):
        lam1_hint = 8.0
    mp.mp.dps = int(2.6 * abs(lam1_hint) * N / math.log(10)) + 120
    P = mp.eye(dim)
    snapN = None
    for n in range(1, 2 * N + 1):
        P = P * build_M_mp(n, shift, dirv, z_num, z_den, dim)
        if n == N:
            snapN = mp.matrix(P)
    last = dim - 1
    pairs = [tuple(row_pair)] if row_pair is not None else [
        (i, j) for i in range(dim) for j in range(dim) if i != j]
    best = None
    for i, j in pairs:
        qn, q2 = snapN[j, last], P[j, last]
        if qn == 0 or q2 == 0:
            continue
        err = abs(snapN[i, last] / qn - P[i, last] / q2)
        if err == 0:
            continue
        lq = mp.log(abs(qn))
        if lq == 0:
            continue
        d = float(-(1 + mp.log(err) / lq))
        if best is None or d > best:
            best = d
    return best


def dir_pool(dim):
    nsh = nshift_for(dim)
    pats = []
    for i in range(nsh):
        d = [0] * nsh
        d[i] = 1
        pats.append(d)
    for i in range(dim, nsh):
        d = [0] * nsh
        d[i] = 2
        pats.append(d)
    for i, j in [(dim - 2, dim - 1), (dim, dim + 2), (dim + 2, dim + 3)]:
        if 0 <= i < nsh and 0 <= j < nsh:
            d = [0] * nsh
            d[i] = d[j] = 1
            pats.append(d)
    return pats


def z_pool(z_max=1.0, z_min=0.1):
    zs = set()
    qmax = int(math.ceil(2 * z_max))
    for q in sorted(set((1, 2, 3, 4, 5, 6, 10, 20)) | {qmax}):
        for p in range(-int(math.ceil(z_max * q)) - 1,
                       int(math.ceil(z_max * q)) + 2):
            if p == 0:
                continue
            fr = Fraction(p, q)
            if z_min <= abs(float(fr)) < z_max:
                zs.add((fr.numerator, fr.denominator))
    return sorted(zs)


def default_seed(dim):
    s = [0] * nshift_for(dim)
    for j in range(dim, len(s)):
        s[j] = -2
    return s
