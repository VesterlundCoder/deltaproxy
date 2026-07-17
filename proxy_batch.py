"""Batched QR proxy with the repository's right-product convention P=M1...MN."""
from __future__ import annotations

import numpy as np


def esym_batch(vals, xp=np):
    B, m = vals.shape
    e = xp.zeros((B, m + 1), dtype=vals.dtype)
    e[:, 0] = 1.0
    for j in range(m):
        e[:, 1:m + 1] = e[:, 1:m + 1] + vals[:, j:j + 1] * e[:, 0:m]
    return e


def build_M_batch(n, shift, dirv, z_num, z_den, dim, xp=np):
    B = shift.shape[0]
    f = shift[:, :dim] + n * dirv[:, :dim] + 1.0
    g = shift[:, dim:] + n * dirv[:, dim:] + 2.0
    ef = esym_batch(f, xp)
    eg = esym_batch(g, xp)
    eg = xp.concatenate([eg, xp.zeros((B, 1), dtype=eg.dtype)], axis=1)
    c = xp.zeros((B, dim), dtype=shift.dtype)
    c[:, 0] = -z_num * ef[:, dim]
    for k in range(1, dim):
        c[:, k] = z_den * eg[:, dim-k] - z_num * ef[:, dim-k]
    M = xp.zeros((B, dim, dim), dtype=shift.dtype)
    for r in range(1, dim):
        M[:, r, r-1] = 1.0
    M[:, :, dim-1] = c
    return M


def lyapunov_batch(shift, dirv, z_num, z_den, dim, N, xp=np, qr_every=1):
    """QR growth rates for P_N=M_1...M_N.

    We propagate P_N^T=M_N^T...M_1^T, hence the transposed batched matrices.
    """
    B = shift.shape[0]
    Q = xp.broadcast_to(xp.eye(dim, dtype=shift.dtype), (B, dim, dim)).copy()
    logs = xp.zeros((B, dim), dtype=shift.dtype)
    for n in range(1, N + 1):
        M = build_M_batch(n, shift, dirv, z_num, z_den, dim, xp)
        MT = xp.swapaxes(M, 1, 2)
        Q = xp.einsum('bij,bjk->bik', MT, Q)
        if n % qr_every == 0 or n == N:
            Qn, R = xp.linalg.qr(Q)
            diag = xp.diagonal(R, axis1=1, axis2=2)
            signs = xp.where(xp.sign(diag) == 0, 1.0, xp.sign(diag))
            Q = Qn * signs[:, None, :]
            logs += xp.log(xp.maximum(xp.abs(diag), 1e-300))
    return logs / float(N)


def ratios_batch(lam, xp=np):
    l1 = lam[:, :1]
    bad = ~xp.isfinite(l1) | (l1 == 0)
    ratios = -lam[:, 1:] / xp.where(bad, 1.0, l1)
    return xp.where(bad, xp.nan, ratios)
