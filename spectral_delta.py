"""
spectral_delta.py
=================
QR-stabilized Lyapunov spectra for recurrence/CMF matrix products.

Product convention
------------------
The arithmetic code uses the right product

    P_N = M_1 M_2 ... M_N.

To obtain the singular-value/Lyapunov spectrum of P_N without materialising it,
we propagate the transpose product

    P_N^T = M_N^T ... M_2^T M_1^T

by QR.  This is essential: using M_n @ Q would instead analyse the reversed
product M_N ... M_1, which is different for non-commuting matrices.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

ProductOrder = Literal["right", "left"]


def lyapunov_spectrum(
    build_fn: Callable[[int], np.ndarray],
    d: int,
    N: int,
    *,
    product_order: ProductOrder = "right",
) -> np.ndarray:
    """Return QR finite-time Lyapunov estimates.

    ``product_order='right'`` means P_N=M_1...M_N and QR is applied to M_n.T.
    ``product_order='left'`` means P_N=M_N...M_1 and QR is applied to M_n.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if product_order not in {"right", "left"}:
        raise ValueError("product_order must be 'right' or 'left'")

    Q = np.eye(d, dtype=np.float64)
    logs = np.zeros(d, dtype=np.float64)
    for n in range(N):
        M = np.asarray(build_fn(n), dtype=np.float64)
        if M.shape != (d, d):
            raise ValueError(f"build_fn({n}) returned {M.shape}, expected {(d, d)}")
        A = M.T if product_order == "right" else M
        Y = A @ Q
        Q, R = np.linalg.qr(Y)
        diag = np.diag(R).copy()
        signs = np.sign(diag)
        signs[signs == 0] = 1.0
        Q = Q @ np.diag(signs)
        logs += np.log(np.maximum(np.abs(diag), 1e-300))
    return logs / float(N)


@dataclass
class Recurrence:
    name: str
    dim: int
    coeffs: Callable[[int], list[float]]
    delta_inf: float
    note: str = ""


def f3g4_coeffs(n: int) -> list[float]:
    return [3 * (n + 4), -3, -1]


def f0g4_coeffs(n: int) -> list[float]:
    return [20 * (n + 2), 21, 0, -21, 0, 7]


RECURRENCES = {
    "f3g4": Recurrence("f3g4", 3, f3g4_coeffs, 0.5,
                        "collapsed 3F2 g4 candidate"),
    "f0g4": Recurrence("f0g4", 6, f0g4_coeffs, 0.2,
                        "non-degenerate 6F5 g4 candidate"),
}


def companion_float(rec: Recurrence, n: int) -> np.ndarray:
    d = rec.dim
    M = np.zeros((d, d), dtype=np.float64)
    M[0, :] = rec.coeffs(n)
    for r in range(1, d):
        M[r, r - 1] = 1.0
    return M


def finite_time_lyapunov(rec: Recurrence, N: int) -> np.ndarray:
    return lyapunov_spectrum(
        lambda n: companion_float(rec, n), rec.dim, N, product_order="right")


@dataclass
class FieldPoint:
    N: int
    delta: float
    E_over_Q: float
    Q_N: float
    E_N: float
    lambdas: np.ndarray
    gaps: np.ndarray


def analyze(rec: Recurrence, N: int) -> FieldPoint | None:
    lam = finite_time_lyapunov(rec, N)
    if lam[0] == 0 or not np.isfinite(lam[0]):
        return None
    Q_N = float(lam[0])
    E_N = float(lam[0] - lam[1])
    delta = E_N / Q_N - 1.0
    gaps = np.diff(lam) * -1.0
    return FieldPoint(N, delta, E_N / Q_N, Q_N, E_N, lam, gaps)


def run_experiment(rec: Recurrence, depths: list[int], csv_writer=None) -> None:
    d = rec.dim
    print(f"\n=== {rec.name} (d={d}, target={rec.delta_inf:.6g}) ===")
    print(f"{'N':>7} {'delta':>12} {'lambda1':>12} {'lambda2':>12} {'error':>12}")
    for N in depths:
        fp = analyze(rec, N)
        if fp is None:
            continue
        err = fp.delta - rec.delta_inf
        print(f"{N:7d} {fp.delta:12.6f} {fp.lambdas[0]:12.6f} "
              f"{fp.lambdas[1]:12.6f} {err:12.6f}")
        if csv_writer is not None:
            row = {"recur": rec.name, "dim": d, "N": N, "delta": fp.delta,
                   "E_over_Q": fp.E_over_Q, "Q_N": fp.Q_N, "E_N": fp.E_N,
                   "delta_inf": rec.delta_inf}
            for k, value in enumerate(fp.lambdas, start=1):
                row[f"lambda{k}"] = float(value)
            csv_writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recur", choices=list(RECURRENCES) + ["all"], default="all")
    ap.add_argument("--depths", default="50,100,200,500")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    recs = list(RECURRENCES.values()) if args.recur == "all" else [RECURRENCES[args.recur]]

    fh = writer = None
    if args.csv:
        maxd = max(r.dim for r in recs)
        fields = ["recur", "dim", "N", "delta", "E_over_Q", "Q_N", "E_N", "delta_inf"]
        fields += [f"lambda{k}" for k in range(1, maxd + 1)]
        fh = open(args.csv, "w", newline="")
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
    try:
        for rec in recs:
            run_experiment(rec, depths, writer)
    finally:
        if fh:
            fh.close()


if __name__ == "__main__":
    main()
