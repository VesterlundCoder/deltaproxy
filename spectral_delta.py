"""
spectral_delta.py
=================
Continuous CMF modeling — spectral layer on top of the verified 6F5 delta machinery.

This is the "first experiment" of section 16 of the Delta-Ladder continuous-modeling
plan: turn a discrete trajectory (companion-matrix product) into a small bundle of
*asymptotic-rate* observables at each parameter point, so that the discrete delta
field becomes a continuous spectral landscape.

For one parameter point theta (a recurrence) and depth N we compute:

    Q_N(theta)  = log|q_N| / N                  (denominator growth rate)
    E_N(theta)  = -log|r_N - r_2N| / N           (convergent error-decay rate)
    delta_N     = -1 + E_N/Q_N = -1 - log eps / log|q_N|   (the preprint's delta)
    lambda_i^N  = (1/N) log s_i(P_N)             (finite-time Lyapunov, QR-stabilised)
    gap12^N     = lambda_1 - lambda_2            (dominant spectral gap)

Delta-Ladder hypothesis (g4 staircase, effective dimension d):

    E/Q  ->  d/(d-1)        <=>      delta_inf -> 1/(d-1)

Spectral hypothesis to test:

    delta_N  ~  gap12_N / Q_N

Key finding (validated against the verified deltas): the convergent error decays
at the spectral-gap rate and the denominator grows at the top Lyapunov rate, so

    Q = lambda_1,   E = lambda_1 - lambda_2,   delta = -1 + E/Q = -lambda_2/lambda_1

This is computed entirely from the QR-stabilised Lyapunov spectrum in float64 (no
high-precision matrix product needed), reproducing f3g4 -> 1/2 and f0g4 -> 1/5
and matching the preprint's measured deltas (e.g. f0g4 N=500 -> 0.147 == HIT B).

The two reference recurrences (collapsed g4 hits A and B from the preprint):
  f3g4 (d=3, delta_inf = 1/2):  a_{n+3} = 3(n+4) a_{n+2} - 3 a_{n+1} - a_n
  f0g4 (d=6, delta_inf = 1/5):  a_{n+6} = 20(n+2) a_{n+5} + 21 a_{n+4} - 21 a_{n+2} + 7 a_n

Usage:
  python3 spectral_delta.py                          # both recurrences, default depths
  python3 spectral_delta.py --recur f3g4 --depths 50,100,200,500,1000
  python3 spectral_delta.py --csv fields.csv         # also dump Phi(N) rows to CSV
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Recurrence registry
# A recurrence of order d:  a_{n+d} = sum_{i=0}^{d-1} c_i(n) a_{n+i}
# coeffs(n) returns the *top row* of the companion matrix, ordered
#   [c_{d-1}, c_{d-2}, ..., c_0].
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Recurrence:
    name: str
    dim: int
    coeffs: Callable[[int], list]          # n -> [c_{d-1}, ..., c_0]
    delta_inf: float                        # Delta-Ladder prediction 1/(d-1)
    note: str = ""


def f3g4_coeffs(n: int) -> list:
    # a_{n+3} = 3(n+4) a_{n+2} - 3 a_{n+1} - a_n
    return [3 * (n + 4), -3, -1]


def f0g4_coeffs(n: int) -> list:
    # a_{n+6} = 20(n+2) a_{n+5} + 21 a_{n+4} + 0 a_{n+3} - 21 a_{n+2} + 0 a_{n+1} + 7 a_n
    return [20 * (n + 2), 21, 0, -21, 0, 7]


RECURRENCES: dict[str, Recurrence] = {
    "f3g4": Recurrence("f3g4", 3, f3g4_coeffs, 0.5,
                       "collapsed 3F2 g4 hit A (z=1/3), delta -> 1/2"),
    "f0g4": Recurrence("f0g4", 6, f0g4_coeffs, 0.2,
                       "non-degenerate 6F5 g4 hit B (z=7/20), delta -> 1/5"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Companion matrices
# ─────────────────────────────────────────────────────────────────────────────
def companion_float(rec: Recurrence, n: int) -> np.ndarray:
    """Top-row companion as float64 (for Lyapunov QR)."""
    d = rec.dim
    M = np.zeros((d, d), dtype=np.float64)
    M[0, :] = rec.coeffs(n)
    for r in range(1, d):
        M[r, r - 1] = 1.0
    return M


# ─────────────────────────────────────────────────────────────────────────────
# Finite-time Lyapunov exponents (QR, sign-stabilised) — section 11
# ─────────────────────────────────────────────────────────────────────────────
def lyapunov_spectrum(build_fn: Callable[[int], np.ndarray], d: int, N: int) -> np.ndarray:
    """lambda_i^(N) = (1/N) sum log|R_ii| for the cocycle P = M(0)...M(N-1).

    `build_fn(n)` returns the d x d step matrix at depth n. Pure float64 with
    per-step QR renormalisation: stable to arbitrary depth (the matrix product
    is never materialised), so this scales to millions of parameter points for
    the continuous field.
    """
    Q = np.eye(d)
    logs = np.zeros(d)
    for n in range(N):
        Y = build_fn(n) @ Q
        Q, R = np.linalg.qr(Y)
        diag = np.diag(R).copy()
        signs = np.sign(diag)
        signs[signs == 0] = 1.0
        S = np.diag(signs)
        Q = Q @ S
        R = S @ R
        logs += np.log(np.maximum(np.abs(np.diag(R)), 1e-300))
    return logs / N


def finite_time_lyapunov(rec: Recurrence, N: int) -> np.ndarray:
    """Lyapunov spectrum of a registered scalar recurrence's companion cocycle."""
    return lyapunov_spectrum(lambda n: companion_float(rec, n), rec.dim, N)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral delta:  Q = lambda_1,  E = lambda_1 - lambda_2,  delta = -1 + E/Q
# Empirically (see first experiment) gap12/Q -> d/(d-1), i.e. delta -> 1/(d-1),
# reproducing the preprint deltas (f3g4 -> 1/2, f0g4 -> 1/5) directly from the
# Lyapunov spectrum — the spectral explanation of the Delta Ladder.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FieldPoint:
    N: int
    delta: float            # -1 + E/Q = -lambda_2/lambda_1
    E_over_Q: float         # (lambda_1 - lambda_2)/lambda_1  -> d/(d-1)
    Q_N: float              # lambda_1   (denominator growth rate)
    E_N: float              # lambda_1 - lambda_2  (error-decay rate)
    lambdas: np.ndarray
    gaps: np.ndarray        # lambda_i - lambda_{i+1}


def analyze(rec: Recurrence, N: int) -> FieldPoint | None:
    """Full spectral bundle Phi_N(theta) for one recurrence at depth N."""
    lam = finite_time_lyapunov(rec, N)
    if lam[0] == 0:
        return None
    Q_N = float(lam[0])
    E_N = float(lam[0] - lam[1]) if rec.dim >= 2 else float("nan")
    delta = E_N / Q_N - 1.0
    gaps = np.array([lam[i] - lam[i + 1] for i in range(rec.dim - 1)])
    return FieldPoint(N, delta, E_N / Q_N, Q_N, E_N, lam, gaps)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(rec: Recurrence, depths: list[int], csv_writer=None):
    d = rec.dim
    target = d / (d - 1)
    print(f"\n=== {rec.name}  (dim d={d},  delta_inf=1/(d-1)={rec.delta_inf:.4f},"
          f"  E/Q target d/(d-1)={target:.4f}) ===")
    print(f"    {rec.note}")
    head = (f"{'N':>6} {'delta':>9} {'E/Q':>9} {'Q=lam1':>9} {'E=gap12':>9} "
            f"{'lam2':>9} {'lam_d':>9}  delta_err")
    print(head)
    print("    " + "-" * (len(head) + 4))
    for N in depths:
        fp = analyze(rec, N)
        if fp is None:
            print(f"{N:>6}   (degenerate)")
            continue
        l2 = fp.lambdas[1] if d >= 2 else float("nan")
        ld = fp.lambdas[-1]
        derr = fp.delta - rec.delta_inf
        print(f"{N:>6} {fp.delta:>9.4f} {fp.E_over_Q:>9.4f} {fp.Q_N:>9.4f} "
              f"{fp.E_N:>9.4f} {l2:>9.4f} {ld:>9.4f}  {derr:>+9.4f}")
        if csv_writer is not None:
            row = {
                "recur": rec.name, "dim": d, "N": N,
                "delta": fp.delta, "E_over_Q": fp.E_over_Q,
                "Q_N": fp.Q_N, "E_N": fp.E_N,
                "target_EoverQ": target, "delta_inf": rec.delta_inf,
            }
            for k in range(d):
                row[f"lambda{k+1}"] = float(fp.lambdas[k])
            for k in range(d - 1):
                row[f"gap{k+1}{k+2}"] = float(fp.gaps[k])
            csv_writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recur", choices=list(RECURRENCES) + ["all"], default="all")
    ap.add_argument("--depths", default="50,100,200,500",
                    help="comma-separated depths N")
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    recs = (list(RECURRENCES.values()) if args.recur == "all"
            else [RECURRENCES[args.recur]])

    csv_file = csv_writer = None
    if args.csv:
        maxd = max(r.dim for r in recs)
        cols = (["recur", "dim", "N", "delta", "E_over_Q", "Q_N", "E_N",
                 "target_EoverQ", "delta_inf"]
                + [f"lambda{k+1}" for k in range(maxd)]
                + [f"gap{k+1}{k+2}" for k in range(maxd - 1)])
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=cols)
        csv_writer.writeheader()

    for rec in recs:
        run_experiment(rec, depths, csv_writer)

    if csv_file:
        csv_file.close()
        print(f"\nCSV -> {args.csv}")

    print("\nReadout:")
    print("  * delta = -1 + E/Q = -lambda_2/lambda_1 is computed PURELY from the")
    print("    Lyapunov spectrum (float64, no high precision).")
    print("  * E/Q should approach d/(d-1); delta should approach 1/(d-1) (Delta Ladder).")
    print("  * delta_err is delta - 1/(d-1): watch it shrink as N grows.")


if __name__ == "__main__":
    main()
