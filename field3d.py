"""
field3d.py
==========
Continuous 3D spectral field over the f0g4 chart of the 6F5 parameter space.

Coordinates (section 14 of the continuous-modeling plan):
    x = log10|z|          spectral / argument axis
    y = sigma             advancing g-root offset
    r = dist(g, g4)       distance from the g4 boundary (4 pinned roots = 0)

At each point we build the verified 6F5 step matrix M(n) and read the spectral
bundle straight off the QR Lyapunov spectrum (float64, one pass / point):

    Phi(x,y,r) = ( delta, E/Q, Q=lam1, E=lam1-lam2, lam1..lam6, gap12, gap12/Q )
    delta = -1 + E/Q = -lam2/lam1
    Omega+ = { delta > 0 } = { E > Q }

The 6F5 matrix (from verify_znegsweep_hits.py / MULTI_Z_TECHNICAL_NOTE.md):
    f_i  (i=0..5)  numerator roots A(lambda)
    g_j  (j=0..4)  denominator roots B(lambda); on g4 four are pinned to 0
    c[0]   = h_f * ef[6]
    c[k]   = h_g * eg[6-k] + h_f * ef[6-k]      (k=1..5)
    M(n): 6x6 companion, subdiagonal = 1, last column = c

IMPORTANT (gauge): the preprint's delta uses the *integer-cleared* denominator,
so we set h_g = D_ref, h_f = -z*D_ref where D_ref is the chart's reference
z-denominator (20 for the f0g4 hit z=7/20). Integer-clearing shifts lambda_1 by
+log(D_ref) while leaving lambda_2 fixed, so delta = -lambda_2/lambda_1 matches
the scalar recurrence (e.g. f0g4 N=500 -> 0.147). The continuous error rate E is
gauge-invariant; only log|q_n| (=> lambda_1) carries the D_ref scaling.

f0g4 chart (collapsed hit B, z=7/20):
    f-roots fixed at [-1,-1,1,1,-1,1]
    advancing g-root  g3(n) = sigma + n + 2
    four pinned g-roots = r/2  (so sqrt(4*(r/2)^2) = r)

Usage:
    python3 field3d.py --selftest
    python3 field3d.py --nx 25 --ny 21 --nr 17 --depth 400 \
                       --csv field3d.csv --png field3d.png
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from spectral_delta import lyapunov_spectrum

# f0g4 chart constants
F_ROOTS_BASE = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
DIM = 6
D_REF = 20.0          # reference z-denominator of the f0g4 chart (hit z=7/20)


def esym(vals: np.ndarray) -> np.ndarray:
    """Elementary symmetric polynomials e[0..len(vals)] (e[0]=1)."""
    n = len(vals)
    e = np.zeros(n + 1)
    e[0] = 1.0
    for v in vals:
        for k in range(n, 0, -1):
            e[k] += v * e[k - 1]
    return e


def build_6f5(n: int, z: float, sigma: float, r: float,
              d_ref: float = D_REF) -> np.ndarray:
    """6x6 f0g4 step matrix at depth n for parameters (z, sigma, r).

    Integer-cleared convention: h_g = d_ref, h_f = -z*d_ref.
    """
    h_f = -z * d_ref
    h_g = d_ref
    f = F_ROOTS_BASE
    g_pinned = r / 2.0
    g = np.array([g_pinned, g_pinned, g_pinned, sigma + n + 2.0, g_pinned])
    ef = esym(f)                       # length 7
    eg = esym(g)                       # length 6
    eg = np.concatenate([eg, [0.0]])   # pad to length 7

    c = np.zeros(DIM)
    c[0] = h_f * ef[6]
    for k in range(1, DIM):
        c[k] = h_g * eg[6 - k] + h_f * ef[6 - k]

    M = np.zeros((DIM, DIM))
    for rr in range(1, DIM):
        M[rr, rr - 1] = 1.0
    M[:, DIM - 1] = c
    return M


def phi(z: float, sigma: float, r: float, depth: int,
        d_ref: float = D_REF) -> dict:
    """Spectral bundle at one parameter point."""
    lam = lyapunov_spectrum(lambda n: build_6f5(n, z, sigma, r, d_ref),
                            DIM, depth)
    Q = float(lam[0])
    E = float(lam[0] - lam[1])
    delta = E / Q - 1.0 if Q != 0 else float("nan")
    gap12 = float(lam[0] - lam[1])
    out = {
        "delta": delta,
        "E_over_Q": E / Q if Q != 0 else float("nan"),
        "Q": Q, "E": E,
        "gap12": gap12,
        "gap12_over_Q": gap12 / Q if Q != 0 else float("nan"),
    }
    for i in range(DIM):
        out[f"lambda{i+1}"] = float(lam[i])
    return out


# ─────────────────────────────────────────────────────────────────────────────
def selftest():
    """On-boundary f0g4 check: delta should track the validated ~0.13-0.16 band."""
    print("f0g4 on-boundary self-test (z=7/20, sigma=0, r=0):")
    print(f"{'N':>6} {'delta':>9} {'E/Q':>9} {'Q=lam1':>9} {'gap12':>9}")
    for N in (100, 500, 1000):
        p = phi(7.0 / 20.0, 0.0, 0.0, N)
        print(f"{N:>6} {p['delta']:>9.4f} {p['E_over_Q']:>9.4f} "
              f"{p['Q']:>9.4f} {p['gap12']:>9.4f}")
    print("Expected (scalar f0g4): N=500 delta ~ 0.147, climbing toward 1/5.")


# ─────────────────────────────────────────────────────────────────────────────
def sample_grid(xr, yr, rr, depth, z_sign=1.0):
    """Return dict of flat arrays over the (x,y,r) grid."""
    pts = []
    for x in xr:
        z = z_sign * (10.0 ** x)
        for y in yr:
            for r in rr:
                p = phi(z, y, r, depth)
                p.update({"x": x, "y": y, "r": r, "z": z})
                pts.append(p)
    keys = pts[0].keys()
    return {k: np.array([p[k] for p in pts]) for k in keys}


def plot_field(G, png, depth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x, y, r = G["x"], G["y"], G["r"]
    delta = G["delta"]
    pos = delta > 0

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f"6F5 f0g4 spectral field  Phi(x,y,r)   depth N={depth}\n"
                 f"x=log10|z|,  y=sigma,  r=dist(g,g4);   delta = -1 + E/Q = "
                 f"-lambda2/lambda1", fontsize=12)

    # 1. 3D scatter coloured by delta, positive region ringed
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    sc = ax.scatter(x, y, r, c=delta, cmap="RdBu_r",
                    vmin=-abs(delta).max(), vmax=abs(delta).max(), s=14)
    if pos.any():
        ax.scatter(x[pos], y[pos], r[pos], facecolors="none",
                   edgecolors="lime", s=46, linewidths=1.1,
                   label="Omega+ (delta>0)")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("x=log10|z|"); ax.set_ylabel("y=sigma"); ax.set_zlabel("r")
    ax.set_title("delta over (x,y,r)")
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="delta")

    # 2. delta vs r (decay off the g4 boundary) for several (x,y)
    ax = fig.add_subplot(2, 3, 2)
    xs = np.unique(x); ys = np.unique(y); rs = np.unique(r)
    xmid = xs[len(xs) // 2]
    for yv in ys[:: max(1, len(ys) // 5)]:
        m = (np.isclose(x, xmid)) & (np.isclose(y, yv))
        if m.any():
            order = np.argsort(r[m])
            ax.plot(r[m][order], delta[m][order], "o-", ms=3,
                    label=f"sigma={yv:.2f}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("r = dist(g,g4)"); ax.set_ylabel("delta")
    ax.set_title(f"delta(r) at x={xmid:.2f}  (boundary decay)")
    ax.legend(fontsize=7)

    # 3. delta vs x at r=0 (z-band universality)
    ax = fig.add_subplot(2, 3, 3)
    r0 = rs[0]
    ymid = ys[len(ys) // 2]
    m = np.isclose(r, r0) & np.isclose(y, ymid)
    if m.any():
        order = np.argsort(x[m])
        ax.plot(x[m][order], delta[m][order], "o-", ms=3)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("x=log10|z|"); ax.set_ylabel("delta")
    ax.set_title(f"delta(x) at r={r0:.2f}, sigma={ymid:.2f}")

    # 4. boundary slice r=r0: delta heatmap over (x,y)
    ax = fig.add_subplot(2, 3, 4)
    m = np.isclose(r, r0)
    _heat(ax, x[m], y[m], delta[m], "x=log10|z|", "y=sigma",
          f"delta on boundary r={r0:.2f}", "RdBu_r", fig)

    # 5. E/Q heatmap on boundary (Delta-Ladder ratio d/(d-1)=1.2)
    ax = fig.add_subplot(2, 3, 5)
    _heat(ax, x[m], y[m], G["E_over_Q"][m], "x=log10|z|", "y=sigma",
          "E/Q on boundary (ladder target 1.2)", "viridis", fig)

    # 6. gap12/Q vs delta (spectral hypothesis delta ~ gap12/Q - 1)
    ax = fig.add_subplot(2, 3, 6)
    ax.scatter(G["gap12_over_Q"] - 1.0, delta, s=10, alpha=0.5)
    lim = [min((G["gap12_over_Q"] - 1.0).min(), delta.min()),
           max((G["gap12_over_Q"] - 1.0).max(), delta.max())]
    ax.plot(lim, lim, "r--", lw=1, label="y=x")
    ax.set_xlabel("gap12/Q - 1"); ax.set_ylabel("delta")
    ax.set_title("spectral identity check")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png, dpi=130)
    print(f"PNG  -> {png}")


def _heat(ax, xv, yv, cv, xl, yl, title, cmap, fig):
    xs = np.unique(xv); ys = np.unique(yv)
    Z = np.full((len(ys), len(xs)), np.nan)
    for xi, yi, ci in zip(xv, yv, cv):
        Z[np.where(np.isclose(ys, yi))[0][0],
          np.where(np.isclose(xs, xi))[0][0]] = ci
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
                   extent=[xs.min(), xs.max(), ys.min(), ys.max()])
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--nx", type=int, default=21)
    ap.add_argument("--ny", type=int, default=17)
    ap.add_argument("--nr", type=int, default=13)
    ap.add_argument("--xmin", type=float, default=-1.4)
    ap.add_argument("--xmax", type=float, default=-0.2)
    ap.add_argument("--ymin", type=float, default=-3.0)
    ap.add_argument("--ymax", type=float, default=3.0)
    ap.add_argument("--rmax", type=float, default=2.0)
    ap.add_argument("--depth", type=int, default=400)
    ap.add_argument("--zsign", type=float, default=1.0)
    ap.add_argument("--csv", default="field3d.csv")
    ap.add_argument("--png", default="field3d.png")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    xr = np.linspace(args.xmin, args.xmax, args.nx)
    yr = np.linspace(args.ymin, args.ymax, args.ny)
    rr = np.linspace(0.0, args.rmax, args.nr)
    npts = args.nx * args.ny * args.nr
    print(f"Sampling {npts} points  (depth N={args.depth}) ...")

    G = sample_grid(xr, yr, rr, args.depth, z_sign=args.zsign)

    with open(args.csv, "w", newline="") as f:
        cols = ["x", "y", "r", "z", "delta", "E_over_Q", "Q", "E",
                "gap12", "gap12_over_Q"] + [f"lambda{i+1}" for i in range(DIM)]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(npts):
            w.writerow({k: G[k][i] for k in cols})
    print(f"CSV  -> {args.csv}")

    pos = (G["delta"] > 0)
    print(f"Omega+ (delta>0): {pos.sum()}/{npts} points "
          f"({100*pos.mean():.1f}%)")
    if pos.any():
        print(f"  max delta = {G['delta'].max():.4f} at "
              f"x={G['x'][np.argmax(G['delta'])]:.3f}, "
              f"y={G['y'][np.argmax(G['delta'])]:.3f}, "
              f"r={G['r'][np.argmax(G['delta'])]:.3f}")

    plot_field(G, args.png, args.depth)


if __name__ == "__main__":
    main()
