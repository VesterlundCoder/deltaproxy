"""
classA.py
=========
Full Class-A pipeline for a discovered 6F5 candidate:

  1. NOVELTY   — check the (shift,dir,z) against the known-hit databases.
  2. DEEP WALK — exact-integer Ramanujan walk to N=3000 (double-depth 6000),
                 combo-delta profile delta(n) -> convergence / trend.
  3. SPECTRUM  — float Lyapunov spectrum: lambda gaps, lam2≈lam3 degeneracy,
                 effective ladder dimension  d_eff = 1 + 1/delta_inf.
  4. IDENTIFY  — high-precision limit L + mpmath.identify + PSLQ over a
                 zeta/pi/log basis (reuses analyse_large_hits.process_candidate).

Reuses the production toolkit in ../ :
  wide_scalpel_sweep.walk_to        (pure-integer snapshot walk)
  campaign_strata_sweep.best_delta_combos / classify_trend
  analyse_large_hits.process_candidate (L, identify, PSLQ, 30 bonus row pairs)

Usage:
  python3 classA.py --cand C1            # delta~0.37 candidate
  python3 classA.py --cand C2            # delta~0.274 candidate
  python3 classA.py --shift "-1,-1,0,-1,0,0,-2,-2,-2,2,-2" \
                    --dir "0,0,0,0,0,0,0,0,0,1,0" --z 7/20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PARENT = Path(__file__).resolve().parent.parent       # 6F5Sweeps/
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys, "set_int_max_str_digits"):       # Python 3.11+; no-op on 3.9
    sys.set_int_max_str_digits(10_000_000)

import mpmath as mp
from wide_scalpel_sweep import walk_to
from analyse_large_hits import process_candidate
from spectral_delta import lyapunov_spectrum
from enrich import build_M_float

DIM = 6
_COEFFS = (-3, -2, -1, 1, 2, 3)


# ── combo-delta (inlined from campaign_strata_sweep.py to avoid 3.11-only import)
def _delta_exact(pn, qn, p2n, q2n):
    if qn == 0 or q2n == 0 or (pn == 0 and p2n == 0):
        return None
    cross = pn * q2n - p2n * qn          # exact integer, no cancellation
    if cross == 0:
        return None
    abs_qn = abs(qn)
    if abs_qn <= 1:
        return None
    with mp.workdps(60):
        log_cross = float(mp.log(abs(cross)))
        log_qn = float(mp.log(abs_qn))
        log_q2n = float(mp.log(abs(q2n)))
    return -(1.0 + (log_cross - log_qn - log_q2n) / log_qn)


def best_delta_combos(snap_n, snap_2n, dps=None):
    v = [snap_n[r * 6 + 5] for r in range(6)]
    w = [snap_2n[r * 6 + 5] for r in range(6)]
    best = None

    def try_combo(pn, qn, p2n, q2n):
        nonlocal best
        d = _delta_exact(pn, qn, p2n, q2n)
        if d is not None and (best is None or d > best):
            best = d

    for ni in range(6):
        for nj in range(6):
            if ni == nj:
                continue
            try_combo(v[ni], v[nj], w[ni], w[nj])
    for ni in range(6):
        for nj in range(ni + 1, 6):
            for nk in range(6):
                if nk == ni or nk == nj or v[nk] == 0 or w[nk] == 0:
                    continue
                for a in _COEFFS:
                    for b in _COEFFS:
                        try_combo(a * v[ni] + b * v[nj], v[nk],
                                  a * w[ni] + b * w[nj], w[nk])
    return best


def classify_trend(prof):
    if not prof:
        return "UNKNOWN"
    ns = sorted(prof)
    d_last = prof[ns[-1]]
    slope = ((prof[ns[-1]] - prof[ns[0]]) /
             (mp.log(ns[-1]) - mp.log(ns[0]))) if len(ns) >= 2 else 0.0
    if d_last >= 0.15:
        return ("INCREASING" if slope > 0.005 else
                "CONVERGING+" if slope < -0.005 else "STABLE")
    return "NEED_DEEPER" if d_last >= 0.05 else "DECLINING"

PRESETS = {
    # delta ~ 0.37, found by enrich.py (fixed hit-B chart)
    "C1": dict(shift=[-1, -1, 0, -1, 0, 0, -2, -2, -2, 2, -2],
               dir=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], z_num=7, z_den=20),
    # delta ~ 0.274, found by discover.py (widened space: g-root adv speed 2)
    "C2": dict(shift=[-3, -1, 0, -1, -1, -3, 2, -2, -2, -2, -2],
               dir=[0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0], z_num=9, z_den=20),
}

HIT_FILES = ["znegsweep_hits_unique.jsonl", "large_hits_lirec.jsonl",
             "large_hits_positive_delta.jsonl", "large_hits_deep_rt.jsonl",
             "large_hits_self_lf_delta.jsonl"]


# ─────────────────────────────────────────────────────────────────────────────
def novelty(shift, dirv, z_num, z_den):
    target = (tuple(shift), tuple(dirv), z_num, z_den)
    hits = []
    for fn in HIT_FILES:
        p = PARENT / fn
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = (tuple(r.get("shift", [])), tuple(r.get("dir", [])),
                 r.get("z_num", r.get("z", None)), r.get("z_den", None))
            if (k[0], k[1]) == (target[0], target[1]):
                hits.append((fn, r.get("z"), r.get("real_delta")))
    return hits


def deep_profile(shift, dirv, z_num, z_den, depths):
    all_d = sorted(set(depths + [2 * d for d in depths]))
    snaps = walk_to(shift, dirv, z_num, z_den, all_d)
    prof = {}
    for n in depths:
        if n in snaps and 2 * n in snaps:
            d = best_delta_combos(snaps[n], snaps[2 * n])
            if d is not None:
                prof[n] = round(d, 5)
    return prof


def spectrum(shift, dirv, z_num, z_den, N):
    z = z_num / z_den
    lam = lyapunov_spectrum(
        lambda n: build_M_float(n + 1, shift, dirv, z_num, z_den), DIM, N)
    return np.array(lam, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", choices=list(PRESETS))
    ap.add_argument("--shift")
    ap.add_argument("--dir", dest="dirv")
    ap.add_argument("--z")
    ap.add_argument("--depths", default="100,200,400,800,1600,3000")
    ap.add_argument("--spec_N", type=int, default=4000)
    ap.add_argument("--id_depth", type=int, default=1500)
    ap.add_argument("--id_dps", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", default=None)
    args = ap.parse_args()

    if args.cand:
        c = PRESETS[args.cand]
        shift, dirv, z_num, z_den = c["shift"], c["dir"], c["z_num"], c["z_den"]
        name = args.cand
    else:
        shift = [int(x) for x in args.shift.split(",")]
        dirv = [int(x) for x in args.dirv.split(",")]
        zn, zd = args.z.split("/")
        z_num, z_den = int(zn), int(zd)
        name = "custom"

    depths = [int(x) for x in args.depths.split(",")]
    out = args.out or f"classA_{name}.json"
    png = args.png or f"classA_{name}.png"

    print(f"{'='*70}\nCLASS-A PIPELINE  [{name}]   z={z_num}/{z_den}")
    print(f"  shift={shift}\n  dir  ={dirv}\n{'='*70}")

    # 1. NOVELTY
    hits = novelty(shift, dirv, z_num, z_den)
    print("\n[1] NOVELTY")
    if hits:
        print(f"  MATCHES known hit(s): {hits}")
    else:
        print("  NEW — (shift,dir) not present in any known-hit database.")

    # 2. DEEP WALK
    print(f"\n[2] DEEP RAMANUJAN WALK  (combo-delta, N up to {max(depths)}, "
          f"double-depth {2*max(depths)})")
    prof = deep_profile(shift, dirv, z_num, z_den, depths)
    for n in sorted(prof):
        print(f"    delta({n:>5}) = {prof[n]:+.5f}")
    trend = classify_trend(prof)
    print(f"  trend = {trend}")

    # 3. SPECTRUM
    print(f"\n[3] LYAPUNOV SPECTRUM  (float, N={args.spec_N})")
    lam = spectrum(shift, dirv, z_num, z_den, args.spec_N)
    gaps = np.diff(-lam)
    delta_inf = -lam[1] / lam[0]
    d_eff = 1 + 1 / delta_inf if delta_inf > 0 else float("nan")
    print(f"    lambda = {np.round(lam, 4)}")
    print(f"    gaps   = {np.round(gaps, 4)}   (lam_i - lam_(i+1))")
    print(f"    delta_inf(spectral) = {delta_inf:+.4f}   E/Q = {delta_inf+1:.4f}")
    print(f"    effective ladder dim d_eff = 1 + 1/delta = {d_eff:.3f}")
    deg23 = abs(lam[1] - lam[2])
    print(f"    |lam2 - lam3| = {deg23:.4f}  "
          f"{'<-- near-degenerate pair' if deg23 < 0.15 else ''}")

    # 4. IDENTIFY / PSLQ
    print(f"\n[4] LIMIT IDENTIFICATION  (depth={args.id_depth}, dps={args.id_dps})")
    # auto-detect nonzero last-column rows (top rows can vanish structurally)
    from analyse_large_hits import _build_M
    with mp.workdps(80):
        Pp = mp.eye(6)
        for n in range(1, 101):
            Pp = Pp * _build_M(n, shift, dirv, z_num, z_den)
        nz = [r for r in range(6) if Pp[r, 5] != 0]
    num_row = nz[0] if nz else 0
    den_row = nz[1] if len(nz) > 1 else (nz[0] if nz else 2)
    print(f"    nonzero last-col rows = {nz}  ->  num_row={num_row}, "
          f"den_row={den_row}  (effective subspace dim = {len(nz)})")
    rec = dict(shift=shift, dir=dirv, z_num=z_num, z_den=z_den,
               z=f"{z_num}/{z_den}", num_row=num_row, den_row=den_row)
    res = process_candidate((0, rec, args.id_depth, args.id_dps, args.id_dps))
    print(f"    L            = {res.get('L')}")
    print(f"    real_delta   = {res.get('real_delta')}  "
          f"(combo: {res.get('best_delta_combo')})")
    print(f"    conv_digits  = {res.get('num_stable_digits')}")
    print(f"    identify     = {res.get('identify')}")
    print(f"    PSLQ(L)      = {res.get('pslq_expr')}  res={res.get('pslq_res')}")
    print(f"    PSLQ(combo)  = {res.get('combo_pslq_expr')}  "
          f"res={res.get('combo_pslq_res')}")
    if res.get("bonus"):
        print("    bonus row-pair PSLQ hits:")
        for b in res["bonus"][:5]:
            print(f"      {b['row_pair']}: {b['pslq']}  "
                  f"(conv {b['conv_digits']}d, res {b['residual']:.1e})")

    # verdict
    print(f"\n[VERDICT]")
    stable = trend in ("STABLE", "CONVERGING+", "INCREASING")
    pos = prof and prof[max(prof)] > 0
    identified = bool(res.get("pslq_expr") or res.get("identify")
                      or res.get("combo_pslq_expr"))
    print(f"    new={not hits}  positive={pos}  trend={trend}  "
          f"spectral_explained={delta_inf>0}  identified={identified}")
    cls = ("A" if (not hits and pos and stable and identified)
           else "B" if (not hits and pos and stable)
           else "C")
    print(f"    => CLASS {cls}")

    report = dict(name=name, shift=shift, dir=dirv, z_num=z_num, z_den=z_den,
                  novelty_matches=hits, profile=prof, trend=trend,
                  lambda_spectrum=lam.tolist(), gaps=gaps.tolist(),
                  delta_inf_spectral=float(delta_inf), d_eff=float(d_eff),
                  deg_lam2_lam3=float(deg23), identification=res, klass=cls)
    Path(out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nJSON -> {out}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ns = sorted(prof)
        ax[0].plot(ns, [prof[n] for n in ns], "o-")
        ax[0].axhline(0, color="k", lw=0.6, ls="--")
        ax[0].axhline(delta_inf, color="r", lw=0.8, ls=":",
                      label=f"spectral δ∞={delta_inf:.3f}")
        ax[0].set_xscale("log"); ax[0].set_xlabel("N"); ax[0].set_ylabel("delta(N)")
        ax[0].set_title(f"[{name}] combo-delta convergence  trend={trend}")
        ax[0].legend()
        ax[1].bar(range(1, DIM + 1), lam)
        ax[1].set_xlabel("i"); ax[1].set_ylabel("lambda_i")
        ax[1].set_title(f"Lyapunov spectrum  d_eff={d_eff:.2f}")
        fig.tight_layout(); fig.savefig(png, dpi=130)
        print(f"PNG  -> {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
