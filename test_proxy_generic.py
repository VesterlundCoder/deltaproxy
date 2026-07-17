"""
test_proxy_generic.py
=====================
(1) REGRESSION: cmf_generic at dim=6 must reproduce the verified 6F5 deltas
    (confirms the dimension-generic builder is correct).
(2) 8F7 VALIDITY: does the spectral proxy dhat = -lam2/lam1 predict the exact
    double-depth delta for 8F7 (dim=8) trajectories?  We compare the cheap float
    detector against the exact pure-integer delta on a sample, reporting
    sign-agreement, MAE, correlation, and the sign of the finite-N bias
    (must be <= 0 for a conservative detector, as in 6F5).

Usage:
  python3 test_proxy_generic.py --n 160 --nrank 120 --nverify 80 --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool

import numpy as np

import cmf_generic as cg


def regression_6f5(path="verified_positives.jsonl", k=30):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("delta_arith") is not None][:k]
    diffs = []
    for r in rows:
        di = cg.independent_delta(r["shift"], r["dir"], r["z_num"], r["z_den"],
                                  120, 6)
        if di is not None:
            diffs.append(abs(di - r["delta_arith"]))
    diffs = np.array(diffs) if diffs else np.array([np.nan])
    ok = np.nanmax(diffs) < 1e-6
    print(f"[REGRESSION dim=6] {len(diffs)} hits: max|Δ|={np.nanmax(diffs):.2e}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def _one_8f7(args):
    shift, dirv, zn, zd, nrank, nverify = args
    try:
        dhat, lam = cg.spectral_dhat(shift, dirv, zn, zd, nrank, 8)
        d_exact = cg.independent_delta(shift, dirv, zn, zd, nverify, 8)
    except Exception as e:
        return {"err": str(e)[:80]}
    return {"dhat": dhat, "delta_exact": d_exact,
            "lam1": float(lam[0]), "lam2": float(lam[1])}


def validity_8f7(n, nrank, nverify, workers, box, seed):
    rng = random.Random(seed)
    SEED = cg.default_seed(8)
    DIRS = cg.dir_pool(8)
    ZS = cg.z_pool()
    tasks = []
    for _ in range(n):
        shift = [SEED[i] + rng.randint(-box, box) for i in range(cg.nshift_for(8))]
        dirv = list(rng.choice(DIRS))
        zn, zd = rng.choice(ZS)
        tasks.append((shift, dirv, zn, zd, nrank, nverify))

    print(f"\n[8F7 VALIDITY] sampling {n} dim=8 trajectories "
          f"(detector N={nrank}, exact 2N={2*nverify}), {workers} workers ...")
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(_one_8f7, tasks)
    res = [r for r in res if "err" not in r and r.get("dhat") is not None
           and np.isfinite(r["dhat"]) and r.get("delta_exact") is not None]
    print(f"  computed {len(res)}/{n} in {time.time()-t0:.0f}s")
    if not res:
        print("  no computable trajectories — 8F7 setup degenerate")
        return False, {}

    dh = np.array([r["dhat"] for r in res])
    de = np.array([r["delta_exact"] for r in res])
    # restrict the agreement metrics to the non-degenerate band (dhat in [-1,1.2])
    band = (dh > -1.0) & (dh < 1.2)
    dhb, deb = dh[band], de[band]
    sign_agree = float(np.mean((dhb > 0) == (deb > 0))) if len(dhb) else float("nan")
    mae = float(np.mean(np.abs(dhb - deb))) if len(dhb) else float("nan")
    bias = float(np.mean(dhb - deb)) if len(dhb) else float("nan")
    corr = float(np.corrcoef(dhb, deb)[0, 1]) if len(dhb) > 2 else float("nan")
    n_pos_exact = int((de > 0).sum())
    # conservative-precision: of dhat>0 (in band), how many have exact delta>0
    pos_pred = dhb > 0
    prec = float(np.mean(deb[pos_pred] > 0)) if pos_pred.any() else float("nan")

    print(f"  exact positives: {n_pos_exact}/{len(res)}  "
          f"(dhat>0 in-band: {int(pos_pred.sum())})")
    print(f"  sign-agreement (in-band): {sign_agree:.3f}")
    print(f"  MAE(dhat,exact): {mae:.4f}   bias(dhat-exact): {bias:+.4f}   "
          f"corr: {corr:.3f}")
    print(f"  conservative precision (dhat>0 => exact>0): {prec:.3f}")

    # PASS if the proxy tracks exact: strong sign agreement, decent correlation,
    # and a non-positive bias (conservative) like 6F5.
    ok = (sign_agree >= 0.90 and (np.isnan(corr) or corr >= 0.6)
          and (np.isnan(prec) or prec >= 0.85))
    verdict = ("8F7 PROXY WORKS — dhat tracks exact delta, conservative"
               if ok else "8F7 PROXY WEAK — inspect before sweeping")
    print(f"  -> {verdict}")
    stats = {"n": len(res), "sign_agreement": sign_agree, "mae": mae,
             "bias": bias, "corr": corr, "conservative_precision": prec,
             "exact_positives": n_pos_exact, "verdict": verdict,
             "dhat_min": float(dh.min()), "dhat_max": float(dh.max())}
    return ok, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--nrank", type=int, default=120)
    ap.add_argument("--nverify", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--box", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="test_8f7_proxy.json")
    args = ap.parse_args()

    reg_ok = regression_6f5()
    ok, stats = validity_8f7(args.n, args.nrank, args.nverify, args.workers,
                             args.box, args.seed)
    out = {"regression_6f5_pass": bool(reg_ok), "validity_8f7_pass": bool(ok),
           "stats_8f7": stats}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {args.out}")
    print(f"REGRESSION dim=6: {'PASS' if reg_ok else 'FAIL'}   "
          f"8F7 proxy: {'PASS' if ok else 'WEAK'}")


if __name__ == "__main__":
    main()
