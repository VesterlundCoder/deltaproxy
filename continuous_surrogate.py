"""
continuous_surrogate.py
========================
Turn the discrete harvest data (shift, dir, z, delta_hat) into a CONTINUOUS
surrogate function over parameter space, following the staged plan:

  Model B (regression):     delta_hat(theta)            -> ranking
  Model A (classifier):     p_+(theta) = P(delta>0)     -> discovery / level set
  (Model C, spectral E/Q, lives in spectral_delta.py / enrich.py.)

Critical: a level-set / boundary model needs BOTH positive and background
(negative) points. harvest.py now streams a random background sample; this
script reads positives + background together. Positives-only -> density model,
NOT a delta=0 boundary.

Stages:
  Fas 2  featurize positives + background  -> feature table (.csv/.parquet)
  Fas 3  3D surrogate  dhat ~ (log|z|, ||shift-SEED||, ||dir||_0)   + level-set plot
  Fas 4  full feature surrogate  dhat ~ (s1..s11, dir-features, log|z|) + classifier
  Fas 5  active-learning validation: model proposes -> spectral_dhat verifies ->
         measure enrichment factor vs uniform sampling.

Usage (tomorrow, once harvest_8h_*.jsonl exist):
  python3 continuous_surrogate.py \
      --positives harvest_8h_positives.jsonl \
      --background harvest_8h_background.jsonl \
      --outdir surrogate_out --propose 4000 --verify 200
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

from enrich import SEED_SHIFT, spectral_dhat
from discover import DIR_POOL, Z_POOL

SEED_SHIFT_ARR = np.array(SEED_SHIFT, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Fas 2 — IO + features
# ─────────────────────────────────────────────────────────────────────────────
def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def featurize_record(r):
    shift = np.array(r["shift"], dtype=float)
    dirv = np.array(r["dir"], dtype=float)
    z = r["z_num"] / r["z_den"]
    log_abs_z = math.log10(abs(z)) if z != 0 else -999.0
    feats = {
        "log_abs_z": log_abs_z,
        "z": z,
        "shift_norm": float(np.linalg.norm(shift - SEED_SHIFT_ARR)),
        "dir_l0": int(np.count_nonzero(dirv)),
        "dir_norm": float(np.linalg.norm(dirv)),
        "dhat": float(r["dhat"]),
        "label": int(r.get("label", int(r["dhat"] > 0.0))),
        "mode": r.get("mode", "unknown"),
        "z_num": r["z_num"], "z_den": r["z_den"],
    }
    for i in range(11):
        feats[f"s{i+1}"] = float(shift[i]) if i < len(shift) else 0.0
    for i in range(11):
        feats[f"d{i+1}"] = float(dirv[i]) if i < len(dirv) else 0.0
    return feats


def build_table(positives_path, background_path):
    pos = read_jsonl(positives_path)
    bg = read_jsonl(background_path)
    rows = [featurize_record(r) for r in pos + bg]
    return rows, len(pos), len(bg)


# ─────────────────────────────────────────────────────────────────────────────
# model helpers (sklearn)
# ─────────────────────────────────────────────────────────────────────────────
def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise SystemExit("scikit-learn required:  pip install scikit-learn pandas")


FEATS_3D = ["log_abs_z", "shift_norm", "dir_l0"]
FEATS_FULL = (["log_abs_z", "shift_norm", "dir_l0", "dir_norm"]
              + [f"s{i+1}" for i in range(11)]
              + [f"d{i+1}" for i in range(11)])


def _matrix(rows, cols):
    return np.array([[r[c] for c in cols] for r in rows], dtype=float)


def train_regressor(rows, cols, seed=42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    X = _matrix(rows, cols)
    y = np.array([r["dhat"] for r in rows], dtype=float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                      max_leaf_nodes=31, random_state=seed)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    return m, {"mae": float(mean_absolute_error(yte, pred)),
               "r2": float(r2_score(yte, pred)), "n": len(rows),
               "n_features": len(cols)}


def train_classifier(rows, cols, seed=42):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    X = _matrix(rows, cols)
    y = np.array([r["label"] for r in rows], dtype=int)
    if len(set(y.tolist())) < 2:
        return None, {"error": "only one class present — need background negatives"}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed,
                                          stratify=y)
    c = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                       max_leaf_nodes=31, random_state=seed)
    c.fit(Xtr, ytr)
    proba = c.predict_proba(Xte)[:, 1]
    pred = (proba > 0.5).astype(int)
    return c, {"precision": float(precision_score(yte, pred, zero_division=0)),
               "recall": float(recall_score(yte, pred, zero_division=0)),
               "auc": float(roc_auc_score(yte, proba)),
               "n": len(rows), "pos_frac": float(y.mean())}


# ─────────────────────────────────────────────────────────────────────────────
# Fas 3 — level-set plot of the 3D surrogate
# ─────────────────────────────────────────────────────────────────────────────
def plot_levelset(model3, rows, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    lz = np.array([r["log_abs_z"] for r in rows])
    sn = np.array([r["shift_norm"] for r in rows])
    dl0 = np.array([r["dir_l0"] for r in rows])
    x_grid = np.linspace(lz.min(), lz.max(), 120)
    y_grid = np.linspace(sn.min(), sn.max(), 120)
    XX, YY = np.meshgrid(x_grid, y_grid)
    u_vals = sorted(set(int(v) for v in dl0))[:4] or [1]
    n = len(u_vals)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), squeeze=False)
    for ax, u in zip(axes[0], u_vals):
        grid = np.column_stack([XX.ravel(), YY.ravel(),
                                np.full(XX.size, u, dtype=float)])
        ZZ = model3.predict(grid).reshape(XX.shape)
        cf = ax.contourf(XX, YY, ZZ, levels=30, cmap="viridis")
        ax.contour(XX, YY, ZZ, levels=[0.0], colors="white", linewidths=2)
        fig.colorbar(cf, ax=ax, label="predicted delta_hat")
        ax.set_xlabel("log10(|z|)")
        ax.set_ylabel("||shift - SEED||")
        ax.set_title(f"dir_l0 = {u}  (white = level set delta=0)")
    fig.tight_layout()
    path = os.path.join(outdir, "levelset_3d.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Fas 5 — active-learning enrichment validation
# ─────────────────────────────────────────────────────────────────────────────
def _candidate(rng, box=4):
    shift = [SEED_SHIFT[i] + rng.randint(-box, box) for i in range(11)]
    dirv = list(DIR_POOL[rng.randrange(len(DIR_POOL))])
    zn, zd = Z_POOL[rng.randrange(len(Z_POOL))]
    return shift, dirv, zn, zd


def _cand_features(shift, dirv, zn, zd, cols):
    rec = featurize_record({"shift": shift, "dir": dirv,
                            "z_num": zn, "z_den": zd, "dhat": 0.0})
    return [rec[c] for c in cols]


def enrichment_test(model, cols, n_propose, n_verify, nrank, seed=7):
    rng = random.Random(seed)
    cands = [_candidate(rng) for _ in range(n_propose)]
    Xc = np.array([_cand_features(*c, cols) for c in cands], dtype=float)
    scores = model.predict(Xc)
    order = np.argsort(-scores)
    top = [cands[i] for i in order[:n_verify]]
    rand_idx = rng.sample(range(n_propose), min(n_verify, n_propose))
    rand = [cands[i] for i in rand_idx]

    def verify(batch):
        pos = 0
        ok = 0
        for shift, dirv, zn, zd in batch:
            try:
                dh, _ = spectral_dhat(shift, dirv, zn, zd, nrank)
            except Exception:
                continue
            if not np.isfinite(dh):
                continue
            ok += 1
            pos += int(dh > 0.0)
        return pos, ok

    mp, mo = verify(top)
    rp, ro = verify(rand)
    p_model = mp / mo if mo else 0.0
    p_rand = rp / ro if ro else 0.0
    ef = (p_model / p_rand) if p_rand else float("inf")
    return {"n_propose": n_propose, "n_verify": n_verify, "nrank": nrank,
            "model_hit_rate": p_model, "random_hit_rate": p_rand,
            "enrichment_factor": ef,
            "model_pos": mp, "model_checked": mo,
            "random_pos": rp, "random_checked": ro}


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positives", default="harvest_8h_positives.jsonl")
    ap.add_argument("--background", default="harvest_8h_background.jsonl")
    ap.add_argument("--outdir", default="surrogate_out")
    ap.add_argument("--propose", type=int, default=4000)
    ap.add_argument("--verify", type=int, default=200)
    ap.add_argument("--nrank", type=int, default=120)
    ap.add_argument("--no-enrich", action="store_true")
    args = ap.parse_args()

    _require_sklearn()
    os.makedirs(args.outdir, exist_ok=True)

    rows, n_pos, n_bg = build_table(args.positives, args.background)
    print(f"Loaded {n_pos} positives + {n_bg} background = {len(rows)} rows")
    if not rows:
        raise SystemExit("No data — run harvest.py first.")
    if n_bg == 0:
        print("WARNING: no background/negative points — classifier and a real "
              "level set are NOT possible (density model only).")

    # write feature table (CSV; parquet if pandas+pyarrow available)
    cols_all = (["mode", "label", "dhat", "z", "log_abs_z", "shift_norm",
                 "dir_l0", "dir_norm", "z_num", "z_den"]
                + [f"s{i+1}" for i in range(11)] + [f"d{i+1}" for i in range(11)])
    csv_path = os.path.join(args.outdir, "harvest_features.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols_all) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols_all) + "\n")
    try:
        import pandas as pd
        pd.DataFrame(rows)[cols_all].to_parquet(
            os.path.join(args.outdir, "harvest_features.parquet"))
    except Exception:
        pass

    report = {"n_positives": n_pos, "n_background": n_bg, "n_rows": len(rows)}

    # Fas 3 — 3D surrogate
    m3, rep3 = train_regressor(rows, FEATS_3D)
    report["regressor_3d"] = rep3
    print(f"[3D reg]  MAE={rep3['mae']:.4f}  R2={rep3['r2']:.3f}")
    lp = plot_levelset(m3, rows, args.outdir)
    if lp:
        print(f"[3D reg]  level-set plot -> {lp}")

    # Fas 4 — full surrogate + classifier
    mf, repf = train_regressor(rows, FEATS_FULL)
    report["regressor_full"] = repf
    print(f"[full reg] MAE={repf['mae']:.4f}  R2={repf['r2']:.3f}  "
          f"({repf['n_features']} feats)")
    clf, repc = train_classifier(rows, FEATS_FULL)
    report["classifier_full"] = repc
    if clf is not None:
        print(f"[clf]     AUC={repc['auc']:.3f}  P={repc['precision']:.3f}  "
              f"R={repc['recall']:.3f}  pos_frac={repc['pos_frac']:.3f}")
    else:
        print(f"[clf]     skipped: {repc.get('error')}")

    # Fas 5 — enrichment validation of the full regressor
    if not args.no_enrich:
        print(f"[enrich]  proposing {args.propose}, verifying top {args.verify} "
              f"vs random with spectral_dhat ...")
        rep_e = enrichment_test(mf, FEATS_FULL, args.propose, args.verify,
                                args.nrank)
        report["enrichment"] = rep_e
        print(f"[enrich]  model hit-rate={rep_e['model_hit_rate']:.3f} "
              f"vs random={rep_e['random_hit_rate']:.3f}  "
              f"=> EF={rep_e['enrichment_factor']:.1f}x")

    out = os.path.join(args.outdir, "surrogate_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDONE. Report -> {out}   Features -> {csv_path}")


if __name__ == "__main__":
    main()
