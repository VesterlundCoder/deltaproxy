"""
pslq_focused.py
===============
Aggressive, z-AWARE PSLQ on the top-delta companion candidates. Hypergeometric
CMF limits are naturally functions of the argument z, so besides the global
constant library we test z-dependent atoms: log(1-z), log(1+z), polylogs
Li_2(z), Li_3(z), sqrt(1-4z), 1/(1-z), etc. Larger maxcoeff, higher precision.

Usage:
  python3 pslq_focused.py --in 4f3_candidates/pslq_4f3.jsonl --dim 4 \
      --topk 400 --dps 240 --maxcoeff 100000000 --N 110 \
      --out 4f3_candidates/pslq_4f3_focused.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmf_generic as cg
from pslq_companion import delta_and_limit, const_library


def z_atoms(zn, zd):
    z = mp.mpf(zn) / zd
    at = {"z": z, "z^2": z * z, "1/(1-z)": 1 / (1 - z) if z != 1 else mp.nan}
    for nm, v in [("log(1-z)", 1 - z), ("log(1+z)", 1 + z), ("log|z|", abs(z))]:
        if v > 0:
            at[nm] = mp.log(v)
    for nm, v in [("sqrt(1-4z)", 1 - 4 * z), ("sqrt(1+4z)", 1 + 4 * z),
                  ("sqrt(1-z)", 1 - z), ("sqrt(1-z^2)", 1 - z * z)]:
        if v >= 0:
            at[nm] = mp.sqrt(v)
    try:
        at["Li2(z)"] = mp.polylog(2, z)
        at["Li3(z)"] = mp.polylog(3, z)
    except Exception:
        pass
    return at


def identify(L, zn, zd, maxcoeff, maxsteps):
    one = mp.mpf(1)
    hits = []

    def rel(vec, names, typ):
        try:
            r = mp.pslq(vec, maxcoeff=maxcoeff, maxsteps=maxsteps)
        except Exception:
            return
        if not r or all(c == 0 for c in r):
            return
        li = names.index("L")
        if r[li] == 0:
            return
        hits.append({"type": typ, "names": names, "coeffs": [int(c) for c in r]})

    rel([one, L], ["1", "L"], "rational")
    rel([one, L, L * L], ["1", "L", "L^2"], "quadratic")
    rel([one, L, L * L, L ** 3], ["1", "L", "L^2", "L^3"], "cubic")
    lib = dict(const_library())
    lib.update(z_atoms(zn, zd))
    for name, c in lib.items():
        if c is None or not mp.isfinite(c):
            continue
        rel([one, L, c], ["1", "L", name], "linear:" + name)
    # curated multi-term (transcendental cores)
    for names in [["1", "L", "pi", "pi^2"], ["1", "L", "pi^3", "zeta3"],
                  ["1", "L", "log(1-z)", "log(1+z)"], ["1", "L", "Li2(z)", "pi^2"],
                  ["1", "L", "z", "sqrt(1-4z)"]]:
        vec = []
        ok = True
        for nm in names:
            if nm == "1":
                vec.append(one)
            elif nm == "L":
                vec.append(L)
            else:
                v = lib.get(nm)
                if v is None or not mp.isfinite(v):
                    ok = False
                    break
                vec.append(v)
        if ok:
            rel(vec, names, "multi")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--topk", type=int, default=400)
    ap.add_argument("--dps", type=int, default=240)
    ap.add_argument("--N", type=int, default=110)
    ap.add_argument("--maxcoeff", type=int, default=100000000)
    ap.add_argument("--maxsteps", type=int, default=50000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mp.mp.dps = args.dps
    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    rows = [r for r in rows if r.get("delta_exact") is not None]
    rows.sort(key=lambda r: -r["delta_exact"])
    rows = rows[:args.topk]
    print(f"[focused] top {len(rows)} by delta  dps={args.dps} N={args.N} "
          f"maxcoeff={args.maxcoeff}", flush=True)

    found = []
    for i, r in enumerate(rows, 1):
        d, L, _ = delta_and_limit(r["shift"], r["dir"], r["z_num"], r["z_den"],
                                  args.dim, args.N)
        if L is None:
            continue
        hits = identify(L, r["z_num"], r["z_den"], args.maxcoeff, args.maxsteps)
        nontrivial = [h for h in hits if h["type"] != "rational"]
        if nontrivial:
            rec = {"gid": r.get("gid"), "delta": d, "z": f"{r['z_num']}/{r['z_den']}",
                   "L": mp.nstr(L, 50), "shift": r["shift"], "dir": r["dir"],
                   "hits": hits}
            found.append(rec)
            print(f"  HIT delta={d:.4f} z={r['z_num']}/{r['z_den']} "
                  f"L={mp.nstr(L,20)}  {[h['type'] for h in nontrivial]}", flush=True)
        if i % 50 == 0:
            print(f"  ...{i}/{len(rows)}  hits so far={len(found)}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"topk": args.topk, "dps": args.dps, "maxcoeff": args.maxcoeff,
                   "n_hits": len(found), "hits": found}, f, indent=2)
    print(f"\n[focused] {len(found)} candidates with non-rational relations -> {args.out}")


if __name__ == "__main__":
    main()
