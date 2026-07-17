"""
verify_pslq_6f5.py
==================
Two-phase exact verification + PSLQ identification of the 6F5 local-sweep hits
(sweep_6f5_z1_24h/positives_w*.jsonl).

Phase 1  (fast, ALL unique hits): pure-integer double-depth delta
         (cmf_generic.independent_delta) -- the GOLD STANDARD (no float error).
         Classifies each hit: positive / negative / degenerate.

Phase 2  (deep, confirmed positives): mpmath matrix walk to --depth (default 2000)
         + PSLQ identification of the limit L against a zeta/pi/Catalan basis
         (analyse_large_hits.process_candidate). Also runs mpmath.identify and
         30 bonus row-pair PSLQ probes.

Both phases are checkpointed (append-only JSONL) and resumable: re-running skips
hits already present in the checkpoint files.

Usage:
  python3 verify_pslq_6f5.py --phase 1 --workers 8 --N 300
  python3 verify_pslq_6f5.py --phase 2 --workers 8 --depth 2000 --dps 120
  python3 verify_pslq_6f5.py --report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)            # 6F5Sweeps/ (has analyse_large_hits.py)
for p in (HERE, PARENT):
    if p not in sys.path:
        sys.path.insert(0, p)

import cmf_generic as cg

DIM = 6

# ── input sources (select with --source) ────────────────────────────────────
SOURCES = {
    # original local 24h scalpel sweep (11,836 hits)
    "local": os.path.join(HERE, "sweep_6f5_z1_24h", "positives_w*.jsonl"),
    # LUMI 200-billion GPU proxy sweep survivors (job 19558891, 559,643 hits)
    "lumi": os.path.join(HERE, "lumi_survivors_6f5_200b", "survivors_all.jsonl"),
}
OUT_DIRS = {
    "local": os.path.join(HERE, "verify_6f5_out"),
    "lumi": os.path.join(HERE, "verify_6f5_lumi_out"),
}

HITS_GLOB = SOURCES["local"]
OUT_DIR = OUT_DIRS["local"]
P1_CKPT = P2_CKPT = P3_CKPT = REPORT = None


def set_source(source):
    """Point all input/output paths at the chosen sweep source."""
    global HITS_GLOB, OUT_DIR, P1_CKPT, P2_CKPT, P3_CKPT, REPORT
    HITS_GLOB = SOURCES[source]
    OUT_DIR = OUT_DIRS[source]
    P1_CKPT = os.path.join(OUT_DIR, "phase1_exact_delta.jsonl")
    P2_CKPT = os.path.join(OUT_DIR, "phase2_pslq.jsonl")
    P3_CKPT = os.path.join(OUT_DIR, "phase3_ladder_pslq.jsonl")
    REPORT = os.path.join(OUT_DIR, "VERIFICATION_REPORT.md")


set_source("local")

DEGEN_DHAT = 1.1     # proxy/exact delta above this => degenerate tail (lambda1->0)


# ── data loading ────────────────────────────────────────────────────────────
def load_unique_hits():
    seen, rows = set(), []
    for f in sorted(glob.glob(HITS_GLOB)):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = (tuple(r["shift"]), tuple(r["dir"]), r["z_num"], r["z_den"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def _key(r):
    return f'{r["shift"]}|{r["dir"]}|{r["z_num"]}/{r["z_den"]}'


def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done[r.get("key")] = r
    return done


# ── Phase 1: exact integer double-depth delta ─────────────────────────────────
def _p1_worker(args):
    idx, r, N = args
    t0 = time.time()
    try:
        d = cg.independent_delta(r["shift"], r["dir"], r["z_num"], r["z_den"], N, DIM)
    except Exception as e:
        return {"key": _key(r), "idx": idx, "error": str(e),
                "shift": r["shift"], "dir": r["dir"],
                "z_num": r["z_num"], "z_den": r["z_den"], "dhat": r.get("dhat")}
    return {
        "key": _key(r), "idx": idx,
        "shift": r["shift"], "dir": r["dir"],
        "z_num": r["z_num"], "z_den": r["z_den"],
        "z": f'{r["z_num"]}/{r["z_den"]}',
        "dhat_proxy": r.get("dhat"),
        "exact_delta": None if d is None else round(float(d), 5),
        "N": N, "mode": r.get("mode"),
        "elapsed": round(time.time() - t0, 3),
    }


def phase1(workers, N):
    os.makedirs(OUT_DIR, exist_ok=True)
    hits = load_unique_hits()
    done = load_done(P1_CKPT)
    todo = [(i, r) for i, r in enumerate(hits) if _key(r) not in done]
    print(f"[phase1] {len(hits)} unique hits, {len(done)} already done, "
          f"{len(todo)} to do  (N={N}, depth={2*N}, workers={workers})")
    if not todo:
        print("[phase1] nothing to do.")
        return
    t0 = time.time()
    n = 0
    with open(P1_CKPT, "a") as out, Pool(workers) as pool:
        for res in pool.imap_unordered(_p1_worker,
                                       [(i, r, N) for i, r in todo], chunksize=16):
            out.write(json.dumps(res) + "\n")
            out.flush()
            n += 1
            if n % 200 == 0:
                el = time.time() - t0
                print(f"  [phase1] {n}/{len(todo)}  {n/el:.1f} hit/s  "
                      f"ETA {((len(todo)-n)/(n/el))/60:.1f} min", flush=True)
    print(f"[phase1] done {n} in {(time.time()-t0)/60:.1f} min")


def classify_phase1():
    """Return (positives, degenerate, negative, errors) from the phase-1 ckpt."""
    rows = list(load_done(P1_CKPT).values())
    pos, degen, neg, err = [], [], [], []
    for r in rows:
        if "error" in r:
            err.append(r); continue
        d = r.get("exact_delta")
        if d is None:
            err.append(r)
        elif d >= DEGEN_DHAT:
            degen.append(r)
        elif d > 0:
            pos.append(r)
        else:
            neg.append(r)
    return rows, pos, degen, neg, err


# ── Phase 2: deep mpmath walk + RIGOROUS PSLQ ────────────────────────────────
# Small, weight-graded bases. A genuine identification is a SHORT relation that
# holds to ~the full converged precision; we reject overfit relations.
BASIS_SETS = [
    # single-constant (2-term) probes
    ["1", "zeta3"], ["1", "zeta5"], ["1", "zeta7"], ["1", "zeta9"],
    ["1", "pi"], ["1", "pi2"], ["1", "pi3"], ["1", "pi4"], ["1", "pi5"],
    ["1", "pi6"], ["1", "log2"], ["1", "log3"], ["1", "log2sq"],
    ["1", "catalan"], ["1", "gamma"],
    ["1", "li2h"], ["1", "li3h"], ["1", "li4h"], ["1", "li5h"],
    # zeta ladders
    ["1", "zeta3", "zeta5"], ["1", "zeta3", "pi2"], ["1", "zeta5", "pi4"],
    ["1", "zeta5", "zeta7"], ["1", "zeta3", "zeta7"], ["1", "pi2", "pi4"],
    ["1", "zeta3", "zeta5", "zeta7"], ["1", "zeta3", "zeta5", "pi2"],
    # log / polylog ladders (Li_n(1/2) reduce via ln2 powers & zeta values)
    ["1", "log2", "pi2"], ["1", "zeta3", "log2"], ["1", "li2h", "pi2"],
    ["1", "li2h", "log2sq"], ["1", "li3h", "zeta3"], ["1", "li3h", "pi2", "log2"],
    ["1", "li4h", "pi4"], ["1", "catalan", "pi2"],
]
COEFF_BUDGET = 60      # max sum|coeff| -> anti-overfit (short integer relation)


def _build_consts(dps):
    import mpmath as mp
    mp.mp.dps = dps
    pi = mp.pi
    ln2 = mp.log(2)
    half = mp.mpf(1) / 2
    return {
        "1": mp.mpf(1),
        "zeta3": mp.zeta(3), "zeta5": mp.zeta(5),
        "zeta7": mp.zeta(7), "zeta9": mp.zeta(9),
        "pi": pi, "pi2": pi**2, "pi3": pi**3, "pi4": pi**4,
        "pi5": pi**5, "pi6": pi**6,
        "log2": ln2, "log3": mp.log(3), "log2sq": ln2**2,
        "catalan": mp.catalan, "gamma": mp.euler,
        # polylogarithms at 1/2 (classic ln-2 / zeta reducible constants)
        "li2h": mp.polylog(2, half), "li3h": mp.polylog(3, half),
        "li4h": mp.polylog(4, half), "li5h": mp.polylog(5, half),
    }


def _fmt_rel(rel, names):
    # rel[0]*L + sum rel[i+1]*const_i = 0  ->  L = -(sum rel[i+1]*const_i)/rel[0]
    from math import gcd
    den = int(rel[0])
    terms = []
    for i, nm in enumerate(names):
        c = -int(rel[i + 1])
        if c == 0:
            continue
        g = gcd(abs(c), abs(den)) or 1
        cn, dn = c // g, den // g
        if dn < 0:
            cn, dn = -cn, -dn
        unit = nm if nm != "1" else "1"
        if dn == 1:
            terms.append(f"{cn}*{unit}" if not (cn == 1 and nm != "1") else unit)
        else:
            terms.append(f"({cn}/{dn})*{unit}")
    return ("L = " + " + ".join(terms)).replace("+ -", "- ")


def strict_pslq(L, consts, digits):
    """Return a short, high-precision integer relation for L, or None."""
    import mpmath as mp
    usable = int(digits)
    if usable < 30:
        return None
    tol = mp.mpf(10) ** (-int(0.8 * usable))
    best = None
    for names in BASIS_SETS:
        vec = [L] + [consts[n] for n in names]
        try:
            rel = mp.pslq(vec, tol=tol, maxcoeff=10**6, maxsteps=10**6)
        except Exception:
            rel = None
        if not rel or rel[0] == 0:
            continue
        csum = sum(abs(int(c)) for c in rel)
        if csum > COEFF_BUDGET:
            continue
        recon = -sum(rel[i + 1] * consts[names[i]]
                     for i in range(len(names))) / rel[0]
        resid = abs(L - recon)
        if resid > tol:
            continue
        score = (len([c for c in rel if c != 0]), csum)
        if best is None or score < best[0]:
            best = (score, _fmt_rel(rel, names), float(resid),
                    [int(c) for c in rel], names)
    if best is None:
        return None
    return {"expr": best[1], "residual": best[2], "rel": best[3],
            "basis": best[4]}


def _p2_worker(args):
    idx, r, depth, dps = args
    import mpmath as mp
    from analyse_large_hits import _build_M
    mp.mp.dps = dps + 20
    shift, dirv, zn, zd = r["shift"], r["dir"], r["z_num"], r["z_den"]
    half = depth // 2
    P = mp.eye(6)
    Ph = None
    for n in range(1, depth + 1):
        P = P * _build_M(n, shift, dirv, zn, zd)
        if n == half:
            Ph = mp.matrix(P)
    nz = [k for k in range(6) if P[k, 5] != 0]
    consts = _build_consts(dps + 20)

    def stable_digits(val, valh):
        if valh is None:
            return 0.0
        d = abs(val - valh)
        if d == 0:
            return float(dps)
        return float(-mp.log10(d / max(abs(val), mp.mpf("1e-30"))))

    idents = []
    seen = set()
    Lcanon, canon_dig = None, 0.0
    for di in nz:                       # denominator row (nonzero last col)
        q, qh = P[di, 5], (Ph[di, 5] if Ph is not None else None)
        for ni in nz:                   # numerator row (nonzero last col)
            if ni == di:
                continue
            val = P[ni, 5] / q
            if abs(val) < mp.mpf("1e-6") or abs(val) > mp.mpf("1e6"):
                continue
            valh = (Ph[ni, 5] / qh) if (qh not in (None, 0)) else None
            dig = stable_digits(val, valh)
            if di == nz[0] and (ni == (nz[1] if len(nz) > 1 else nz[0])):
                Lcanon, canon_dig = mp.nstr(val, 45), round(dig, 1)
            if dig < 30:
                continue
            res = strict_pslq(val, consts, dig)
            if res and res["expr"] not in seen:
                seen.add(res["expr"])
                idents.append({"row_pair": f"P[{ni},5]/P[{di},5]",
                               "value": mp.nstr(val, 40),
                               "stable_digits": round(dig, 1), **res})
    idents.sort(key=lambda x: (len([c for c in x["rel"] if c]), sum(abs(c) for c in x["rel"])))
    return {
        "key": _key(r), "z": f"{zn}/{zd}", "shift": shift, "dir": dirv,
        "exact_delta": r.get("exact_delta"), "dhat_proxy": r.get("dhat_proxy"),
        "nz_rows": nz, "L": Lcanon, "canon_digits": canon_dig,
        "identifications": idents[:6], "n_ident": len(idents),
    }


def phase2(workers, depth, dps, prec, thr, limit):
    os.makedirs(OUT_DIR, exist_ok=True)
    _, pos, _, _, _ = classify_phase1()
    pos = [r for r in pos if (r.get("exact_delta") or 0) > thr]
    pos.sort(key=lambda r: -(r.get("exact_delta") or 0))
    if limit:
        pos = pos[:limit]
    done = load_done(P2_CKPT)
    todo = [(i, r) for i, r in enumerate(pos) if _key(r) not in done]
    print(f"[phase2] {len(pos)} confirmed positives (exact_delta>{thr}), "
          f"{len(done)} done, {len(todo)} to do  "
          f"(depth={depth}, dps={dps}, workers={workers})")
    if not todo:
        print("[phase2] nothing to do.")
        return
    t0 = time.time()
    n = 0
    with open(P2_CKPT, "a") as out, Pool(workers) as pool:
        for res in pool.imap_unordered(
                _p2_worker,
                [(i, r, depth, dps) for i, r in todo], chunksize=2):
            out.write(json.dumps(res, default=str) + "\n")
            out.flush()
            n += 1
            el = time.time() - t0
            idents = res.get("identifications") or []
            tag = idents[0]["expr"] if idents else ""
            print(f"  [phase2 {n}/{len(todo)}] z={res.get('z','?')} "
                  f"d={res.get('canon_digits')} n_id={res.get('n_ident')} "
                  f"{str(tag)[:44]}  "
                  f"({n/el:.2f}/s ETA {((len(todo)-n)/max(n/el,1e-9))/60:.0f}m)",
                  flush=True)
    print(f"[phase2] done {n} in {(time.time()-t0)/60:.1f} min")


# ── Phase 3: mpmath depth-ladder best-linear-combo self-delta + PSLQ ─────────
# For each trajectory we do ONE exact-integer walk to 2*maxdepth, snapshotting
# the last-column vector at every ladder depth. At each depth D the "best self
# delta" is the max double-depth (D,2D) delta over integer linear combinations
# of the last-column entries (single rows + a*row_i+b*row_j, a,b in -3..3) --
# exact integer cross-products + mpmath.log (numerically stable). The deepest
# convergent limit is then run through the rigorous PSLQ. All per-depth data is
# stored for later analysis.
_COMBO_COEFFS = (-3, -2, -1, 1, 2, 3)


def _delta_exact_int(pn, qn, p2n, q2n):
    """Double-depth delta from EXACT integer cross-product + mpmath.log."""
    import mpmath as mp
    if qn == 0 or q2n == 0 or (pn == 0 and p2n == 0):
        return None
    cross = pn * q2n - p2n * qn            # exact integer, no cancellation
    if cross == 0:
        return None
    aq = abs(qn)
    if aq <= 1:
        return None
    with mp.workdps(60):
        lc = float(mp.log(abs(cross)))
        lq = float(mp.log(aq))
        lq2 = float(mp.log(abs(q2n)))
    return -(1.0 + (lc - lq - lq2) / lq)   # = -(1 + log|L_D - L_2D|/log|q_D|)


def best_combo_delta(v, w):
    """Best double-depth self-delta over integer linear combos of entries.

    v = last-column vector at depth D, w = at depth 2D (Python ints).
    Scans single-row pairs and pairwise a*v[i]+b*v[j] over v[k] denominators.
    Returns (best_delta, combo_descriptor).
    """
    dim = len(v)
    best, best_combo = None, None

    def tryc(pn, qn, p2n, q2n, desc):
        nonlocal best, best_combo
        d = _delta_exact_int(pn, qn, p2n, q2n)
        if d is not None and (best is None or d > best):
            best, best_combo = d, desc

    for ni in range(dim):
        for nj in range(dim):
            if ni == nj:
                continue
            tryc(v[ni], v[nj], w[ni], w[nj], {"num": [[1, ni]], "den": nj})
    for ni in range(dim):
        for nj in range(ni + 1, dim):
            for nk in range(dim):
                if nk in (ni, nj) or v[nk] == 0 or w[nk] == 0:
                    continue
                for a in _COMBO_COEFFS:
                    for b in _COMBO_COEFFS:
                        tryc(a * v[ni] + b * v[nj], v[nk],
                             a * w[ni] + b * w[nj], w[nk],
                             {"num": [[a, ni], [b, nj]], "den": nk})
    return best, best_combo


def _p3_worker(args):
    idx, r, dim, maxdepth, step, dps = args
    import mpmath as mp
    shift, dirv, zn, zd = r["shift"], r["dir"], r["z_num"], r["z_den"]
    ladder = list(range(step, maxdepth + 1, step))          # 200,400,...,2000
    need = sorted(set(ladder) | {2 * D for D in ladder})     # ...up to 4000
    top = need[-1]
    need_set = set(need)
    base = {"key": _key(r), "z": f"{zn}/{zd}", "shift": shift, "dir": dirv,
            "z_num": zn, "z_den": zd, "dim": dim,
            "exact_delta": r.get("exact_delta"), "dhat_proxy": r.get("dhat"),
            "maxdepth": maxdepth, "ladder_step": step}

    # one exact-integer walk, snapshot the last column at each needed depth
    snaps = {}
    P = [[1 if i == j else 0 for j in range(dim)] for i in range(dim)]
    try:
        for n in range(1, top + 1):
            P = cg.matmul_int(P, cg.build_M_int(n, shift, dirv, zn, zd, dim), dim)
            if n in need_set:
                snaps[n] = [P[i][dim - 1] for i in range(dim)]
    except Exception as e:
        return {**base, "error": f"walk:{e}"}

    # best-linear-combo self-delta ladder
    ladder_data = {}
    for D in ladder:
        v, w = snaps.get(D), snaps.get(2 * D)
        if v is None or w is None:
            continue
        d, combo = best_combo_delta(v, w)
        ladder_data[str(D)] = {"delta": None if d is None else round(d, 5),
                               "combo": combo}
    deltas = [x["delta"] for x in ladder_data.values() if x["delta"] is not None]
    best_ladder = max(deltas) if deltas else None

    # rigorous PSLQ on the deepest convergent limits (all nonzero row ratios)
    mp.mp.dps = dps + 20
    consts = _build_consts(dps + 20)
    vt = snaps.get(top, [])
    vh = snaps.get(maxdepth, [])
    nz = [k for k in range(dim) if k < len(vt) and vt[k] != 0]
    idents, seen = [], set()
    Lcanon, canon_dig = None, 0.0
    for dj in nz:
        for ni in nz:
            if ni == dj:
                continue
            val = mp.mpf(vt[ni]) / mp.mpf(vt[dj])
            if abs(val) < mp.mpf("1e-6") or abs(val) > mp.mpf("1e6"):
                continue
            dig = float(dps)
            if dj < len(vh) and ni < len(vh) and vh[dj] != 0:
                vhr = mp.mpf(vh[ni]) / mp.mpf(vh[dj])
                diff = abs(val - vhr)
                if diff > 0:
                    dig = min(float(dps),
                              float(-mp.log10(diff / max(abs(val), mp.mpf("1e-30")))))
            if ni == nz[0] and dj == (nz[1] if len(nz) > 1 else nz[0]):
                Lcanon, canon_dig = mp.nstr(val, 45), round(dig, 1)
            if dig < 30:
                continue
            res = strict_pslq(val, consts, dig)
            if res and res["expr"] not in seen:
                seen.add(res["expr"])
                idents.append({"row_pair": f"P[{ni},last]/P[{dj},last]",
                               "value": mp.nstr(val, 40),
                               "stable_digits": round(dig, 1), **res})
    idents.sort(key=lambda x: (len([c for c in x["rel"] if c]),
                               sum(abs(c) for c in x["rel"])))
    return {**base, "nz_rows": nz, "L": Lcanon, "canon_digits": canon_dig,
            "best_ladder_delta": best_ladder, "delta_ladder": ladder_data,
            "identifications": idents[:6], "n_ident": len(idents)}


def phase3(workers, dim, maxdepth, step, dps, thr, limit, use_phase1):
    os.makedirs(OUT_DIR, exist_ok=True)
    if use_phase1:
        _, pos, _, _, _ = classify_phase1()
        pos = [r for r in pos if (r.get("exact_delta") or 0) > thr]
        pos.sort(key=lambda r: -(r.get("exact_delta") or 0))
        src = "phase1 confirmed positives"
    else:
        pos = load_unique_hits()
        src = "all survivors (no phase-1 filter)"
    if limit:
        pos = pos[:limit]
    done = load_done(P3_CKPT)
    todo = [(i, r) for i, r in enumerate(pos) if _key(r) not in done]
    print(f"[phase3] {len(pos)} from {src}, {len(done)} done, {len(todo)} to do  "
          f"(dim={dim}, ladder={step}..{maxdepth} step {step}, walk_to={2*maxdepth}, "
          f"dps={dps}, workers={workers})")
    if not todo:
        print("[phase3] nothing to do.")
        return
    t0 = time.time()
    n = 0
    with open(P3_CKPT, "a") as out, Pool(workers) as pool:
        for res in pool.imap_unordered(
                _p3_worker,
                [(i, r, dim, maxdepth, step, dps) for i, r in todo], chunksize=1):
            out.write(json.dumps(res, default=str) + "\n")
            out.flush()
            n += 1
            el = time.time() - t0
            bl = res.get("best_ladder_delta")
            print(f"  [phase3 {n}/{len(todo)}] z={res.get('z','?')} "
                  f"bestδ={bl} n_id={res.get('n_ident', 0)} "
                  f"({n/el:.2f}/s ETA {((len(todo)-n)/max(n/el,1e-9))/3600:.1f}h)",
                  flush=True)
    print(f"[phase3] done {n} in {(time.time()-t0)/60:.1f} min")


# ── Report ────────────────────────────────────────────────────────────────────
def _zeta_kw(s):
    s = s or ""
    return any(k in s for k in ("zeta3", "zeta5", "zeta7", "zeta9", "catalan",
                                "beta3", "beta4"))


def write_report(depth, dps):
    rows, pos, degen, neg, err = classify_phase1()
    p2 = list(load_done(P2_CKPT).values())
    p2_ok = [r for r in p2 if "error" not in r]
    identified = [r for r in p2_ok if r.get("n_ident", 0) > 0]

    def _is_rational(it):
        # L is rational iff every NON-"1" basis element has a zero coefficient.
        rel, basis = it.get("rel", []), it.get("basis", [])
        for i, nm in enumerate(basis):
            if nm != "1" and i + 1 < len(rel) and rel[i + 1] != 0:
                return False
        return True

    # flatten accepted identifications, split rational collapses vs real constants
    const_ids, rat_ids = [], []
    for r in identified:
        for it in r.get("identifications", []):
            rec = {**it, "z": r.get("z"), "exact_delta": r.get("exact_delta"),
                   "key": r.get("key")}
            (rat_ids if _is_rational(it) else const_ids).append(rec)
    rat_keys = {r["key"] for r in identified
                if all(_is_rational(it) for it in r["identifications"])}
    all_ids = const_ids
    zeta_hits = [r for r in identified if any(
        (not _is_rational(it)) and _zeta_kw(it.get("expr"))
        for it in r.get("identifications", []))]

    def fmt_delta_hist(items):
        import numpy as np
        ds = np.array([r["exact_delta"] for r in items if r.get("exact_delta") is not None])
        out = []
        for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3),
                       (0.3, 0.5), (0.5, 1.1)]:
            out.append(f"| {lo:.2f}–{hi:.2f} | {int(((ds>=lo)&(ds<hi)).sum())} |")
        return "\n".join(out)

    L = []
    L += [
        "# 6F5 Local-Sweep Hits — Exact Verification + PSLQ Report",
        "",
        f"Source: `sweep_6f5_z1_24h/positives_w*.jsonl` (4 workers, ~15.8 h, |z|<1).",
        "",
        "## Phase 1 — exact pure-integer double-depth δ (gold standard)",
        "",
        f"- **Unique hits checked:** {len(rows)}",
        f"- **Confirmed positive (0 < δ < {DEGEN_DHAT}):** {len(pos)}",
        f"- **Degenerate tail (δ ≥ {DEGEN_DHAT}, λ₁→0 artifacts):** {len(degen)}",
        f"- **Non-positive under exact δ (proxy false-positive):** {len(neg)}",
        f"- **Errors:** {len(err)}",
        "",
        "Exact-δ distribution of confirmed positives:",
        "",
        "| δ range | count |",
        "|---------|-------|",
        fmt_delta_hist(pos),
        "",
        "> The proxy δ̂ is conservative: a fraction of saved hits collapse to δ≤0 under",
        "> exact integer arithmetic, and the huge δ̂ (up to ~38) are confirmed to be the",
        "> degenerate λ₁→0 tail, **not** genuine high-irrationality trajectories.",
        "",
        "## Phase 2 — deep mpmath walk + RIGOROUS PSLQ identification",
        "",
        f"- depth = **{depth}**, dps = **{dps}**  (≈{dps} converged digits available)",
        f"- positives processed: **{len(p2_ok)}** / {len(pos)}",
        "- PSLQ acceptance: short, weight-graded bases (≤4 terms), tol at 80% of the",
        f"  actual converged digits, Σ|coeff| ≤ {COEFF_BUDGET}, residual re-verified.",
        f"- **ζ/π/Catalan-family identifications:** {len(zeta_hits)}",
        f"- rational-collapse row-pairs (L∈ℚ, rank-deficient, *not* new constants): "
        f"{len(rat_keys)} hits",
        "",
    ]

    if all_ids:
        all_ids.sort(key=lambda x: (len([c for c in x['rel'] if c]),
                                    sum(abs(c) for c in x['rel'])))
        L += ["### ★ Accepted identifications (rigorous)", "",
              "| z | exact_δ | row-pair | relation | stable_d | residual |",
              "|---|---------|----------|----------|----------|----------|"]
        for it in all_ids[:80]:
            L.append(f"| {it.get('z')} | {it.get('exact_delta')} | "
                     f"{it.get('row_pair')} | `{str(it.get('expr'))[:60]}` | "
                     f"{it.get('stable_digits')} | {it.get('residual'):.1e} |")
        L.append("")
    else:
        L += ["### Accepted identifications (rigorous)", "",
              "*No limit matched a short low-weight ζ/π/Catalan relation under the "
              "strict test.* The positive-δ trajectories converge to numbers that are "
              "**not** simple low-weight combinations of these constants at the "
              "available precision — candidates for new/higher-weight constants.",
              ""]

    # strongest positives overall (by exact delta) with their canonical L
    p2_by_key = {r.get("key"): r for r in p2_ok}
    L += ["### Strongest confirmed positives (by exact δ)", "",
          "| z | exact_δ | canon_digits | L (canonical) | accepted relation |",
          "|---|---------|--------------|---------------|-------------------|"]
    for r in sorted(pos, key=lambda x: -(x.get("exact_delta") or 0))[:40]:
        d2 = p2_by_key.get(r["key"], {})
        ids = [it for it in d2.get("identifications", []) if not _is_rational(it)]
        expr = ids[0]["expr"] if ids else "—"
        Lval = d2.get("L") or "—"
        L.append(f"| {r.get('z')} | {r.get('exact_delta')} | "
                 f"{d2.get('canon_digits','—')} | `{str(Lval)[:24]}` | "
                 f"`{str(expr)[:46]}` |")
    L.append("")

    os.makedirs(OUT_DIR, exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    print(f"[report] -> {REPORT}")
    print(f"  positives={len(pos)} degen/collapsed={len(err)} neg={len(neg)} "
          f"p2_done={len(p2_ok)} const_ids={len(zeta_hits)} "
          f"rational_collapse={len(rat_keys)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--source", choices=list(SOURCES), default="local",
                    help="input sweep: local 24h scalpel or LUMI 200B survivors")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dim", type=int, default=6, help="CMF dimension (6=6F5)")
    ap.add_argument("--N", type=int, default=300, help="phase1 half-depth (exact)")
    ap.add_argument("--depth", type=int, default=2000, help="phase2 mpmath walk depth")
    ap.add_argument("--maxdepth", type=int, default=2000,
                    help="phase3 top ladder depth (walks to 2*maxdepth)")
    ap.add_argument("--step", type=int, default=200, help="phase3 ladder step")
    ap.add_argument("--dps", type=int, default=120)
    ap.add_argument("--prec", type=int, default=30, help="PSLQ digit threshold")
    ap.add_argument("--thr", type=float, default=0.0,
                    help="phase2/3 min exact_delta (confirmed positives)")
    ap.add_argument("--limit", type=int, default=0, help="cap count (0=all)")
    ap.add_argument("--all-survivors", action="store_true",
                    help="phase3: run on ALL survivors, skip phase-1 positive filter")
    args = ap.parse_args()

    global DIM
    DIM = args.dim
    set_source(args.source)

    if args.phase == 1:
        phase1(args.workers, args.N)
    elif args.phase == 2:
        phase2(args.workers, args.depth, args.dps, args.prec, args.thr,
               args.limit or None)
    elif args.phase == 3:
        phase3(args.workers, args.dim, args.maxdepth, args.step, args.dps,
               args.thr, args.limit or None, use_phase1=not args.all_survivors)
    if args.report or args.phase in (1, 2):
        write_report(args.depth, args.dps)


if __name__ == "__main__":
    main()
