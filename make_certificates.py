"""
make_certificates.py  —  PROOF LADDER V3 (CAS certificate)
==========================================================
For each top confirmed positive, emit a machine-readable verification dossier that
another researcher can re-run. Delta is triangulated across THREE independent engines:

  engine 1  arith   : enrich.arith_delta        (mpmath, build_M_mp construction)
  engine 2  integer : verify_independent         (pure Python int, polynomial-convolution esym)
  engine 3  sage    : verify.sage                (SageMath, ZZ matrices, RealField logs)

plus depth-stability (N = 120, 160, 240 via the integer engine).

Dossier layout (per candidate):
  verification/candidates/<id>/
    input.json          raw (shift, dir, z)
    verify.sage         independent SageMath re-verification (runnable standalone)
    sage_output.json    captured Sage result
    certificate.json    normal form, 3-engine deltas, depth stability, hashes,
                        versions, novelty, status_level
    README.md           human summary + how to reproduce

Usage:
  python3 make_certificates.py --in verified_positives.jsonl --topk 12 \
      --depths 120,160,240 --run-sage
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

from enrich import arith_delta
from verify_independent import independent_delta

PARENT = Path(__file__).resolve().parent.parent
HIT_FILES = ["znegsweep_hits_unique.jsonl", "large_hits_lirec.jsonl",
             "large_hits_positive_delta.jsonl", "large_hits_deep_rt.jsonl",
             "large_hits_self_lf_delta.jsonl"]

SAGE_TEMPLATE = r'''# verify.sage  —  INDEPENDENT SageMath re-verification (proof-ladder V3)
# Re-derives the 6F5 companion matrix over ZZ from scratch and computes the
# double-depth delta with exact integer products + high-precision logs.
import json
shift = {shift}
dirv  = {dirv}
z_num = {z_num}
z_den = {z_den}
N     = {N}
DIM   = 6

def esym(roots):
    R = PolynomialRing(ZZ, 't'); t = R.gen()
    p = prod(t + r for r in roots)        # prod (t + r_i)
    cs = list(p) + [0]*(len(roots)+1)     # low->high coeffs
    cs = cs[:len(roots)+1]
    return cs[::-1]                        # high->low = [e0, e1, ..., e_k]

def buildM(n):
    h_f, h_g = -z_num, z_den
    f = [shift[i] + n*dirv[i] + 1 for i in range(6)]
    g = [shift[6+j] + n*dirv[6+j] + 2 for j in range(5)]
    ef = esym(f)
    eg = esym(g) + [0]
    c = [0]*DIM
    c[0] = h_f*ef[6]
    for k in range(1, DIM):
        c[k] = h_g*eg[6-k] + h_f*ef[6-k]
    M = matrix(ZZ, DIM, DIM, 0)
    for r in range(1, DIM):
        M[r, r-1] = 1
    for r in range(DIM):
        M[r, DIM-1] = c[r]
    return M

P = identity_matrix(ZZ, DIM)
snapN = None
for n in range(1, 2*N+1):
    P = P * buildM(n)
    if n == N:
        snapN = copy(P)

RF = RealField(256)
best = None
for i in range(DIM):
    for j in range(DIM):
        if i == j:
            continue
        pn, qn = snapN[i, DIM-1], snapN[j, DIM-1]
        p2, q2 = P[i, DIM-1], P[j, DIM-1]
        if qn == 0 or q2 == 0:
            continue
        cross = pn*q2 - p2*qn
        if cross == 0:
            continue
        log_err = RF(abs(cross)).log() - RF(abs(qn)).log() - RF(abs(q2)).log()
        log_q = RF(abs(qn)).log()
        if log_q == 0:
            continue
        d = float(-(1 + log_err/log_q))
        if best is None or d > best:
            best = d

print(json.dumps({{"delta_sage": (None if best is None else float(best)),
                   "N": int(N),
                   "q_N_digits": int(len(str(abs(snapN[5, DIM-1])))) }}))
'''


def load_known():
    known = set()
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
            known.add((tuple(r.get("shift", [])), tuple(r.get("dir", []))))
    return known


def sha256_json(obj):
    return "sha256:" + hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normal_form(shift, dirv, z_num, z_den):
    g = math.gcd(abs(z_num), abs(z_den)) or 1
    sgn = -1 if z_den < 0 else 1
    zn, zd = sgn * z_num // g, abs(z_den) // g
    return {"shift": list(shift), "dir": list(dirv), "z": [zn, zd]}


def sage_version():
    try:
        out = subprocess.run(["sage", "--version"], capture_output=True,
                             text=True, timeout=60)
        return out.stdout.strip().splitlines()[0]
    except Exception as e:
        return f"unavailable ({e})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="verified_positives.jsonl")
    ap.add_argument("--outdir", default="verification")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--depths", default="120,160,240")
    ap.add_argument("--run-sage", action="store_true",
                    help="actually execute verify.sage (needs sage on PATH)")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    recs = [json.loads(l) for l in open(args.inp) if l.strip()]
    conf = [r for r in recs if r.get("delta_arith") is not None
            and r["delta_arith"] > 0]
    conf.sort(key=lambda r: -r["delta_arith"])
    conf = conf[:args.topk]
    known = load_known()

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mpmath": __import__("mpmath").__version__,
        "sympy": __import__("sympy").__version__,
        "sage": sage_version() if args.run_sage else "not run",
    }

    root = Path(args.outdir) / "candidates"
    root.mkdir(parents=True, exist_ok=True)
    index = []

    for rank, r in enumerate(conf, 1):
        shift, dirv = r["shift"], r["dir"]
        zn, zd = r["z_num"], r["z_den"]
        nf = normal_form(shift, dirv, zn, zd)
        cid = "CMF_6F5_%03d_%s" % (rank, sha256_json(nf)[7:19])
        cdir = root / cid
        cdir.mkdir(exist_ok=True)

        # engine 1 (mpmath) + engine 2 (integer) at N=120, depth stability via integer
        d_arith = arith_delta(shift, dirv, zn, zd, 120)
        d_int = {N: independent_delta(shift, dirv, zn, zd, N) for N in depths}

        (cdir / "input.json").write_text(json.dumps(
            {"shift": shift, "dir": dirv, "z_num": zn, "z_den": zd}, indent=2))

        sage_src = SAGE_TEMPLATE.format(shift=shift, dirv=dirv, z_num=zn,
                                        z_den=zd, N=120)
        (cdir / "verify.sage").write_text(sage_src)

        d_sage = None
        sage_out = {"status": "not run"}
        if args.run_sage:
            try:
                proc = subprocess.run(["sage", str(cdir / "verify.sage")],
                                      capture_output=True, text=True, timeout=600)
                line = [l for l in proc.stdout.splitlines()
                        if l.strip().startswith("{")]
                if line:
                    sage_out = json.loads(line[-1])
                    d_sage = sage_out.get("delta_sage")
                else:
                    sage_out = {"status": "no_json", "stderr": proc.stderr[-400:]}
            except Exception as e:
                sage_out = {"status": "error", "msg": str(e)[:200]}
        (cdir / "sage_output.json").write_text(json.dumps(sage_out, indent=2))

        # agreement
        engines = {"arith_mpmath_N120": d_arith,
                   "integer_python_N120": d_int.get(120),
                   "sage_ZZ_N120": d_sage}
        vals = [v for v in engines.values() if v is not None]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        is_new = (tuple(shift), tuple(dirv)) not in known

        cert = {
            "candidate_id": cid,
            "rank_by_delta": rank,
            "construction": "6F5_conservative_matrix_field_trajectory",
            "normal_form": nf,
            "normal_form_hash": sha256_json(nf),
            "input_hash": sha256_json({"shift": shift, "dir": dirv,
                                       "z_num": zn, "z_den": zd}),
            "delta_engines_N120": engines,
            "delta_engine_spread": spread,
            "delta_depth_stability_integer": {str(k): v for k, v in d_int.items()},
            "delta_hat_float_screen": r.get("dhat"),
            "is_positive_delta": all(v > 0 for v in vals),
            "novelty_new_vs_known_DB": is_new,
            "engines_agree_within_1e-6": spread < 1e-6,
            "software_versions": versions,
            "verification_level": ("V3" if (args.run_sage and d_sage is not None
                                            and spread < 1e-6) else "V2"),
        }
        cert_hash = sha256_json(cert)
        cert["certificate_hash"] = cert_hash
        (cdir / "certificate.json").write_text(json.dumps(cert, indent=2))

        readme = f"""# Verification dossier — {cid}

**6F5 CMF positive-delta trajectory** (proof-ladder certificate).

- normal form: shift={nf['shift']}, dir={nf['dir']}, z={nf['z'][0]}/{nf['z'][1]}
- delta (3 engines @ N=120): arith={d_arith:.6f}, integer={d_int.get(120):.6f}, """ \
            f"""sage={d_sage if d_sage is None else round(d_sage,6)}
- engine spread: {spread:.2e}  ({'AGREE' if spread<1e-6 else 'REVIEW'})
- depth stability (integer): {', '.join(f'N={k}:{v:.4f}' for k,v in d_int.items())}
- novelty (new vs known DB): {is_new}
- verification level: {cert['verification_level']}

## Reproduce
```bash
sage verify.sage            # independent CAS re-derivation over ZZ
python3 -c "from verify_independent import independent_delta; \\
  print(independent_delta({shift},{dirv},{zn},{zd},120))"
```
certificate_hash: {cert_hash}
"""
        (cdir / "README.md").write_text(readme)

        index.append({"candidate_id": cid, "rank": rank,
                      "delta": d_int.get(120), "z": f"{nf['z'][0]}/{nf['z'][1]}",
                      "spread": spread, "level": cert["verification_level"],
                      "new": is_new})
        print(f"[{rank:2d}] {cid}  d={d_int.get(120):.4f}  "
              f"spread={spread:.1e}  level={cert['verification_level']}  "
              f"new={is_new}", flush=True)

    (Path(args.outdir) / "index.json").write_text(json.dumps(
        {"versions": versions, "candidates": index}, indent=2))
    n_v3 = sum(1 for c in index if c["level"] == "V3")
    print(f"\nWrote {len(index)} dossiers -> {root}/")
    print(f"V3 (CAS-certified, 3 engines agree): {n_v3}/{len(index)}")
    print(f"index -> {Path(args.outdir) / 'index.json'}")


if __name__ == "__main__":
    main()
