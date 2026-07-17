"""
rl_e_hunt_5day.py
=================
5-day RL-guided search for e-related CF limits.

Improvements over the 1-hour version:
  - Island model with 4 sub-populations to maintain diversity
  - Deep verification on any hit > 50 bits (re-run at depth 2000)
  - All verified hits saved to verified_hits.jsonl immediately
  - Periodic checkpoint every 30 min (population state + stats)
  - Stronger diversity: fitness sharing + random immigrants
  - Broader target set including e*p/q, e/q, e^p, log combos
  - Excludes trivial e-n integer family from reward
  - Logs all configs with > 30 bits to candidates.jsonl

Runs for 5 days on 6 cores.
"""
from __future__ import annotations

import json
import os
import sys
import time
import math
import random
import pickle
from multiprocessing import Pool
from collections import defaultdict

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmf_generic as cg

DIM = 6
BOX = 8
NSH = 11
N_DEPTH = 300
DPS = 200  # enough for screening; deep verify auto-scales
RUN_DAYS = 5.0
POP_SIZE = 60
N_ISLANDS = 4
ELITE_FRAC = 0.2
MUTATION_RATE = 0.35
N_WORKERS = 6
BITS_CAP = 900
VERIFY_THRESHOLD = 50
VERIFY_DEPTH = 2000
CHECKPOINT_INTERVAL = 1800
MIGRATION_INTERVAL = 200
MIGRATION_RATE = 0.1

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "cluster_out", "rl_e_hunt_5day")

BASE_SHIFT = [-1, -1, 0, -1, -1, 1, -2, -2, -1, -2, 1]
BASE_DIR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

TRIVIAL_TARGETS = {"e-2", "e-1", "e", "e+1", "e+2", "e+3", "e-3", "e-4", "e-5",
                   "0-e", "1-e", "2-e", "3-e", "4-e",
                   # reducible fractions equivalent to integer shifts
                   "e-2/2", "e-4/2", "e-6/2", "e-6/3", "e-4/4", "e-8/2", "e-8/4",
                   "e-3/3", "e-6/6", "e-9/3", "e-9/9", "e-10/2", "e-10/5",
                   "e-12/2", "e-12/3", "e-12/4", "e-12/6", "e-12/12"}


def build_targets():
    """e-related target values. Kept compact (DPS+20 precision)."""
    mp.mp.dps = DPS + 50
    e = mp.e
    targets = {}

    # e - p/q (fractional, most important) — skip reducible (p%q==0)
    for q in range(2, 13):
        for p in range(1, 3 * q + 1):
            if p % q == 0:
                continue
            targets[f"e-{p}/{q}"] = e - mp.mpf(p) / mp.mpf(q)

    # n - e
    for n in range(0, 6):
        targets[f"{n}-e"] = mp.mpf(n) - e

    # e^(1/n) and e^(-1/n)
    for n in range(1, 8):
        targets[f"e^(1/{n})"] = mp.e ** (mp.mpf(1) / n)
        targets[f"e^(-1/{n})"] = mp.e ** (mp.mpf(-1) / n)

    # e^(p/q) for small p/q — skip reducible
    for q in range(2, 6):
        for p in range(1, 2 * q + 1):
            if p % q == 0:
                continue
            targets[f"e^({p}/{q})"] = mp.e ** (mp.mpf(p) / mp.mpf(q))
            targets[f"e^(-{p}/{q})"] = mp.e ** (mp.mpf(-p) / mp.mpf(q))

    # e^(1/n) - m
    for n in range(2, 6):
        for m in range(0, 3):
            targets[f"e^(1/{n})-{m}"] = mp.e ** (mp.mpf(1) / n) - m

    # Log-related
    for b in [2, 3, 5]:
        targets[f"log({b})"] = mp.log(b)
        targets[f"1/log({b})"] = 1 / mp.log(b)
    targets["log(2)/log(3)"] = mp.log(2) / mp.log(3)

    # e * log, e / log
    for b in [2, 3]:
        targets[f"e*log({b})"] = e * mp.log(b)
        targets[f"e/log({b})"] = e / mp.log(b)

    # e / pi, e * pi
    targets["e/pi"] = e / mp.pi
    targets["e*pi"] = e * mp.pi

    # sqrt(e)
    targets["sqrt(e)"] = mp.sqrt(e)
    targets["1/sqrt(e)"] = 1 / mp.sqrt(e)

    # e * sqrt(k)
    for k in [2, 3]:
        targets[f"e*sqrt({k})"] = e * mp.sqrt(k)
        targets[f"e/sqrt({k})"] = e / mp.sqrt(k)

    # Keep e-2 and e-1 as reference (penalized)
    targets["e-2"] = e - 2
    targets["e-1"] = e - 1

    return {name: mp.nstr(val, DPS + 20) for name, val in targets.items()}


def random_config(rng, near_base=True):
    if near_base and rng.random() < 0.5:
        shift = list(BASE_SHIFT)
        n_pert = rng.randint(0, 5)
        for _ in range(n_pert):
            idx = rng.randint(0, NSH - 1)
            shift[idx] += rng.choice([-2, -1, 1, 2])
    else:
        shift = [rng.randint(-BOX, BOX) for _ in range(NSH)]
    shift = [max(-BOX, min(BOX, s)) for s in shift]

    dirv = [0] * NSH
    n_adv = rng.choice([1, 1, 1, 1, 2])
    positions = rng.sample(range(NSH), n_adv)
    for pos in positions:
        dirv[pos] = 1

    zn, zd = rng.choice([(1, 1), (1, 1), (1, 1), (1, 1), (-1, 1)])
    return (shift, dirv, zn, zd)


def mutate(config, rng):
    shift, dirv, zn, zd = list(config[0]), list(config[1]), config[2], config[3]

    if rng.random() < MUTATION_RATE:
        idx = rng.randint(0, NSH - 1)
        shift[idx] += rng.choice([-1, 1])
        shift[idx] = max(-BOX, min(BOX, shift[idx]))

    if rng.random() < 0.1:
        idx = rng.randint(0, NSH - 1)
        shift[idx] += rng.choice([-3, -2, 2, 3])
        shift[idx] = max(-BOX, min(BOX, shift[idx]))

    if rng.random() < 0.15:
        dirv = [0] * NSH
        n_adv = rng.choice([1, 1, 2])
        positions = rng.sample(range(NSH), n_adv)
        for pos in positions:
            dirv[pos] = 1

    if rng.random() < 0.08:
        zn = -zn

    return (shift, dirv, zn, zd)


def crossover(p1, p2, rng):
    s1, d1, zn1, zd1 = p1
    s2, d2, zn2, zd2 = p2
    shift = [s1[i] if rng.random() < 0.5 else s2[i] for i in range(NSH)]
    shift = [max(-BOX, min(BOX, s)) for s in shift]
    dirv = [d1[i] if rng.random() < 0.5 else d2[i] for i in range(NSH)]
    zn = zn1 if rng.random() < 0.5 else zn2
    zd = zd1 if rng.random() < 0.5 else zd2
    return (shift, dirv, zn, zd)


def config_distance(c1, c2):
    d = sum(1 for a, b in zip(c1[0], c2[0]) if a != b)
    d += sum(1 for a, b in zip(c1[1], c2[1]) if a != b)
    d += (c1[2] != c2[2])
    return d


def evaluate_config(args):
    """Evaluate one config: bits of accuracy against best target,
    with two-depth convergence verification."""
    config, targets_serialized = args
    shift, dirv, zn, zd = config

    try:
        mp.mp.dps = DPS
        targets = {name: mp.mpf(val) for name, val in targets_serialized.items()}

        dim = DIM
        P = [[1 if i == j else 0 for j in range(dim)] for i in range(dim)]
        snapN = None
        P_mid = None
        for n in range(1, 2 * N_DEPTH + 1):
            P = cg.matmul_int(P, cg.build_M_int(n, shift, dirv, zn, zd, dim), dim)
            if n == N_DEPTH:
                snapN = [row[:] for row in P]
            if n == 3 * N_DEPTH // 2:
                P_mid = [row[:] for row in P]

        last = dim - 1
        best_bits = -1e9
        best_target = "none"
        best_L_str = "0"
        best_pair = None
        best_blind_delta = None

        for i in range(dim):
            for j in range(dim):
                if i == j:
                    continue
                q2 = P[j][last]
                p2 = P[i][last]
                if q2 == 0 or p2 == 0:
                    continue

                L = mp.mpf(p2) / mp.mpf(q2)
                log_q2 = cg.log_bigint(q2)
                if log_q2 <= 0:
                    continue

                # Convergence check
                q_mid = P_mid[j][last]
                p_mid = P_mid[i][last]
                if q_mid == 0:
                    continue
                L_mid = mp.mpf(p_mid) / mp.mpf(q_mid)
                conv_diff = abs(L - L_mid)
                if conv_diff == 0:
                    conv_bits = 9999.0
                else:
                    conv_bits = float(-mp.log(conv_diff) / mp.log(2))
                if conv_bits < 20:
                    continue

                # Blind delta
                qn = snapN[j][last]
                pn = snapN[i][last]
                blind_d = None
                if qn != 0 and pn != 0:
                    cross = pn * q2 - p2 * qn
                    if cross != 0:
                        log_err = cg.log_bigint(cross) - cg.log_bigint(qn) - cg.log_bigint(q2)
                        log_qn = cg.log_bigint(qn)
                        if log_qn > 0:
                            blind_d = -(1.0 + log_err / log_qn)

                # Bits against each target
                for tname, T in targets.items():
                    diff = abs(L - T)
                    if diff == 0:
                        bits = 9999.0
                    else:
                        bits = float(-mp.log(diff) / mp.log(2))
                    bits_capped = min(bits, BITS_CAP)
                    if tname in TRIVIAL_TARGETS:
                        bits_capped = min(bits_capped, 0)
                    if bits_capped > best_bits:
                        best_bits = bits_capped
                        best_target = tname
                        best_L_str = mp.nstr(L, 50)
                        best_pair = (i, j)
                        best_blind_delta = blind_d

        return {
            "shift": shift,
            "dir": dirv,
            "z_num": zn,
            "z_den": zd,
            "bits": best_bits,
            "target": best_target,
            "L": best_L_str,
            "pair": best_pair,
            "blind_delta": best_blind_delta,
        }
    except Exception:
        return {
            "shift": shift,
            "dir": dirv,
            "z_num": zn,
            "z_den": zd,
            "bits": -1e9,
            "target": "error",
            "L": "0",
            "pair": None,
            "blind_delta": None,
        }


def _recompute_target(target_name, dps):
    """Recompute a target value at full precision from its name."""
    mp.mp.dps = dps
    e = mp.e
    name = target_name

    # e - p/q
    if name.startswith("e-") and "/" in name:
        pq = name[2:]
        p, q = pq.split("/")
        return e - mp.mpf(int(p)) / mp.mpf(int(q))
    # n - e
    if name.endswith("-e") and name[0].isdigit():
        n = int(name.split("-")[0])
        return mp.mpf(n) - e
    # e^(p/q) or e^(-p/q)
    if name.startswith("e^(") or name.startswith("e^(-"):
        is_neg = name.startswith("e^(-")
        inner = name[3:-1] if not is_neg else name[4:-1]
        p, q = inner.split("/")
        exp = mp.mpf(int(p)) / mp.mpf(int(q))
        if is_neg:
            exp = -exp
        return mp.e ** exp
    # e^(1/n) or e^(-1/n)
    if name.startswith("e^(1/") or name.startswith("e^(-1/"):
        is_neg = name.startswith("e^(-1/")
        n = int(name.split("/")[1].rstrip(")"))
        exp = mp.mpf(1) / n
        if is_neg:
            exp = -exp
        return mp.e ** exp
    # e^(1/n)-m
    if name.startswith("e^(1/") and "-" in name.split(")")[1]:
        n = int(name.split("/")[1].split(")")[0])
        m = int(name.split(")")[1].lstrip("-"))
        return mp.e ** (mp.mpf(1) / n) - m
    # log(b)
    if name.startswith("log(") and name.endswith(")"):
        b = int(name[4:-1])
        return mp.log(b)
    # 1/log(b)
    if name.startswith("1/log(") and name.endswith(")"):
        b = int(name[6:-1])
        return 1 / mp.log(b)
    # log(2)/log(3)
    if name == "log(2)/log(3)":
        return mp.log(2) / mp.log(3)
    # e*log(b)
    if name.startswith("e*log(") and name.endswith(")"):
        b = int(name[6:-1])
        return e * mp.log(b)
    # e/log(b)
    if name.startswith("e/log(") and name.endswith(")"):
        b = int(name[6:-1])
        return e / mp.log(b)
    # e/pi, e*pi
    if name == "e/pi":
        return e / mp.pi
    if name == "e*pi":
        return e * mp.pi
    # sqrt(e), 1/sqrt(e)
    if name == "sqrt(e)":
        return mp.sqrt(e)
    if name == "1/sqrt(e)":
        return 1 / mp.sqrt(e)
    # e*sqrt(k), e/sqrt(k)
    if name.startswith("e*sqrt(") and name.endswith(")"):
        k = int(name[7:-1])
        return e * mp.sqrt(k)
    if name.startswith("e/sqrt(") and name.endswith(")"):
        k = int(name[7:-1])
        return e / mp.sqrt(k)
    # e-2, e-1 (trivial)
    if name == "e-2":
        return e - 2
    if name == "e-1":
        return e - 1

    raise ValueError(f"Unknown target: {name}")


def deep_verify(config, target_name, target_val_str, screen_pair):
    """Deep verification: check ALL pairs at multiple depths with auto-scaled precision.
    Recomputes target at full precision from name to detect genuine convergence."""
    shift, dirv, zn, zd = config
    dim = DIM
    last = dim - 1

    # Compute target at very high precision (recompute from name, not low-precision string)
    T_high = _recompute_target(target_name, 20000)

    # Find best pair at screening depth
    best_pair = None
    best_bits_screen = -1
    mp.mp.dps = DPS + 50
    P_screen = [[1 if i2 == j2 else 0 for j2 in range(dim)] for i2 in range(dim)]
    for n in range(1, 2 * N_DEPTH + 1):
        P_screen = cg.matmul_int(P_screen, cg.build_M_int(n, shift, dirv, zn, zd, dim), dim)

    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            q2 = P_screen[j][last]
            p2 = P_screen[i][last]
            if q2 == 0 or p2 == 0:
                continue
            L = mp.mpf(p2) / mp.mpf(q2)
            diff = abs(L - T_high)
            if diff == 0:
                bits = 9999.0
            else:
                bits = float(-mp.log(diff) / mp.log(2))
            if bits > best_bits_screen:
                best_bits_screen = bits
                best_pair = (i, j)

    if best_pair is None:
        return {"config": list(config), "target": target_name, "pair": None,
                "depth_results": [], "is_genuine": False, "pslq": None}

    i, j = best_pair
    results = []
    for N in [200, 500, 1000, VERIFY_DEPTH]:
        P = [[1 if i2 == j2 else 0 for j2 in range(dim)] for i2 in range(dim)]
        for n in range(1, 2 * N + 1):
            P = cg.matmul_int(P, cg.build_M_int(n, shift, dirv, zn, zd, dim), dim)

        q2 = P[j][last]
        p2 = P[i][last]
        if q2 == 0:
            results.append({"N": N, "bits": -1, "L": "0", "diff": "inf"})
            continue

        int_digits = max(len(str(abs(p2))), len(str(abs(q2))))
        # Use precision just above integer digits so diff is meaningful
        mp.mp.dps = int_digits + 50
        L = mp.mpf(p2) / mp.mpf(q2)
        # T_high is at 20000 dps — diff reflects L's actual accuracy
        diff = abs(L - T_high)
        if diff == 0:
            bits = 9999.0
        else:
            bits = float(-mp.log(diff) / mp.log(2))

        results.append({
            "N": N,
            "bits": bits,
            "L": mp.nstr(L, 40),
            "diff": mp.nstr(diff, 5),
            "int_digits": int_digits,
            "pair": [i, j],
        })

    # Check if bits scale linearly with N (genuine convergence)
    valid = [r for r in results if r["bits"] > 0 and r["bits"] < 9999]
    if len(valid) >= 2:
        ratio = valid[-1]["bits"] / valid[0]["bits"]
        depth_ratio = valid[-1]["N"] / valid[0]["N"]
        is_genuine = ratio > 0.3 * depth_ratio
    elif len(results) >= 2 and all(r["bits"] >= 9999 for r in results):
        # All exact matches at all depths — definitely genuine
        is_genuine = True
    else:
        is_genuine = False

    # PSLQ if genuine
    pslq_result = None
    if is_genuine and valid:
        mp.mp.dps = valid[-1]["int_digits"] + 100
        P = [[1 if i2 == j2 else 0 for j2 in range(dim)] for i2 in range(dim)]
        for n in range(1, 2 * VERIFY_DEPTH + 1):
            P = cg.matmul_int(P, cg.build_M_int(n, shift, dirv, zn, zd, dim), dim)
        L = mp.mpf(P[i][last]) / mp.mpf(P[j][last])

        try:
            pslq_result = mp.pslq([L, mp.mpf(1), mp.e], maxcoeff=10000)
        except:
            pslq_result = None

    return {
        "config": list(config),
        "target": target_name,
        "pair": [i, j],
        "depth_results": results,
        "is_genuine": is_genuine,
        "pslq": pslq_result,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    deadline = t0 + RUN_DAYS * 86400

    targets_serialized = build_targets()
    n_targets = len(targets_serialized)
    print(f"[rl5d] {n_targets} targets, depth={N_DEPTH}, dps={DPS}, "
          f"pop={POP_SIZE}x{N_ISLANDS} islands, workers={N_WORKERS}, "
          f"run={RUN_DAYS} days", flush=True)
    print(f"[rl5d] Output: {OUT}", flush=True)

    # Initialize islands
    islands = []
    for isl in range(N_ISLANDS):
        rng = random.Random(42 + isl * 1000)
        pop = []
        if isl == 0:
            pop.append((list(BASE_SHIFT), list(BASE_DIR), 1, 1))
            pop.append((list(BASE_SHIFT), [0,0,0,0,0,1,0,0,0,0,1], 1, 1))
        elif isl == 1:
            pop.append(([-1,-3,-1,-1,-1,1,-2,-2,-1,-3,-1], [0,0,0,0,0,0,0,0,1,0,0], 1, 1))
        elif isl == 2:
            pop.append(([-2,-1,-1,-1,-1,-1,-2,-2,-1,-2,-1], [0,0,0,0,1,0,1,0,0,0,0], 1, 1))
        else:
            pop.append((list(BASE_SHIFT), list(BASE_DIR), -1, 1))

        while len(pop) < POP_SIZE:
            pop.append(random_config(rng, near_base=(isl < 3)))
        islands.append(pop)

    generation = 0
    all_time_best_bits = -1e9
    verified_hits = []
    candidates_log = open(os.path.join(OUT, "candidates.jsonl"), "a")
    verified_log = open(os.path.join(OUT, "verified_hits.jsonl"), "a")
    seen_configs = set()
    last_checkpoint = t0

    # Load existing hits if resuming
    hits_file = os.path.join(OUT, "verified_hits.jsonl")
    if os.path.exists(hits_file) and os.path.getsize(hits_file) > 0:
        for line in open(hits_file):
            try:
                h = json.loads(line)
                verified_hits.append(h)
                key = (tuple(h["config"][0]), tuple(h["config"][1]), h["config"][2], h["config"][3])
                seen_configs.add(key)
                if h.get("screen_bits", 0) > all_time_best_bits:
                    all_time_best_bits = h["screen_bits"]
            except:
                pass
        print(f"[rl5d] Loaded {len(verified_hits)} existing verified hits", flush=True)

    print(f"[rl5d] Starting evolution...", flush=True)

    # Persistent pool (like the 1-hour script that worked)
    pool = Pool(N_WORKERS)

    while time.time() < deadline:
        gen_start = time.time()
        time_left = deadline - time.time()
        if time_left < 30:
            break

        # Evaluate all islands
        all_tasks = []
        island_ranges = []
        for isl_idx, pop in enumerate(islands):
            start = len(all_tasks)
            for config in pop:
                all_tasks.append((config, targets_serialized))
            island_ranges.append((start, len(all_tasks)))

        all_results = pool.map(evaluate_config, all_tasks, chunksize=4)

        # Split results back to islands
        island_results = []
        for start, end in island_ranges:
            island_results.append(all_results[start:end])

        # Process each island
        global_best_this_gen = None
        global_best_bits = -1e9
        pending_verifications = []

        for isl_idx, results in enumerate(island_results):
            results.sort(key=lambda r: r["bits"], reverse=True)

            elite_n = max(2, int(POP_SIZE * ELITE_FRAC))
            elite_configs = [(r["shift"], r["dir"], r["z_num"], r["z_den"])
                             for r in results[:elite_n]]

            for r in results:
                r_config = (r["shift"], r["dir"], r["z_num"], r["z_den"])
                min_dist = min(config_distance(r_config, e) for e in elite_configs)
                r["diversity_adj"] = r["bits"] + min_dist * 5.0
                if r["bits"] >= BITS_CAP:
                    r["diversity_adj"] = BITS_CAP + min_dist * 10.0

            results.sort(key=lambda r: r["diversity_adj"], reverse=True)

            raw_best = max(results, key=lambda r: r["bits"])
            if raw_best["bits"] > global_best_bits:
                global_best_bits = raw_best["bits"]
                global_best_this_gen = raw_best

            # Log candidates > 30 bits
            for r in results:
                if r["bits"] > 30 and r["target"] not in TRIVIAL_TARGETS:
                    key = (tuple(r["shift"]), tuple(r["dir"]), r["z_num"], r["z_den"])
                    if key not in seen_configs:
                        r_entry = {
                            "config": [r["shift"], r["dir"], r["z_num"], r["z_den"]],
                            "bits": r["bits"],
                            "target": r["target"],
                            "L": r["L"],
                            "pair": r["pair"],
                            "blind_delta": r.get("blind_delta"),
                            "generation": generation,
                            "island": isl_idx,
                        }
                        candidates_log.write(json.dumps(r_entry) + "\n")
                        candidates_log.flush()

            # Collect candidates for deep verification
            for r in results:
                if r["bits"] > VERIFY_THRESHOLD and r["target"] not in TRIVIAL_TARGETS:
                    key = (tuple(r["shift"]), tuple(r["dir"]), r["z_num"], r["z_den"])
                    if key not in seen_configs:
                        seen_configs.add(key)
                        pending_verifications.append((r, isl_idx))

            # Selection
            elite_configs = [(r["shift"], r["dir"], r["z_num"], r["z_den"])
                             for r in results[:elite_n]]
            new_pop = list(elite_configs)

            rng = random.Random(42 + isl_idx * 1000 + generation)
            while len(new_pop) < POP_SIZE:
                r = rng.random()
                if r < 0.15 and len(elite_configs) >= 2:
                    p1 = rng.choice(elite_configs)
                    p2 = rng.choice(elite_configs)
                    child = crossover(p1, p2, rng)
                    child = mutate(child, rng)
                elif r < 0.30:
                    child = random_config(rng, near_base=(isl_idx < 3))
                else:
                    parent = rng.choice(elite_configs)
                    child = mutate(parent, rng)
                new_pop.append(child)

            islands[isl_idx] = new_pop

        # Migration between islands
        if generation > 0 and generation % MIGRATION_INTERVAL == 0:
            n_migrate = max(1, int(POP_SIZE * MIGRATION_RATE))
            migrants = []
            for isl_idx in range(N_ISLANDS):
                migrants.append(islands[isl_idx][:n_migrate])
            for isl_idx in range(N_ISLANDS):
                target_isl = (isl_idx + 1) % N_ISLANDS
                islands[target_isl] = islands[target_isl][:-n_migrate] + migrants[isl_idx]
            print(f"[rl5d] GEN {generation} migration: {n_migrate} per island", flush=True)

        # Print gen summary
        elapsed = time.time() - t0
        unique_targets = set()
        for results in island_results:
            for r in results[:10]:
                if r["bits"] > 5:
                    unique_targets.add(r["target"])

        if global_best_this_gen and global_best_this_gen["bits"] > all_time_best_bits:
            all_time_best_bits = global_best_this_gen["bits"]
            print(f"[rl5d] GEN {generation:4d} NEW BEST: {global_best_this_gen['bits']:.1f} bits "
                  f"target={global_best_this_gen['target']} "
                  f"shift={global_best_this_gen['shift']} dir={global_best_this_gen['dir']}", flush=True)

        if generation % 20 == 0:
            print(f"[rl5d] GEN {generation:4d}  top: {global_best_bits:.1f} bits "
                  f"targets: {len(unique_targets)}  "
                  f"elapsed: {elapsed:.0f}s ({elapsed/3600:.1f}h)  "
                  f"left: {(deadline-time.time())/3600:.1f}h  "
                  f"verified: {len(verified_hits)}", flush=True)

        # Deferred deep verification (pool workers are idle, main process is free)
        for r, isl_idx in pending_verifications:
            print(f"[rl5d] GEN {generation} ISL {isl_idx} VERIFYING: "
                  f"{r['bits']:.1f} bits target={r['target']} "
                  f"shift={r['shift']} dir={r['dir']}", flush=True)

            config = (r["shift"], r["dir"], r["z_num"], r["z_den"])
            verify = deep_verify(config, r["target"],
                                 targets_serialized[r["target"]],
                                 r["pair"])

            if verify["is_genuine"]:
                hit_entry = {
                    "config": [list(config[0]), list(config[1]), config[2], config[3]],
                    "target": r["target"],
                    "pair": verify["pair"],
                    "screen_bits": r["bits"],
                    "depth_results": verify["depth_results"],
                    "pslq": verify["pslq"],
                    "blind_delta": r.get("blind_delta"),
                    "generation": generation,
                    "island": isl_idx,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                verified_log.write(json.dumps(hit_entry) + "\n")
                verified_log.flush()
                verified_hits.append(hit_entry)

                if r["bits"] > all_time_best_bits:
                    all_time_best_bits = r["bits"]

                bits_final = verify["depth_results"][-1]["bits"] if verify["depth_results"] else 0
                print(f"[rl5d] *** VERIFIED HIT *** target={r['target']} "
                      f"pslq={verify['pslq']} "
                      f"bits@N2000={bits_final:.0f}", flush=True)
            else:
                print(f"[rl5d]   -> false positive (bits don't scale)", flush=True)

        # Checkpoint
        if time.time() - last_checkpoint > CHECKPOINT_INTERVAL:
            ckpt = {
                "generation": generation,
                "elapsed": time.time() - t0,
                "islands": islands,
                "all_time_best_bits": all_time_best_bits,
                "verified_hits_count": len(verified_hits),
                "seen_configs_count": len(seen_configs),
            }
            ckpt_file = os.path.join(OUT, "checkpoint.pkl")
            with open(ckpt_file, "wb") as f:
                pickle.dump(ckpt, f)
            last_checkpoint = time.time()
            print(f"[rl5d] CHECKPOINT gen={generation} verified={len(verified_hits)} "
                  f"seen={len(seen_configs)} ({(deadline-time.time())/3600:.1f}h left)", flush=True)

        generation += 1

    pool.close()
    pool.join()
    candidates_log.close()
    verified_log.close()

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"[rl5d] DONE: {generation} generations in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  Verified hits: {len(verified_hits)}")
    print(f"  All-time best (screen): {all_time_best_bits:.1f} bits")
    print(f"\n  Verified hits summary:")
    for h in verified_hits:
        pslq = h.get("pslq")
        bits_final = h["depth_results"][-1]["bits"] if h["depth_results"] else 0
        print(f"    {h['target']:15s}  bits={bits_final:.0f}  pslq={pslq}  "
              f"shift={h['config'][0]}  dir={h['config'][1]}")
    print(f"\n  Results -> {OUT}/")


if __name__ == "__main__":
    main()
