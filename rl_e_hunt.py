"""
rl_e_hunt.py
============
Reinforcement-guided search for e-related CF limits.

Uses an evolutionary strategy (population-based REINFORCE variant):
  - State: (shift, dir, z) CMF configuration
  - Actions: perturb one position by ±1, or change advancing root
  - Reward: bits of accuracy = -log2(|L - target|) at depth N
  - Policy: population of configs, mutate best performers, elitism

Targets: e-2, e-1, e-3, e-1/2, e-3/2, e-1/4, e-3/4, e^(-1), e^(-1/2), etc.
Base seed: the e-2 hit (shift=[-1,-1,0,-1,-1,1,-2,-2,-1,-2,1], dir=[...,1])

Runs for 1 hour on 6 cores.
"""
from __future__ import annotations

import json
import os
import sys
import time
import math
import random
from multiprocessing import Pool
from collections import defaultdict

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmf_generic as cg

DIM = 6
BOX = 6
NSH = 11  # 2*dim - 1
N_DEPTH = 300  # CF depth (2N = 600 matrix mults) + mid-snapshot at 450
DPS = 1000  # high precision to distinguish exact matches
RUN_HOURS = 1.0
POP_SIZE = 80
ELITE_FRAC = 0.2
MUTATION_RATE = 0.35
N_WORKERS = 6
BITS_CAP = 900  # cap bits at precision limit
DIVERSITY_BONUS = 50  # bits bonus for novel configs
TRIVIAL_TARGETS = {"e-2", "e-1", "e", "e+1", "e+2", "e+3", "e-3", "e-4", "e-5",
                   "0-e", "1-e", "2-e", "3-e", "4-e"}  # integer-shift family

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "cluster_out", "rl_e_hunt")

BASE_SHIFT = [-1, -1, 0, -1, -1, 1, -2, -2, -1, -2, 1]
BASE_DIR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def build_targets():
    """e-related target values at sufficient precision.
    Excludes trivial e-n (integer shifts) to force the search to find
    fractional/radical e-relations instead."""
    mp.mp.dps = DPS + 50
    e = mp.e
    targets = {}
    # Fractional e - p/q
    for p, q in [(1,2), (3,2), (1,4), (3,4), (5,2), (7,3), (8,3), (9,4), (11,4),
                 (1,3), (2,3), (4,3), (5,3), (1,5), (2,5), (3,5), (7,5), (9,5),
                 (1,6), (5,6), (7,6), (11,6)]:
        targets[f"e-{p}/{q}"] = e - mp.mpf(p) / mp.mpf(q)
    # n - e (negative side)
    for n in range(0, 5):
        targets[f"{n}-e"] = mp.mpf(n) - e
    # e^(1/n) and e^(-1/n)
    for n in [1, 2, 3, 4, 5, 6]:
        targets[f"e^(1/{n})"] = mp.e ** (mp.mpf(1) / n)
        targets[f"e^(-1/{n})"] = mp.e ** (mp.mpf(-1) / n)
    # e^(p/q)
    for p, q in [(1,2), (3,2), (1,4), (3,4), (5,3), (7,4), (5,4), (7,3)]:
        targets[f"e^({p}/{q})"] = mp.e ** (mp.mpf(p) / mp.mpf(q))
        targets[f"e^(-{p}/{q})"] = mp.e ** (mp.mpf(-p) / mp.mpf(q))
    # e^(1/n) - 1
    for n in [2, 3, 4, 5, 6]:
        targets[f"e^(1/{n})-1"] = mp.e ** (mp.mpf(1) / n) - 1
    # e^(1/n) - m
    for n in [2, 3, 4]:
        for m in [1, 2]:
            targets[f"e^(1/{n})-{m}"] = mp.e ** (mp.mpf(1) / n) - m
    # Log-related
    targets["1/log(2)"] = 1 / mp.log(2)
    targets["log(2)"] = mp.log(2)
    targets["log(3)"] = mp.log(3)
    targets["1/log(3)"] = 1 / mp.log(3)
    targets["log(2)/log(3)"] = mp.log(2) / mp.log(3)
    # e * log(2), e / log(2)
    targets["e*log(2)"] = e * mp.log(2)
    targets["e/log(2)"] = e / mp.log(2)
    # e / pi, e * pi
    targets["e/pi"] = e / mp.pi
    targets["e*pi"] = e * mp.pi
    # sqrt(e)
    targets["sqrt(e)"] = mp.sqrt(e)
    targets["1/sqrt(e)"] = 1 / mp.sqrt(e)
    # Keep e-2 and e-1 as reference but they won't be the main targets
    targets["e-2"] = e - 2
    targets["e-1"] = e - 1
    # Serialize as strings for multiprocessing
    return {name: mp.nstr(val, DPS + 20) for name, val in targets.items()}


def random_config(rng):
    """Generate a random CMF config near the base hit."""
    shift = list(BASE_SHIFT)
    n_pert = rng.randint(0, 4)
    for _ in range(n_pert):
        idx = rng.randint(0, NSH - 1)
        shift[idx] += rng.choice([-2, -1, 1, 2])
    shift = [max(-BOX, min(BOX, s)) for s in shift]

    # Random dir: pick 1-2 advancing positions
    dirv = [0] * NSH
    n_adv = rng.choice([1, 1, 1, 2])
    positions = rng.sample(range(NSH), n_adv)
    for pos in positions:
        dirv[pos] = 1

    zn, zd = rng.choice([(1, 1), (1, 1), (1, 1), (-1, 1)])
    return (shift, dirv, zn, zd)


def mutate(config, rng):
    """Mutate a config: perturb shift, change dir, or flip z."""
    shift, dirv, zn, zd = list(config[0]), list(config[1]), config[2], config[3]

    # Shift mutations
    if rng.random() < MUTATION_RATE:
        idx = rng.randint(0, NSH - 1)
        shift[idx] += rng.choice([-1, 1])
        shift[idx] = max(-BOX, min(BOX, shift[idx]))

    # Occasionally change dir
    if rng.random() < 0.15:
        # Reset and pick new advancing positions
        dirv = [0] * NSH
        n_adv = rng.choice([1, 1, 2])
        positions = rng.sample(range(NSH), n_adv)
        for pos in positions:
            dirv[pos] = 1

    # Occasionally flip z
    if rng.random() < 0.1:
        zn = -zn

    return (shift, dirv, zn, zd)


def crossover(parent1, parent2, rng):
    """Uniform crossover of two configs."""
    s1, d1, zn1, zd1 = parent1
    s2, d2, zn2, zd2 = parent2
    shift = [s1[i] if rng.random() < 0.5 else s2[i] for i in range(NSH)]
    shift = [max(-BOX, min(BOX, s)) for s in shift]
    dirv = [d1[i] if rng.random() < 0.5 else d2[i] for i in range(NSH)]
    zn = zn1 if rng.random() < 0.5 else zn2
    zd = zd1 if rng.random() < 0.5 else zd2
    return (shift, dirv, zn, zd)


def evaluate_config(args):
    """Evaluate one config: compute CF limit at depth N, return bits of accuracy
    against best-matching target. Uses two-depth verification to filter
    coincidental matches."""
    config, targets_serialized = args
    shift, dirv, zn, zd = config

    try:
        mp.mp.dps = DPS
        targets = {name: mp.mpf(val) for name, val in targets_serialized.items()}

        # Exact integer double-depth walk at N_DEPTH
        dim = DIM
        P = [[1 if i == j else 0 for j in range(dim)] for i in range(dim)]
        snapN = None
        for n in range(1, 2 * N_DEPTH + 1):
            P = cg.matmul_int(P, cg.build_M_int(n, shift, dirv, zn, zd, dim), dim)
            if n == N_DEPTH:
                snapN = [row[:] for row in P]
            # Snapshot at 3/4 depth for convergence check
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

                # Convergence check: compare L at depth 2N vs 1.5N
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
                # Require at least 20 bits of convergence (L is stable)
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

                # Bits of accuracy against each target
                for tname, T in targets.items():
                    diff = abs(L - T)
                    if diff == 0:
                        bits = 9999.0
                    else:
                        bits = float(-mp.log(diff) / mp.log(2))
                    bits_capped = min(bits, BITS_CAP)
                    # Penalize trivial e-n targets
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


def config_distance(c1, c2):
    """Hamming-like distance between two configs."""
    d = sum(1 for a, b in zip(c1[0], c2[0]) if a != b)
    d += sum(1 for a, b in zip(c1[1], c2[1]) if a != b)
    d += (c1[2] != c2[2])
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    deadline = t0 + RUN_HOURS * 3600

    targets_serialized = build_targets()
    print(f"[rl] {len(targets_serialized)} targets, depth={N_DEPTH}, dps={DPS}, "
          f"pop={POP_SIZE}, workers={N_WORKERS}, run={RUN_HOURS}h", flush=True)

    # Initialize population: base hit + random configs
    rng = random.Random(42)
    population = []
    # Always include the base e-2 hit
    population.append((list(BASE_SHIFT), list(BASE_DIR), 1, 1))
    # And the golden ratio variant
    population.append((list(BASE_SHIFT), [0,0,0,0,0,1,0,0,0,0,1], 1, 1))
    # e-1 variant
    population.append(([-1,-1,0,-1,-1,2,-2,-2,-1,-2,1], list(BASE_DIR), 1, 1))
    # Fill rest with random + mutated seeds
    while len(population) < POP_SIZE:
        if rng.random() < 0.3:
            population.append(mutate(population[0], rng))
        else:
            population.append(random_config(rng))

    pool = Pool(N_WORKERS)

    generation = 0
    all_time_best = None
    all_time_best_bits = -1e9
    hits_log = open(os.path.join(OUT, "hits.jsonl"), "w")

    while time.time() < deadline:
        gen_start = time.time()
        gen_time_left = deadline - time.time()
        if gen_time_left < 10:
            break

        # Evaluate population
        tasks = [(config, targets_serialized) for config in population]
        results = pool.map(evaluate_config, tasks, chunksize=3)

        # Sort by bits (descending)
        results.sort(key=lambda r: r["bits"], reverse=True)

        # Apply diversity bonus: configs close to existing elites get penalized
        elite_n = max(2, int(POP_SIZE * ELITE_FRAC))
        elite_configs_raw = [(r["shift"], r["dir"], r["z_num"], r["z_den"]) 
                             for r in results[:elite_n]]
        
        for r in results:
            r_config = (r["shift"], r["dir"], r["z_num"], r["z_den"])
            min_dist = min(config_distance(r_config, e) for e in elite_configs_raw)
            r["diversity_adj"] = r["bits"] + min_dist * DIVERSITY_BONUS / 10.0
            # Cap: if bits >= BITS_CAP (exact match), diversity is the tiebreaker
            if r["bits"] >= BITS_CAP:
                r["diversity_adj"] = BITS_CAP + min_dist * 10.0

        # Re-sort by diversity-adjusted score
        results.sort(key=lambda r: r["diversity_adj"], reverse=True)

        # Track all-time best by raw bits
        raw_best = max(results, key=lambda r: r["bits"])
        if raw_best["bits"] > all_time_best_bits:
            all_time_best_bits = raw_best["bits"]
            all_time_best = raw_best
            hits_log.write(json.dumps(raw_best) + "\n")
            hits_log.flush()
            print(f"[rl] GEN {generation} NEW BEST: {raw_best['bits']:.1f} bits "
                  f"target={raw_best['target']} L={raw_best['L'][:25]}... "
                  f"shift={raw_best['shift']} dir={raw_best['dir']}", flush=True)

        # Print gen summary
        top5 = results[:5]
        gen_time = time.time() - gen_start
        elapsed = time.time() - t0
        unique_targets = set(r["target"] for r in results[:20])
        print(f"[rl] GEN {generation:3d}  top: {top5[0]['bits']:.1f} bits "
              f"({top5[0]['target']})  avg_top5: {sum(r['bits'] for r in top5)/5:.1f}  "
              f"targets_in_top20: {len(unique_targets)}  "
              f"elapsed: {elapsed:.0f}s", flush=True)

        # Show top 3 details every 10 gens
        if generation % 10 == 0:
            for i, r in enumerate(top5[:3]):
                print(f"  #{i+1}  bits={r['bits']:.1f}  target={r['target']:12s}  "
                      f"blind_δ={r.get('blind_delta', 'N/A')}  "
                      f"shift={r['shift']}  dir={r['dir']}", flush=True)

        # Selection: keep diverse elites + mutate + crossover + random injection
        elite_configs = [(r["shift"], r["dir"], r["z_num"], r["z_den"]) 
                         for r in results[:elite_n]]
        new_pop = list(elite_configs)

        # Fill rest with mutations, crossovers, and random injections
        while len(new_pop) < POP_SIZE:
            r = rng.random()
            if r < 0.2 and len(elite_configs) >= 2:
                # Crossover
                p1 = rng.choice(elite_configs)
                p2 = rng.choice(elite_configs)
                child = crossover(p1, p2, rng)
                child = mutate(child, rng)
            elif r < 0.35:
                # Random injection for diversity
                child = random_config(rng)
            else:
                # Mutate elite
                parent = rng.choice(elite_configs)
                child = mutate(parent, rng)
            new_pop.append(child)

        population = new_pop
        generation += 1

    pool.close()
    pool.join()
    hits_log.close()

    elapsed = time.time() - t0

    # Final summary
    print(f"\n{'='*70}")
    print(f"[rl] DONE: {generation} generations in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  All-time best: {all_time_best_bits:.1f} bits")
    if all_time_best:
        print(f"  target:  {all_time_best['target']}")
        print(f"  L:       {all_time_best['L']}")
        print(f"  shift:   {all_time_best['shift']}")
        print(f"  dir:     {all_time_best['dir']}")
        print(f"  z:       {all_time_best['z_num']}/{all_time_best['z_den']}")
        print(f"  blind_δ: {all_time_best.get('blind_delta', 'N/A')}")

    # Show all hits
    print(f"\n  All hits (new bests):")
    hits_file = os.path.join(OUT, "hits.jsonl")
    if os.path.exists(hits_file):
        for line in open(hits_file):
            h = json.loads(line)
            print(f"    {h['bits']:.1f} bits  {h['target']:12s}  "
                  f"L={h['L'][:25]}...  shift={h['shift']}")

    print(f"\n  hits -> {OUT}/hits.jsonl")


if __name__ == "__main__":
    main()
