"""
dedup_6f5_v2.py
===============
Isolate genuinely NEW 6F5 v2 survivors that were not present in batch 1.

Key = (tuple(shift), tuple(dir), z_num, z_den) -- the CMF-defining parameters
(gid/r2/lam1 are run-specific and excluded from the key).
"""
from __future__ import annotations

import json
import sys


def key(r):
    return (tuple(r["shift"]), tuple(r["dir"]), int(r["z_num"]), int(r["z_den"]))


def load_keys(path):
    ks = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ks.add(key(json.loads(line)))
    return ks


def main():
    batch1 = sys.argv[1]
    v2 = sys.argv[2]
    out = sys.argv[3]

    print(f"[dedup] loading batch-1 keys from {batch1}")
    seen = load_keys(batch1)
    print(f"[dedup] batch-1 unique keys: {len(seen):,}")

    n_v2 = 0
    n_new = 0
    v2_new_keys = set()
    with open(v2) as f, open(out, "w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_v2 += 1
            r = json.loads(line)
            k = key(r)
            if k in seen or k in v2_new_keys:
                continue
            v2_new_keys.add(k)
            g.write(json.dumps(r) + "\n")
            n_new += 1

    print(f"[dedup] v2 records:            {n_v2:,}")
    print(f"[dedup] NEW unique (not in b1): {n_new:,} -> {out}")
    print(f"[dedup] overlap/dupe dropped:   {n_v2 - n_new:,}")


if __name__ == "__main__":
    main()
