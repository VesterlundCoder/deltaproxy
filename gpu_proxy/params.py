"""
params.py
=========
Deterministic, counter-based parameter generation for the GPU proxy sweep.

The whole point: we screen BILLIONS of trajectories, so we must NOT store or
transfer their parameters. Instead every trajectory is identified by a single
64-bit global index `gid` and a 64-bit `seed`; its parameters (shift, dir, z)
are produced by a splitmix64 stream seeded from (seed, gid). The GPU kernel
contains a bit-identical copy of this generator, so:

    * the kernel generates params on-device from gid (no host->device transfer),
    * the host re-derives the params of the few SURVIVORS from their gids
      (vectorized, below) to hand to the exact-arithmetic verifier.

draw order per trajectory (MUST match proxy_kernel.cu):
    draws[0 .. nsh-1]  -> shift[i] = (v mod (2*box+1)) - box
    draws[nsh]         -> dir index  = v mod ndir
    draws[nsh+1]       -> z   index  = v mod nz
"""
from __future__ import annotations

import numpy as np

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)
U30, U27, U31 = np.uint64(30), np.uint64(27), np.uint64(31)


def _splitmix_next(s):
    """Advance a vector of splitmix64 states in place; return the output vector.
    All ops are uint64 (wrap mod 2^64, matching C unsigned long long)."""
    s += GOLDEN
    z = s.copy()
    z = (z ^ (z >> U30)) * M1
    z = (z ^ (z >> U27)) * M2
    z = z ^ (z >> U31)
    return s, z


def gen_params(seed, gids, nsh, box, ndir, nz):
    """Vectorized re-derivation of (shift, dir_idx, z_idx) for an array of gids.

    Returns:
        shift   : (B, nsh) int64 in [-box, box]
        dir_idx : (B,)     int64 in [0, ndir)
        z_idx   : (B,)     int64 in [0, nz)
    Bit-identical to the on-device generator in proxy_kernel.cu.
    """
    gids = np.asarray(gids, dtype=np.uint64)
    seed = np.uint64(seed)
    # init state s = seed ^ ((gid+1) * GOLDEN)
    s = seed ^ ((gids + np.uint64(1)) * GOLDEN)
    B = gids.shape[0]
    shift = np.empty((B, nsh), dtype=np.int64)
    span = np.uint64(2 * box + 1)
    for i in range(nsh):
        s, v = _splitmix_next(s)
        shift[:, i] = (v % span).astype(np.int64) - box
    s, vd = _splitmix_next(s)
    dir_idx = (vd % np.uint64(ndir)).astype(np.int64)
    s, vz = _splitmix_next(s)
    z_idx = (vz % np.uint64(nz)).astype(np.int64)
    return shift, dir_idx, z_idx


def z_pool_arr(z_max=1.0):
    """Return the shared z pool (NZ, 2) int64 array (num, den)."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import cmf_generic as cg
    return np.array(cg.z_pool(z_max=z_max), dtype=np.int64)


def build_pools(dim, z_max=1.0):
    """Return (dir_pool, z_pool) as int64 arrays matching cmf_generic."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import cmf_generic as cg
    dirs = np.array(cg.dir_pool(dim), dtype=np.int64)            # (NDIR, nsh)
    zs = np.array(cg.z_pool(z_max=z_max), dtype=np.int64)        # (NZ, 2)
    return dirs, zs


if __name__ == "__main__":
    # self-check: regenerate and show a couple of trajectories
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import cmf_generic as cg
    dim = 6
    nsh = cg.nshift_for(dim)
    dirs, zs = build_pools(dim)
    sh, di, zi = gen_params(seed=12345, gids=np.arange(5), nsh=nsh, box=6,
                            ndir=len(dirs), nz=len(zs))
    for k in range(5):
        print(f"gid={k}: shift={sh[k].tolist()} dir={dirs[di[k]].tolist()} "
              f"z={zs[zi[k]].tolist()}")
