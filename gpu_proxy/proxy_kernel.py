"""
proxy_kernel.py
===============
Stage-1 GPU spectral-proxy kernel: one thread per trajectory.

Each thread
  1. generates its parameters (shift, dir, z) on-device from (seed, gid) with a
     splitmix64 stream IDENTICAL to params.gen_params (no host transfer),
  2. evolves the d x d companion cocycle for N steps in registers,
  3. re-orthonormalizes every step with modified Gram-Schmidt (MGS) QR,
     accumulating sum log|r_jj|,
  4. writes the detector  r_2 = -lambda_2/lambda_1  (and lambda_1) to global mem.

No big integers, no RNS -> fixed-width float, embarrassingly parallel. Portable
to AMD MI250X (LUMI) and NVIDIA via cupy.RawModule (HIP / CUDA).

A numpy fallback (run_numpy) reproduces the SAME result via proxy_batch so the
full pipeline is testable on a laptop with no GPU, then flipped to GPU on LUMI.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cmf_generic as cg                       # noqa: E402
from params import gen_params, build_pools     # noqa: E402

_CUDA_SRC = r"""
extern "C" {{

__device__ __forceinline__ unsigned long long sm_next(unsigned long long* s) {{
    *s += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = *s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}}

__device__ __forceinline__ void esym(const {REAL}* vals, int m, {REAL}* e) {{
    for (int k = 0; k <= m; ++k) e[k] = 0;
    e[0] = 1;
    for (int j = 0; j < m; ++j)
        for (int k = m; k >= 1; --k)
            e[k] += vals[j] * e[k - 1];
}}

__global__ void proxy(const long long* __restrict__ dir_pool,   // NDIR * NSH
                      const long long* __restrict__ z_pool,     // NZ   * 2
                      unsigned long long seed,
                      unsigned long long gid0,                   // batch offset
                      unsigned long long n_traj,                 // this batch
                      {REAL}* __restrict__ out_r2,
                      {REAL}* __restrict__ out_lam1) {{
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_traj) return;
    unsigned long long gid = gid0 + tid;

    // ---- on-device parameter generation (matches params.gen_params) ----
    unsigned long long s = seed ^ ((gid + 1ULL) * 0x9E3779B97F4A7C15ULL);
    long long shift[{NSH}];
    #pragma unroll
    for (int i = 0; i < {NSH}; ++i) {{
        unsigned long long v = sm_next(&s);
        shift[i] = (long long)(v % {SPAN}ULL) - {BOX}LL;
    }}
    unsigned long long vd = sm_next(&s);
    int di = (int)(vd % {NDIR}ULL);
    unsigned long long vz = sm_next(&s);
    int zi = (int)(vz % {NZ}ULL);
    const long long* dir = dir_pool + (long long)di * {NSH};
    long long zn = z_pool[2 * zi];
    long long zd = z_pool[2 * zi + 1];

    // ---- cocycle Lyapunov via per-step MGS-QR ----
    {REAL} Q[{DIM}][{DIM}];
    #pragma unroll
    for (int a = 0; a < {DIM}; ++a)
        #pragma unroll
        for (int b = 0; b < {DIM}; ++b)
            Q[a][b] = (a == b) ? 1 : 0;
    {REAL} logs[{DIM}];
    #pragma unroll
    for (int a = 0; a < {DIM}; ++a) logs[a] = 0;

    for (int n = 1; n <= {NSTEPS}; ++n) {{
        {REAL} f[{DIM}], g[{DIM}], ef[{DIM} + 1], eg[{DIM}], c[{DIM}];
        #pragma unroll
        for (int i = 0; i < {DIM}; ++i)
            f[i] = (({REAL})shift[i]) + ({REAL})n * dir[i] + 1;
        #pragma unroll
        for (int j = 0; j < {DIM} - 1; ++j)
            g[j] = (({REAL})shift[{DIM} + j]) + ({REAL})n * dir[{DIM} + j] + 2;
        esym(f, {DIM}, ef);            // ef[0..DIM]
        esym(g, {DIM} - 1, eg);        // eg[0..DIM-1]
        {REAL} hf = ({REAL})(-zn), hg = ({REAL})zd;
        c[0] = hf * ef[{DIM}];
        #pragma unroll
        for (int k = 1; k < {DIM}; ++k)
            c[k] = hg * eg[{DIM} - k] + hf * ef[{DIM} - k];

        // V = M @ Q,  M: subdiag 1, last column = c
        //   V[r][j] = c[r]*Q[DIM-1][j] + (r>=1 ? Q[r-1][j] : 0)
        {REAL} V[{DIM}][{DIM}];
        #pragma unroll
        for (int r = 0; r < {DIM}; ++r)
            #pragma unroll
            for (int j = 0; j < {DIM}; ++j)
                V[r][j] = c[r] * Q[{DIM} - 1][j] + (r >= 1 ? Q[r - 1][j] : ({REAL})0);

        // MGS over columns of V -> new Q, diag(R) magnitudes
        #pragma unroll
        for (int j = 0; j < {DIM}; ++j) {{
            {REAL} v[{DIM}];
            #pragma unroll
            for (int r = 0; r < {DIM}; ++r) v[r] = V[r][j];
            for (int i = 0; i < j; ++i) {{
                {REAL} dot = 0;
                #pragma unroll
                for (int r = 0; r < {DIM}; ++r) dot += Q[r][i] * v[r];
                #pragma unroll
                for (int r = 0; r < {DIM}; ++r) v[r] -= dot * Q[r][i];
            }}
            {REAL} nrm = 0;
            #pragma unroll
            for (int r = 0; r < {DIM}; ++r) nrm += v[r] * v[r];
            nrm = {SQRT}(nrm);
            logs[j] += {LOG}({FMAX}(nrm, ({REAL})1e-300));
            {REAL} inv = ({REAL})1 / ({FMAX}(nrm, ({REAL})1e-300));
            #pragma unroll
            for (int r = 0; r < {DIM}; ++r) Q[r][j] = v[r] * inv;
        }}
    }}

    {REAL} l1 = logs[0] / ({REAL}){NSTEPS};
    {REAL} l2 = logs[1] / ({REAL}){NSTEPS};
    out_lam1[tid] = l1;
    out_r2[tid] = (l1 != 0) ? (-l2 / l1) : ({REAL})(0.0 / 0.0);
}}

}}  // extern "C"
"""


class ProxyKernel:
    """Compiled GPU proxy. Use .run(gid0, n) -> (r2, lam1) cupy arrays."""

    def __init__(self, dim, nsteps=120, box=6, z_max=1.0, dtype="float64",
                 threads=128, zs_override=None):
        import cupy as cp
        self.cp = cp
        self.dim = dim
        self.nsh = cg.nshift_for(dim)
        self.nsteps = nsteps
        self.box = box
        self.threads = threads
        self.dtype = np.float64 if dtype == "float64" else np.float32
        dirs, zs = build_pools(dim, z_max=z_max)
        if zs_override is not None:
            zs = np.asarray(zs_override, dtype=np.int64)
        self.ndir, self.nz = len(dirs), len(zs)
        self.dir_pool = cp.asarray(dirs.ravel(), dtype=cp.int64)
        self.z_pool = cp.asarray(zs.ravel(), dtype=cp.int64)
        real = "double" if self.dtype == np.float64 else "float"
        sfx = "" if real == "double" else "f"
        src = _CUDA_SRC.format(
            REAL=real, DIM=dim, NSH=self.nsh, NSTEPS=nsteps,
            BOX=box, SPAN=2 * box + 1, NDIR=self.ndir, NZ=self.nz,
            SQRT="sqrt" + sfx, LOG="log" + sfx, FMAX="fmax" + sfx)
        self.module = cp.RawModule(code=src, options=("--std=c++14",))
        self.kernel = self.module.get_function("proxy")

    def run(self, seed, gid0, n):
        cp = self.cp
        out_r2 = cp.empty(n, dtype=self.dtype)
        out_l1 = cp.empty(n, dtype=self.dtype)
        blocks = (n + self.threads - 1) // self.threads
        self.kernel((blocks,), (self.threads,),
                    (self.dir_pool, self.z_pool,
                     np.uint64(seed), np.uint64(gid0), np.uint64(n),
                     out_r2, out_l1))
        return out_r2, out_l1


def _pfq_run(xp, seed, gid0, n, dim, nsteps, box, z_max, device=None, td=None):
    """Shared pFq screening for numpy/torch: derive params from (seed,gid), group
    the batch by trajectory pattern, run the exact RamanujanTools pFq Lyapunov
    proxy. Returns (r2, lam1) numpy arrays of length n."""
    from pfq_gpu import PFqBuilder, pfq_dir_pool, map_shift_to_xy
    from params import gen_params, z_pool_arr
    istorch = xp.__name__ != "numpy"
    nsh = cg.nshift_for(dim)
    dirs = pfq_dir_pool(dim)
    zs = z_pool_arr(z_max)
    ndir, nz = len(dirs), len(zs)
    gids = np.arange(gid0, gid0 + n, dtype=np.uint64)
    shift, di, zi = gen_params(seed, gids, nsh, box, ndir, nz)
    # map every trajectory's shift -> (x_start, y_start) in disjoint ranges
    xy = [map_shift_to_xy(shift[t], dim, box) for t in range(n)]
    xstart = np.array([r[0] for r in xy], dtype=np.float64)          # (n,dim)
    ystart = np.array([r[1] for r in xy], dtype=np.float64)          # (n,dim-1)
    startmat = np.concatenate([xstart, ystart], axis=1)             # (n, nsh)
    zval = (zs[zi, 0] / zs[zi, 1]).astype(np.float64)               # (n,)

    builder = PFqBuilder(dim, dim - 1, xp)
    r2 = np.full(n, np.nan)
    lam1 = np.full(n, np.nan)
    for d in range(ndir):
        grp = np.nonzero(di == d)[0]
        if grp.size == 0:
            continue
        B = grp.size
        st_np = startmat[grp]                                       # (B,nsh)
        z_np = zval[grp]
        if istorch:
            base0 = xp.zeros(B, dtype=td, device=device)
            startv = [xp.as_tensor(st_np[:, a], dtype=td, device=device)
                      for a in range(nsh)]
            zv = xp.as_tensor(z_np, dtype=td, device=device)
        else:
            base0 = np.zeros(B)
            startv = [st_np[:, a].copy() for a in range(nsh)]
            zv = z_np.copy()
        gr2, gl1 = builder.lyapunov_batch_pfq(startv, dirs[d], zv, nsteps, base0)
        if istorch:
            gr2 = gr2.detach().cpu().numpy(); gl1 = gl1.detach().cpu().numpy()
        r2[grp] = gr2
        lam1[grp] = gl1
    return r2, lam1


def run_numpy_pfq(seed, gid0, n, dim, nsteps=120, box=6, z_max=1.0):
    return _pfq_run(np, seed, gid0, n, dim, nsteps, box, z_max)


def run_torch_pfq(seed, gid0, n, dim, nsteps=120, box=6, z_max=1.0,
                  device="cuda", dtype="float64"):
    import torch
    td = torch.float64 if dtype == "float64" else torch.float32
    with torch.no_grad():
        return _pfq_run(torch, seed, gid0, n, dim, nsteps, box, z_max,
                        device=device, td=td)


def run_numpy(seed, gid0, n, dim, nsteps=120, box=6, z_max=1.0, zs_override=None):
    """CPU fallback producing the SAME r_2 as the kernel (validation / no-GPU).
    Re-derives params from (seed, gid) then runs the batched proxy.
    If zs_override is given (Nx2 int64 array), it replaces the z pool."""
    from proxy_batch import lyapunov_batch, ratios_batch
    nsh = cg.nshift_for(dim)
    dirs, zs = (build_pools(dim, z_max=z_max) if zs_override is None
                else (build_pools(dim, z_max=z_max)[0], np.asarray(zs_override, dtype=np.int64)))
    gids = np.arange(gid0, gid0 + n, dtype=np.uint64)
    shift, di, zi = gen_params(seed, gids, nsh, box, len(dirs), len(zs))
    dirv = dirs[di].astype(float)
    zn = zs[zi, 0].astype(float)
    zd = zs[zi, 1].astype(float)
    lam = lyapunov_batch(shift.astype(float), dirv, zn, zd, dim, nsteps, np)
    r = ratios_batch(lam, np)
    return r[:, 0], lam[:, 0]


# ── Torch backend (runs in LUMI's existing ROCm PyTorch container) ──────────
def _esym_torch(vals, torch):
    B, m = vals.shape
    e = torch.zeros(B, m + 1, dtype=vals.dtype, device=vals.device)
    e[:, 0] = 1.0
    for j in range(m):
        v = vals[:, j:j + 1]
        e[:, 1:m + 1] = e[:, 1:m + 1] + v * e[:, 0:m]
    return e


def _build_M_torch(n, shift, dirv, zn, zd, dim, torch):
    B = shift.shape[0]
    f = shift[:, :dim] + n * dirv[:, :dim] + 1.0
    g = shift[:, dim:] + n * dirv[:, dim:] + 2.0
    ef = _esym_torch(f, torch)                       # (B,dim+1)
    eg = _esym_torch(g, torch)                       # (B,dim)
    eg = torch.cat([eg, torch.zeros(B, 1, dtype=eg.dtype, device=eg.device)], 1)
    hf = (-zn).view(-1, 1)
    hg = zd.view(-1, 1)
    c = torch.zeros(B, dim, dtype=shift.dtype, device=shift.device)
    c[:, 0] = hf[:, 0] * ef[:, dim]
    for k in range(1, dim):
        c[:, k] = hg[:, 0] * eg[:, dim - k] + hf[:, 0] * ef[:, dim - k]
    M = torch.zeros(B, dim, dim, dtype=shift.dtype, device=shift.device)
    for r in range(1, dim):
        M[:, r, r - 1] = 1.0
    M[:, :, dim - 1] = c
    return M


def run_torch(seed, gid0, n, dim, nsteps=120, box=6, z_max=1.0,
              device="cuda", dtype="float32", zs_override=None):
    """GPU backend via PyTorch (ROCm/CUDA). Same r_2 as the kernel/numpy ref.
    Uses batched einsum + torch.linalg.qr; no cupy needed.
    If zs_override is given (Nx2 int64 array), it replaces the z pool."""
    import torch
    td = torch.float64 if dtype == "float64" else torch.float32
    nsh = cg.nshift_for(dim)
    if zs_override is None:
        dirs, zs = build_pools(dim, z_max=z_max)
    else:
        dirs, _ = build_pools(dim, z_max=z_max)
        zs = np.asarray(zs_override, dtype=np.int64)
    gids = np.arange(gid0, gid0 + n, dtype=np.uint64)
    shift_np, di, zi = gen_params(seed, gids, nsh, box, len(dirs), len(zs))
    with torch.no_grad():
        shift = torch.as_tensor(shift_np.astype(np.float64), device=device, dtype=td)
        dirv = torch.as_tensor(dirs[di].astype(np.float64), device=device, dtype=td)
        zn = torch.as_tensor(zs[zi, 0].astype(np.float64), device=device, dtype=td)
        zd = torch.as_tensor(zs[zi, 1].astype(np.float64), device=device, dtype=td)
        Q = torch.eye(dim, dtype=td, device=device).expand(n, dim, dim).contiguous()
        logs = torch.zeros(n, dim, dtype=td, device=device)
        for step in range(1, nsteps + 1):
            M = _build_M_torch(step, shift, dirv, zn, zd, dim, torch)
            V = torch.einsum('bij,bjk->bik', M, Q)
            # Manual modified Gram-Schmidt over columns of V (batched, no rocSOLVER):
            # only elementwise + reductions -> fast on ROCm/CUDA. r_jj = ||.|| > 0
            # so no sign convention needed. Matches the cupy in-register kernel.
            Qn = torch.empty_like(V)
            for j in range(dim):
                v = V[:, :, j].clone()
                for i in range(j):
                    qi = Qn[:, :, i]
                    dot = (qi * v).sum(dim=1, keepdim=True)
                    v = v - dot * qi
                nrm = torch.clamp(v.norm(dim=1, keepdim=True), min=1e-30)
                logs[:, j] = logs[:, j] + torch.log(nrm.squeeze(1))
                Qn[:, :, j] = v / nrm
            Q = Qn
        lam = logs / nsteps
        l1 = lam[:, 0]
        r2 = torch.where(l1 != 0, -lam[:, 1] / l1,
                         torch.full_like(l1, float("nan")))
    return r2.detach().cpu().numpy(), l1.detach().cpu().numpy()
