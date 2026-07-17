"""GPU spectral proxy for the right-product cocycle P_N=M_1...M_N.

The kernel accumulates every QR growth rate. It can return either r_2 only or
the full ladder r_2,...,r_d. QR is applied to M_n^T so that it analyses the
same product as the exact integer engine.
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cmf_generic as cg
from params import build_pools, gen_params

_CUDA_SRC = r'''extern "C" {{
__device__ __forceinline__ unsigned long long sm_next(unsigned long long* s) {{
    *s += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = *s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}}
__device__ __forceinline__ void esym(const {REAL}* vals, int m, {REAL}* e) {{
    for (int k=0;k<=m;++k) e[k]=0;
    e[0]=1;
    for (int j=0;j<m;++j)
        for (int k=m;k>=1;--k) e[k]+=vals[j]*e[k-1];
}}
__global__ void proxy(const long long* dir_pool, const long long* z_pool,
                      unsigned long long seed, unsigned long long gid0,
                      unsigned long long n_traj, {REAL}* out_ratios,
                      {REAL}* out_lam1) {{
    unsigned long long tid=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=n_traj) return;
    unsigned long long gid=gid0+tid;
    unsigned long long state=seed ^ ((gid+1ULL)*0x9E3779B97F4A7C15ULL);
    long long shift[{NSH}];
    #pragma unroll
    for(int i=0;i<{NSH};++i) shift[i]=(long long)(sm_next(&state)%{SPAN}ULL)-{BOX}LL;
    int di=(int)(sm_next(&state)%{NDIR}ULL);
    int zi=(int)(sm_next(&state)%{NZ}ULL);
    const long long* dir=dir_pool+(long long)di*{NSH};
    long long zn=z_pool[2*zi], zd=z_pool[2*zi+1];

    {REAL} Q[{DIM}][{DIM}], logs[{DIM}];
    #pragma unroll
    for(int i=0;i<{DIM};++i) {{
        logs[i]=0;
        #pragma unroll
        for(int j=0;j<{DIM};++j) Q[i][j]=(i==j)?1:0;
    }}

    for(int n=1;n<={NSTEPS};++n) {{
        {REAL} f[{DIM}],g[{DIM}],ef[{DIM}+1],eg[{DIM}],c[{DIM}];
        #pragma unroll
        for(int i=0;i<{DIM};++i) f[i]=({REAL})shift[i]+({REAL})n*dir[i]+1;
        #pragma unroll
        for(int j=0;j<{DIM}-1;++j) g[j]=({REAL})shift[{DIM}+j]+({REAL})n*dir[{DIM}+j]+2;
        esym(f,{DIM},ef); esym(g,{DIM}-1,eg);
        c[0]=({REAL})(-zn)*ef[{DIM}];
        #pragma unroll
        for(int k=1;k<{DIM};++k) c[k]=({REAL})zd*eg[{DIM}-k]+({REAL})(-zn)*ef[{DIM}-k];

        // V=M^T Q. M has subdiagonal ones and last column c.
        {REAL} V[{DIM}][{DIM}];
        #pragma unroll
        for(int j=0;j<{DIM};++j) {{
            #pragma unroll
            for(int r=0;r<{DIM}-1;++r) V[r][j]=Q[r+1][j];
            {REAL} acc=0;
            #pragma unroll
            for(int k=0;k<{DIM};++k) acc+=c[k]*Q[k][j];
            V[{DIM}-1][j]=acc;
        }}

        #pragma unroll
        for(int j=0;j<{DIM};++j) {{
            {REAL} v[{DIM}];
            #pragma unroll
            for(int r=0;r<{DIM};++r) v[r]=V[r][j];
            for(int i=0;i<j;++i) {{
                {REAL} dot=0;
                #pragma unroll
                for(int r=0;r<{DIM};++r) dot+=Q[r][i]*v[r];
                #pragma unroll
                for(int r=0;r<{DIM};++r) v[r]-=dot*Q[r][i];
            }}
            {REAL} nrm=0;
            #pragma unroll
            for(int r=0;r<{DIM};++r) nrm+=v[r]*v[r];
            nrm={FMAX}({SQRT}(nrm),({REAL}){TINY});
            logs[j]+={LOG}(nrm);
            {REAL} inv=({REAL})1/nrm;
            #pragma unroll
            for(int r=0;r<{DIM};++r) Q[r][j]=v[r]*inv;
        }}
    }}
    {REAL} l1=logs[0]/({REAL}){NSTEPS};
    out_lam1[tid]=l1;
    #pragma unroll
    for(int k=1;k<{DIM};++k) {{
        {REAL} lk=logs[k]/({REAL}){NSTEPS};
        out_ratios[tid*({DIM}-1)+(k-1)]=(l1!=0)?(-lk/l1):({REAL})(0.0/0.0);
    }}
}}
}}'''


class ProxyKernel:
    def __init__(self, dim, nsteps=120, box=6, z_max=1.0, dtype="float64",
                 threads=128, zs_override=None):
        import cupy as cp
        self.cp = cp; self.dim = dim; self.nsteps = nsteps; self.box = box
        self.threads = threads
        self.dtype = np.float64 if dtype == "float64" else np.float32
        dirs, zs = build_pools(dim, z_max=z_max)
        if zs_override is not None: zs = np.asarray(zs_override, dtype=np.int64)
        self.dir_pool = cp.asarray(dirs.ravel(), dtype=cp.int64)
        self.z_pool = cp.asarray(zs.ravel(), dtype=cp.int64)
        real = "double" if self.dtype == np.float64 else "float"
        suffix = "" if real == "double" else "f"
        src = _CUDA_SRC.format(
            REAL=real, DIM=dim, NSH=cg.nshift_for(dim), NSTEPS=nsteps,
            BOX=box, SPAN=2*box+1, NDIR=len(dirs), NZ=len(zs),
            SQRT="sqrt"+suffix, LOG="log"+suffix, FMAX="fmax"+suffix,
            TINY="1e-300" if real == "double" else "1e-30f")
        self.module = cp.RawModule(code=src, options=("--std=c++14",))
        self.kernel = self.module.get_function("proxy")

    def run(self, seed, gid0, n, *, full_ladder=False):
        cp = self.cp
        ratios = cp.empty((n, self.dim-1), dtype=self.dtype)
        lam1 = cp.empty(n, dtype=self.dtype)
        blocks = (n + self.threads - 1)//self.threads
        self.kernel((blocks,), (self.threads,),
                    (self.dir_pool, self.z_pool, np.uint64(seed), np.uint64(gid0),
                     np.uint64(n), ratios, lam1))
        return (ratios, lam1) if full_ladder else (ratios[:, 0], lam1)


def _params(seed, gid0, n, dim, box, z_max, zs_override=None):
    dirs, zs = build_pools(dim, z_max=z_max)
    if zs_override is not None: zs = np.asarray(zs_override, dtype=np.int64)
    gids = np.arange(gid0, gid0+n, dtype=np.uint64)
    shift, di, zi = gen_params(seed, gids, cg.nshift_for(dim), box, len(dirs), len(zs))
    return shift, dirs[di], zs[zi,0], zs[zi,1]


def run_numpy(seed, gid0, n, dim, nsteps=120, box=6, z_max=1.0,
              zs_override=None, full_ladder=False, dtype="float64"):
    from proxy_batch import lyapunov_batch, ratios_batch
    shift, dirv, zn, zd = _params(seed, gid0, n, dim, box, z_max, zs_override)
    td = np.float32 if dtype == "float32" else np.float64
    lam = lyapunov_batch(shift.astype(td), dirv.astype(td), zn.astype(td), zd.astype(td),
                         dim, nsteps, np)
    ratios = ratios_batch(lam, np)
    return (ratios, lam[:,0]) if full_ladder else (ratios[:,0], lam[:,0])


def run_torch(seed, gid0, n, dim, nsteps=120, box=6, z_max=1.0,
              device="cuda", dtype="float32", zs_override=None,
              full_ladder=False):
    import torch
    from proxy_batch import build_M_batch
    td = torch.float64 if dtype == "float64" else torch.float32
    shift_np, dir_np, zn_np, zd_np = _params(seed, gid0, n, dim, box, z_max, zs_override)
    with torch.no_grad():
        shift = torch.as_tensor(shift_np, dtype=td, device=device)
        dirv = torch.as_tensor(dir_np, dtype=td, device=device)
        zn = torch.as_tensor(zn_np, dtype=td, device=device)
        zd = torch.as_tensor(zd_np, dtype=td, device=device)
        Q = torch.eye(dim, dtype=td, device=device).expand(n,dim,dim).contiguous()
        logs = torch.zeros(n,dim,dtype=td,device=device)
        for step in range(1,nsteps+1):
            M = build_M_batch(step, shift, dirv, zn, zd, dim, torch)
            V = torch.einsum('bij,bjk->bik', M.transpose(1,2), Q)
            Qn = torch.empty_like(V)
            for j in range(dim):
                v = V[:,:,j].clone()
                for i in range(j):
                    qi = Qn[:,:,i]
                    v -= (qi*v).sum(dim=1,keepdim=True)*qi
                nrm = torch.clamp(v.norm(dim=1,keepdim=True), min=1e-30)
                logs[:,j] += torch.log(nrm.squeeze(1)); Qn[:,:,j] = v/nrm
            Q = Qn
        lam = logs/float(nsteps); ratios = -lam[:,1:]/lam[:,:1]
    ratios_np = ratios.detach().cpu().numpy(); l1_np = lam[:,0].detach().cpu().numpy()
    return (ratios_np,l1_np) if full_ladder else (ratios_np[:,0],l1_np)


def run_numpy_pfq(*args, **kwargs):
    raise NotImplementedError("pFq backend requires a separately audited right-product implementation")


def run_torch_pfq(*args, **kwargs):
    raise NotImplementedError("pFq backend requires a separately audited right-product implementation")
