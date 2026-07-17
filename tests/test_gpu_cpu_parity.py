import os,sys,unittest
import numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT);sys.path.insert(0,os.path.join(ROOT,'gpu_proxy'))
import cmf_generic as cg
from proxy_kernel import run_numpy
from params import build_pools,gen_params

class CpuLadderTests(unittest.TestCase):
    def test_r2_is_first_full_ladder_column(self):
        r2,l1=run_numpy(123,0,32,6,nsteps=20,full_ladder=False)
        ladder,l1b=run_numpy(123,0,32,6,nsteps=20,full_ladder=True)
        np.testing.assert_allclose(r2,ladder[:,0])
        np.testing.assert_allclose(l1,l1b)

    def test_float32_float64_sign_parity_on_eligible_batch(self):
        seed=456;n=128;N=30;dim=6;box=6
        r32,l32=run_numpy(seed,0,n,dim,nsteps=N,dtype='float32')
        r64,l64=run_numpy(seed,0,n,dim,nsteps=N,dtype='float64')
        dirs,zs=build_pools(dim,1.0)
        shifts,di,zi=gen_params(seed,np.arange(n,dtype=np.uint64),cg.nshift_for(dim),box,len(dirs),len(zs))
        eligible=np.array([not cg.is_degenerate(shifts[k],dirs[di[k]],dim,1,N) for k in range(n)])
        m=eligible & np.isfinite(r32)&np.isfinite(r64)&(l32>0)&(l64>0)
        self.assertGreater(m.sum(),10)
        self.assertGreaterEqual(np.mean((r32[m]>0)==(r64[m]>0)),0.99)

if __name__=='__main__':unittest.main()
