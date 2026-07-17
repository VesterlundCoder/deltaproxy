import os
import sys
import unittest
import numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
import cmf_generic as cg
from spectral_delta import lyapunov_spectrum


class ProductOrderTests(unittest.TestCase):
    def setUp(self):
        self.dim=3
        self.shift=[0,1,2,-1,0]
        self.dir=[1,0,0,1,0]
        self.z_num=1
        self.z_den=3

    def test_incremental_qr_matches_direct_qr_of_right_product_transpose(self):
        N=8
        P=np.eye(self.dim)
        for n in range(1,N+1):
            P=P@cg.build_M_float(n,self.shift,self.dir,self.z_num,self.z_den,self.dim)
        _,R=np.linalg.qr(P.T)
        direct=np.log(np.maximum(np.abs(np.diag(R)),1e-300))/N
        incremental=lyapunov_spectrum(
            lambda k: cg.build_M_float(k+1,self.shift,self.dir,self.z_num,self.z_den,self.dim),
            self.dim,N,product_order='right')
        np.testing.assert_allclose(incremental,direct,rtol=1e-9,atol=1e-9)

    def test_integer_and_float_right_products_agree_at_small_depth(self):
        N=5
        Pi=[[1 if i==j else 0 for j in range(self.dim)] for i in range(self.dim)]
        Pf=np.eye(self.dim)
        for n in range(1,N+1):
            Mi=cg.build_M_int(n,self.shift,self.dir,self.z_num,self.z_den,self.dim)
            Pi=cg.matmul_int(Pi,Mi,self.dim)
            Pf=Pf@cg.build_M_float(n,self.shift,self.dir,self.z_num,self.z_den,self.dim)
        np.testing.assert_allclose(Pf,np.asarray(Pi,dtype=float),rtol=0,atol=0)

    def test_fixed_pair_is_not_silently_maximized(self):
        N=20
        values=cg.pair_deltas_int(self.shift,self.dir,self.z_num,self.z_den,N,self.dim)
        max_value,max_pair=cg.independent_delta(
            self.shift,self.dir,self.z_num,self.z_den,N,self.dim,return_pair=True)
        self.assertAlmostEqual(max_value,max(values.values()))
        fixed=next(iter(values))
        self.assertEqual(
            cg.independent_delta(self.shift,self.dir,self.z_num,self.z_den,N,self.dim,row_pair=fixed),
            values[fixed])
        self.assertIn(max_pair,values)

    def test_dynamic_zero_detection(self):
        shift=[-3,0,0,-1,0]
        direction=[1,0,0,0,0]
        events=cg.numerator_zero_steps(shift,direction,3,1,10)
        self.assertIn((0,'dynamic',2),events)


if __name__=='__main__':
    unittest.main()
