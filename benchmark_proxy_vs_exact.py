#!/usr/bin/env python3
"""Matched CPU benchmark after product-order and observable corrections.

The comparator is pure-Python integer arithmetic, not GPU RNS. The script can
measure one fixed row-pair observable or the explicitly declared finite-family
statistic delta_O^max over all ordered row pairs.
"""
from __future__ import annotations
import argparse, json, os, platform, socket, sys, time
from multiprocessing import Pool
import numpy as np

ROOT=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT);sys.path.insert(0,os.path.join(ROOT,"gpu_proxy"))
import cmf_generic as cg
from params import gen_params,build_pools


def env_info():
    return {"hostname":socket.gethostname(),"platform":platform.platform(),
            "python":platform.python_version(),"numpy":np.__version__}


def gen_candidates(seed,n,dim,box):
    dirs,zs=build_pools(dim,z_max=1.0);gids=np.arange(n,dtype=np.uint64)
    sh,di,zi=gen_params(seed,gids,cg.nshift_for(dim),box,len(dirs),len(zs))
    return [{"gid":int(gids[k]),"shift":sh[k].tolist(),"dir":dirs[di[k]].tolist(),
             "z_num":int(zs[zi[k],0]),"z_den":int(zs[zi[k],1])} for k in range(n)]


def proxy_one(task):
    rec,dim,N=task
    r,lam=cg.spectral_ratios(rec["shift"],rec["dir"],rec["z_num"],rec["z_den"],N,dim)
    return (float(r[0]) if len(r) and np.isfinite(r[0]) else float("nan"), float(lam[0]))


def exact_one(task):
    rec,dim,N,row_pair=task
    value,pair=cg.independent_delta(rec["shift"],rec["dir"],rec["z_num"],rec["z_den"],
                                    N,dim,row_pair=row_pair,return_pair=True)
    return value,pair


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dim",type=int,default=6);ap.add_argument("--n-candidates",type=int,default=2000)
    ap.add_argument("--depth",type=int,default=120);ap.add_argument("--box",type=int,default=6)
    ap.add_argument("--seed",type=int,default=20260626);ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--thresh",type=float,default=0.02);ap.add_argument("--n-exact",type=int,default=500)
    ap.add_argument("--row-pair",default=None,help="fixed i,j; omit for delta_O^max")
    ap.add_argument("--out",default="benchmark_results_corrected.json")
    args=ap.parse_args()
    row_pair=tuple(map(int,args.row_pair.split(','))) if args.row_pair else None
    cands=gen_candidates(args.seed,args.n_candidates,args.dim,args.box)

    pargs=[(c,args.dim,args.depth) for c in cands]
    with Pool(args.workers) as p:p.map(proxy_one,pargs[:min(16,len(pargs))])
    t=time.time()
    with Pool(args.workers) as p:proxy_res=p.map(proxy_one,pargs)
    proxy=np.asarray([x[0] for x in proxy_res],float)
    proxy_lam1=np.asarray([x[1] for x in proxy_res],float)
    proxy_s=time.time()-t

    n_ex=min(args.n_exact,len(cands));eargs=[(c,args.dim,args.depth,row_pair) for c in cands[:n_ex]]
    with Pool(args.workers) as p:p.map(exact_one,eargs[:min(4,len(eargs))])
    t=time.time()
    with Pool(args.workers) as p:exact_res=p.map(exact_one,eargs)
    exact_s=time.time()-t
    exact=np.asarray([x[0] if x[0] is not None else np.nan for x in exact_res],float)
    pairs=[x[1] for x in exact_res]

    pd=proxy[:n_ex]; pl1=proxy_lam1[:n_ex]
    nondeg=np.asarray([not cg.is_degenerate(c["shift"],c["dir"],args.dim,1,args.depth) for c in cands[:n_ex]])
    eligible=nondeg & (pl1>0)
    fin=np.isfinite(pd)&np.isfinite(exact)&eligible
    pp=pd[fin]>args.thresh;ep=exact[fin]>0
    tp=int(np.sum(pp&ep));fp=int(np.sum(pp&~ep));fn=int(np.sum(~pp&ep));tn=int(np.sum(~pp&~ep))
    all_nondeg=np.asarray([not cg.is_degenerate(c["shift"],c["dir"],args.dim,1,args.depth) for c in cands])
    all_eligible=all_nondeg & (proxy_lam1>0) & np.isfinite(proxy)
    survivors=int(np.sum(all_eligible & (proxy>args.thresh)))
    pthr=len(cands)/proxy_s;ethr=n_ex/exact_s
    verify_est=survivors/ethr if survivors else 0.0
    total_proxy=proxy_s+verify_est;total_exact=len(cands)/ethr
    result={
      "environment":env_info(),"parameters":vars(args),
      "product_order":"P_N=M_1...M_N; QR on M_n^T",
      "filters":{"nondegenerate":True,"lambda1_positive":True,"eligible_exact_n":int(np.sum(eligible)),"eligible_all_n":int(np.sum(all_eligible))},
      "observable":{"kind":"fixed_row_pair" if row_pair else "delta_O_max",
                    "row_pair":row_pair,"family":"all ordered row pairs" if row_pair is None else None},
      "timing":{"proxy_s":proxy_s,"proxy_per_s":pthr,"exact_s":exact_s,
                "exact_n":n_ex,"exact_per_s":ethr},
      "metrics":{"raw_speedup":pthr/ethr,
                 "sign_agreement":float(np.mean((pd[fin]>0)==(exact[fin]>0))) if np.any(fin) else None,
                 "mae":float(np.mean(np.abs(pd[fin]-exact[fin]))) if np.any(fin) else None,
                 "tp":tp,"fp":fp,"fn":fn,"tn":tn,
                 "precision":tp/(tp+fp) if tp+fp else None,
                 "recall":tp/(tp+fn) if tp+fn else None,
                 "proxy_survivors":survivors,
                 "e2e_speedup_estimate":total_exact/total_proxy if total_proxy else None,
                 "cost_per_hit_s":total_proxy/tp if tp else None,
                 "n_exact_positive":int(np.sum(ep)),"n_proxy_positive":int(np.sum(pp))},
      "maximizing_pairs":pairs if row_pair is None else None,
    }
    with open(args.out,'w') as f:json.dump(result,f,indent=2,default=str)
    print(json.dumps(result["metrics"],indent=2));print("wrote",args.out)


if __name__=="__main__":main()
