#!/usr/bin/env python3
"""Matched float32/float64 versus exact validation after product-order correction.

The exact target is delta_O^max over the fixed finite family of ordered row-pair
observables. This is stated explicitly; no result is presented as an independent
fixed-readout J* identification.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

ROOT=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT);sys.path.insert(0,os.path.join(ROOT,"gpu_proxy"))
import cmf_generic as cg
from params import build_pools, gen_params


def qr_dtype(rec, dim, N, dtype):
    Q=np.eye(dim,dtype=dtype)
    logs=np.zeros(dim,dtype=np.float64)
    tiny=np.finfo(dtype).tiny
    for n in range(1,N+1):
        M=cg.build_M_float(n,rec["shift"],rec["dir"],rec["z_num"],rec["z_den"],dim).astype(dtype)
        Y=M.T @ Q
        Q,R=np.linalg.qr(Y)
        diag=np.diag(R)
        logs += np.log(np.maximum(np.abs(diag).astype(np.float64),float(tiny)))
        signs=np.where(diag<0,dtype(-1),dtype(1))
        Q=Q*signs[None,:]
    lam=logs/N
    ratios=-lam[1:]/lam[0] if lam[0]!=0 else np.full(dim-1,np.nan)
    return ratios,lam


def metrics(rows,key):
    ex=np.array([r["exact_delta_max"] for r in rows],float)
    pr=np.array([r[key] for r in rows],float)
    finite=np.isfinite(ex)&np.isfinite(pr)
    ex=ex[finite];pr=pr[finite]
    if len(ex)==0:return {"n":0}
    return {"n":int(len(ex)),
      "sign_agreement":float(np.mean((ex>0)==(pr>0))),
      "mae":float(np.mean(np.abs(ex-pr))),
      "max_abs_error":float(np.max(np.abs(ex-pr))),
      "false_negative":int(np.sum((ex>0)&(pr<=0))),
      "false_positive":int(np.sum((ex<=0)&(pr>0)))}


def evaluate(rec,dim,N,label):
    exact,pair=cg.independent_delta(rec["shift"],rec["dir"],rec["z_num"],rec["z_den"],N,dim,return_pair=True)
    r64,l64=qr_dtype(rec,dim,N,np.float64)
    r32,l32=qr_dtype(rec,dim,N,np.float32)
    return {**rec,"label":label,"exact_delta_max":exact,"max_pair":pair,
            "r2_float64":float(r64[0]),"r2_float32":float(r32[0]),
            "ratios_float64":r64.tolist(),"ratios_float32":r32.tolist(),
            "lambda1_float64":float(l64[0]),"lambda1_float32":float(l32[0]),
            "degenerate":cg.is_degenerate(rec["shift"],rec["dir"],dim,1,N)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dim",type=int,default=6);ap.add_argument("--depth",type=int,default=80)
    ap.add_argument("--random",type=int,default=150);ap.add_argument("--attempts",type=int,default=2000)
    ap.add_argument("--boundary",type=float,default=0.05);ap.add_argument("--boundary-n",type=int,default=20)
    ap.add_argument("--box",type=int,default=6);ap.add_argument("--seed",type=int,default=314159)
    ap.add_argument("--out",default="precision_validation_results.json")
    args=ap.parse_args()

    dirs,zs=build_pools(args.dim,z_max=1.0)
    gids=np.arange(args.attempts,dtype=np.uint64)
    shifts,di,zi=gen_params(args.seed,gids,cg.nshift_for(args.dim),args.box,len(dirs),len(zs))
    rows=[];boundary=[];generic_count=0
    for k in range(args.attempts):
        rec={"gid":int(gids[k]),"shift":shifts[k].tolist(),"dir":dirs[di[k]].tolist(),
             "z_num":int(zs[zi[k],0]),"z_den":int(zs[zi[k],1])}
        if cg.is_degenerate(rec["shift"],rec["dir"],args.dim,1,args.depth):
            continue
        result=evaluate(rec,args.dim,args.depth,"generic" if generic_count<args.random else "candidate")
        if generic_count<args.random:
            rows.append(result);generic_count+=1
        if result["exact_delta_max"] is not None and abs(result["exact_delta_max"])<=args.boundary and len(boundary)<args.boundary_n:
            boundary.append({**result,"label":"boundary"})
        if generic_count>=args.random and len(boundary)>=args.boundary_n:
            break
    rows.extend(boundary)

    shift=[-2,-2,0,0,-2,0,-2,-2,-2,0,-2]
    dirv=[0,0,0,0,0,0,0,0,0,1,0]
    for zn,zd in [(7,20),(1,3),(-1,3),(1,1),(-1,1)]:
        rows.append(evaluate({"gid":None,"shift":shift,"dir":dirv,"z_num":zn,"z_den":zd},
                             args.dim,args.depth,"known_positive"))

    strata={}
    for label in sorted({r["label"] for r in rows}):
        subset=[r for r in rows if r["label"]==label]
        strata[label]={"float64":metrics(subset,"r2_float64"),
                       "float32":metrics(subset,"r2_float32"),"n":len(subset)}
    result={"parameters":vars(args),"product_order":"P_N=M_1...M_N; QR on transposes",
            "observable":"delta_O^max over ordered row pairs","strata":strata,"rows":rows}
    with open(args.out,"w") as f:json.dump(result,f,indent=2,default=str)
    print(json.dumps(strata,indent=2));print("wrote",args.out)


if __name__=="__main__":main()
