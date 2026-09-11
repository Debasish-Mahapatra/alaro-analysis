#!/usr/bin/env python3
"""Native-grid IOP versus continuous rainfall and updraft workflow.

This deliberately reads the raw FA stream (rather than a cropped converter
product), and caches one complete date/experiment reduction at a time.  It is
safe to rerun while IOP extraction is in progress: only dates with
``.untar_done`` and all required lead files are admitted to a comparison.
"""
from __future__ import annotations
import argparse, json, os, re, time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import epygram

ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS")
CONT = ROOT / "ALARO"; IOP = ROOT / "iop-runs" / "iop-runs"
# Actual extraction tree is .../iop-runs/iop{1,2}/<exp>/untar-output.
IOP = ROOT / "iop-runs"
OUT = IOP / "figures"; CACHE = IOP / "analysis" / "cache"; LOG = IOP / "analysis" / "logs"
EXPS = ("NO3M", "graupel", "2mom")
LABEL = {"NO3M":"G2M-XCU", "graupel":"G1M", "2mom":"G2M"}
VARS = ("SURFPREC.EAU.GEC","SURFPREC.EAU.CON","SURFPREC.NEI.GEC","SURFPREC.NEI.CON","SURFPREC.GRA.GEC")
PERIODS = {1:(date(2014,1,15),date(2014,2,28)), 2:(date(2014,7,29),date(2014,8,31))}
G = 9.80665; OFFSET = -4; NLEV = 87; WORKERS = 32
FH = re.compile(r"\+(\d{4})$")

def fa(path):
    return epygram.formats.resource(str(path), "r")

def read(r, name):
    f = r.readfield(name)
    if getattr(f, "spectral", False): f.sp2gp()
    return np.asarray(f.getdata(), dtype=np.float64)

def fpath(root, d, h): return root / f"pf{d:%Y%m%d}" / f"pfABOFABOF+{h:04d}"

def required(root, d, rain=True):
    hs = range(25) if rain else range(1,25)
    return all(fpath(root,d,h).is_file() for h in hs)

def available(root, start, end):
    out=[]; d=start
    while d<=end:
        p=root/f"pf{d:%Y%m%d}"
        # Continuous untar trees predate the marker convention; their
        # complete 25-file day is the completion evidence.  IOP extraction
        # must provide .untar_done so an in-progress day is never compared.
        continuous = root.parent.parent.name == "ALARO"
        if required(root,d) and (continuous or (p/".untar_done").exists()): out.append(d)
        d += timedelta(days=1)
    return out

def preflight():
    LOG.mkdir(parents=True, exist_ok=True); stamp=__import__('datetime').datetime.now().isoformat()
    rec={"timestamp":stamp,"periods":{}}
    for i,(a,b) in PERIODS.items():
        key=f"iop{i}"; rec["periods"][key]={}
        for e in EXPS+('control',):
            roots=[IOP/key/e/'untar-output'] if e!='control' else [CONT/e/'untar-output']
            got=available(roots[0],a,b)
            rec["periods"][key][e]={"available":[x.isoformat() for x in got],"missing":[(a+timedelta(days=j)).isoformat() for j in range((b-a).days+1) if a+timedelta(days=j) not in got]}
    (LOG/"iop_rainfall_updraft_manifest.json").write_text(json.dumps(rec,indent=2))
    print(json.dumps(rec,indent=2)); return rec

def reduce_day(root,d):
    # Include source tree in the cache key: continuous and IOP both contain
    # ``<experiment>/untar-output`` and must never share reductions.
    source_key = f"{root.parent.parent.name}_{root.parent.name}"
    cache=CACHE/source_key/f"{d:%Y%m%d}.npz"; cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        old=dict(np.load(cache,allow_pickle=False))
        if old.get('hourly', np.empty(0)).shape == (24,)+old.get('rain', np.empty(0)).shape:
            return old
        print(f"invalidating old cache with bad hourly shape: {cache}", flush=True)
    rain=[]; prof={k:[] for k in ('extent','flux','cloud','temp')}; height=[]
    for h in range(25):
        r=fa(fpath(root,d,h))
        try:
            x=sum(read(r,v) for v in VARS); rain.append(x)
            if h>0:
                ext=[]; flux=[]; cloud=[]; temp=[]; z=[]
                for lev in range(1,NLEV+1):
                    mesh=read(r,f"S{lev:03d}UD_MESH_FRAC"); omg=read(r,f"S{lev:03d}UD_OMEGA")
                    cl=read(r,f"S{lev:03d}CLOUD_FRACTI"); te=read(r,f"S{lev:03d}TEMPERATURE"); gg=read(r,f"S{lev:03d}GEOPOTENTIEL")
                    finite=np.isfinite(mesh)&np.isfinite(omg)
                    ext.append(np.nanmean(mesh)); flux.append(np.nanmean(np.where(finite,np.where(mesh>0,-omg*mesh/G,0.),np.nan)))
                    cloud.append(np.nanmean(cl)); temp.append(np.nanmean(te)); z.append(np.nanmean(gg)/G)
                for k,v in (('extent',ext),('flux',flux),('cloud',cloud),('temp',temp)): prof[k].append(v)
                height.append(z)
        finally: r.close()
    a=np.stack(rain); out={'rain':a[24]-a[0], 'hourly':np.diff(a, axis=0), 'height':np.nanmean(height,axis=0), 'schema':np.array('rain_updraft_v2')}
    for k,v in prof.items(): out[k]=np.stack(v)
    np.savez(cache,**out); return out

def _reduce_task(task):
    root, dstr = task
    return str(root), dstr, reduce_day(Path(root), date.fromisoformat(dstr))['rain'].shape

def warm_continuous(key, available):
    tasks=[]
    for e in EXPS:
        root=CONT/e/'untar-output'
        tasks.extend((str(root), dstr) for dstr in available)
    if not tasks: return
    print(f"{key}: warming {len(tasks)} continuous date caches with {WORKERS} workers", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        done=0
        for _ in pool.map(_reduce_task,tasks,chunksize=1):
            done += 1
            if done % 5 == 0 or done == len(tasks): print(f"{key}: continuous cache {done}/{len(tasks)}",flush=True)

def plot_pair(i,e,dates,other,cont):
    base=f"iop{i}_{e}_smoke"; OUT.mkdir(parents=True,exist_ok=True)
    m=np.nanmean([x['rain'] for x in other],axis=0); c=np.nanmean([x['rain'] for x in cont],axis=0); delta=m-c
    # Native FA lon/lat are read once; no crop or radar mask is applied.
    rr=fa(fpath(CONT/e/'untar-output',dates[0],0));
    try:
        geom=rr.geometry; lon=np.asarray(geom.get_lonlat_grid()[0]); lat=np.asarray(geom.get_lonlat_grid()[1])
    finally: rr.close()
    absmax=float(np.nanpercentile(np.stack([c,m]),99.5)); dmax=float(np.nanpercentile(np.abs(delta),99.5)); dmax=max(dmax,1e-12)
    fig,ax=plt.subplots(1,3,figsize=(20,6),constrained_layout=True)
    for j,(a,z,t,cm,norm) in enumerate(zip(ax,(c,m,delta),(f'{LABEL[e]} continuous',f'{LABEL[e]} IOP',f'{LABEL[e]} IOP − continuous'),('viridis','viridis','RdBu_r'),(None,None,None))):
        vmin,vmax=(0,absmax) if j<2 else (-dmax,dmax)
        p=a.pcolormesh(lon,lat,z,cmap=cm,vmin=vmin,vmax=vmax,shading='auto'); a.set_title(t); a.set_xlabel('longitude (deg)'); a.set_ylabel('latitude (deg)'); cb=fig.colorbar(p,ax=a); cb.set_label('mm day$^{-1}$'); cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f'{v:g}'))
    out=OUT/f'{base}_rainfall_maps.png'; fig.savefig(out,dpi=450); plt.close(fig)
    # Hourly fields are UTC interval starts; reorder to local UTC-4 display.
    order=np.argsort((np.arange(24)+OFFSET)%24); lh=(np.arange(24)+OFFSET)%24
    cd=np.nanmean([x['hourly'].mean(axis=(1,2)) for x in cont],axis=0); md=np.nanmean([x['hourly'].mean(axis=(1,2)) for x in other],axis=0); dd=md-cd
    fig,ax=plt.subplots(figsize=(11,6)); ax.plot(lh[order],cd[order],label=f'{LABEL[e]} continuous'); ax.plot(lh[order],md[order],label=f'{LABEL[e]} IOP'); ax.plot(lh[order],dd[order],label='IOP − continuous'); ax.set_xlabel('local hour (UTC-4)'); ax.set_ylabel('total precipitation (mm h$^{-1}$)'); ax.set_xticks(range(24)); ax.grid(alpha=.3); ax.legend()
    out2=OUT/f'{base}_rainfall_diurnal.png'; fig.savefig(out2,dpi=450); plt.close(fig)
    # Profiles have columns endpoint hours 1..24; display those endpoints in local time.
    z=np.nanmean([x['height'] for x in cont],axis=0)/1000; profout=[]; fig,ax=plt.subplots(3,3,figsize=(20,18),constrained_layout=True)
    for r,k,unit in ((0,'extent','fraction'),(1,'flux','kg m$^{-2}$ s$^{-1}$'),(2,'cloud','fraction')):
        cv=np.nanmean([x[k] for x in cont],axis=0); iv=np.nanmean([x[k] for x in other],axis=0); dv=iv-cv; av=float(np.nanpercentile(np.stack([cv,iv]),99)); q=max(float(np.nanpercentile(np.abs(dv),99)),1e-12)
        for j,(v,cm,vmin,vmax,title) in enumerate(((cv,'viridis',0,av,f'{LABEL[e]} continuous'),(iv,'viridis',0,av,f'{LABEL[e]} IOP'),(dv,'RdBu_r',-q,q,f'{LABEL[e]} IOP − continuous'))):
            p=ax[r,j].pcolormesh(lh[order],z,v[order,:].T,cmap=cm,vmin=vmin,vmax=vmax,shading='auto'); ax[r,j].set_ylim(0,16); ax[r,j].set_xlabel('local endpoint hour (UTC-4)'); ax[r,j].set_ylabel('height (km)'); ax[r,j].set_title(f'{title}: {k}'); cb=fig.colorbar(p,ax=ax[r,j]); cb.set_label(unit); cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x:g}'))
            if j < 2:
                run = cont if j == 0 else other
                tt=np.nanmean([x['temp'] for x in run],axis=0); zz=np.nanmean([x['height'] for x in run],axis=0)/1000
                freeze=[]
                for h in range(24):
                    good=np.isfinite(zz)&np.isfinite(tt[h]); orderz=np.argsort(zz[good]); zs=zz[good][orderz]; ts=tt[h][good][orderz]; cross=np.where(np.diff(np.sign(ts-273.15))!=0)[0]
                    freeze.append(float(np.interp(273.15,ts[cross[0]:cross[0]+2][::-1],zs[cross[0]:cross[0]+2][::-1])) if cross.size else np.nan)
                ax[r,j].plot(lh[order],np.asarray(freeze)[order],color='black',ls='--',lw=1.2,label='0 °C')
                if r == 0 and j == 0: ax[r,j].legend(loc='upper right',fontsize=8)
        profout.append((k,cv,iv,dv))
    out3=OUT/f'{base}_updraft.png'; fig.savefig(out3,dpi=450); plt.close(fig)
    header=f'IOP {i} {e} ({LABEL[e]}); dates={",".join(x.isoformat() for x in dates)}; n_dates={len(dates)}; full native grid={m.shape}; sources=raw FA continuous and IOP; primary delta=IOP-continuous\n'
    (OUT/f'{base}_rainfall_maps.txt').write_text(header+'lon\n'+np.array2string(lon,threshold=np.inf)+'\nlat\n'+np.array2string(lat,threshold=np.inf)+'\ncontinuous_mm_day\n'+np.array2string(c,threshold=np.inf)+'\nIOP_mm_day\n'+np.array2string(m,threshold=np.inf)+'\ndelta_mm_day\n'+np.array2string(delta,threshold=np.inf))
    (OUT/f'{base}_rainfall_diurnal.txt').write_text(header+f'local_hours={lh[order].tolist()}\ncontinuous_mm_h={cd[order].tolist()}\nIOP_mm_h={md[order].tolist()}\ndelta_mm_h={dd[order].tolist()}')
    with (OUT/f'{base}_updraft.txt').open('w') as f:
        f.write(header+f'height_km={z.tolist()}\nlocal_endpoint_hours={lh[order].tolist()}\n')
        for k,cv,iv,dv in profout: f.write(f'## {k}\ncontinuous_{k}\n{np.array2string(cv[order,:],threshold=np.inf)}\nIOP_{k}\n{np.array2string(iv[order,:],threshold=np.inf)}\ndelta_{k}\n{np.array2string(dv[order,:],threshold=np.inf)}\n')
    return out,out2,out3

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--preflight',action='store_true'); ap.add_argument('--run',action='store_true'); ap.add_argument('--watch',action='store_true'); ap.add_argument('--period',type=int,choices=(1,2)); ap.add_argument('--interval',type=int,default=300); args=ap.parse_args(); epygram.init_env()
    while True:
      manifest=preflight()
      if args.run:
       for i in ((args.period,) if args.period else (1,2)):
        a,b=PERIODS[i]; key=f'iop{i}'
        if args.watch:
         # Warm the continuous cache immediately, independent of extraction.
         warm_continuous(key, manifest['periods'][key]["control"]["available"])
        for e in EXPS:
            ir=IOP/key/e/'untar-output'; cr=CONT/e/'untar-output'
            # Every plotted quantity uses the identical date intersection of
            # all three paired IOP experiments and the shared continuous run.
            common=set(manifest['periods'][key]['control']['available'])
            for ee in EXPS: common &= set(manifest['periods'][key][ee]['available'])
            dates=sorted(date.fromisoformat(x) for x in common)
            other=[reduce_day(ir,d) for d in dates]; cont=[reduce_day(cr,d) for d in dates];
            expected=(b-a).days+1
            if len(dates)==expected: plot_pair(i,e,dates,other,cont)
            elif dates: print(f"{key}: holding {len(dates)}/{expected} common dates; no figures", flush=True)
      if not args.watch: return
      print(f"watch: waiting {args.interval}s for complete IOP extraction", flush=True); time.sleep(args.interval)

if __name__=='__main__': main()
