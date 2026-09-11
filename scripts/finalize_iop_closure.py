"""Wait for complete DDHB extraction, aggregate exact periods, and plot."""
import time, subprocess
from pathlib import Path
from datetime import date, timedelta

ROOT=Path('/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/iop-runs/analysis/ddh')
PER={1:(date(2014,1,15),date(2014,2,28)),2:(date(2014,7,29),date(2014,8,31))}
EX=('graupel','2mom','NO3M')
REQ=('QV.DHFDLABOF+0024.condcv.dta','QV.DHFDLABOF+0024.condrs.dta','QL.DHFDLABOF+0024.VQLM.dta','QR.DHFDLABOF+0024.VQRM.dta')

def ready(i,e):
    a,b=PER[i]; d=a; n=0
    while d<=b:
        p=ROOT/f'iop{i}'/e/f'DDH{d:%Y%m%d}'
        n += all((p/x).is_file() for x in REQ); d += timedelta(days=1)
    return n == (b-a).days+1

def main():
    while not all(ready(i,e) for i in (1,2) for e in EX):
        print('waiting for complete DDHB periods',flush=True); time.sleep(30)
    py='/mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/envs/epygram/bin/python'; s='scripts/make_iop_closure_budget.py'
    for i in (1,2): subprocess.run([py,s,'--aggregate','--iop',str(i)],check=True)
    subprocess.run([py,s],check=True)

if __name__=='__main__': main()
