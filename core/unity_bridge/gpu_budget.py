"""Cooperative, process-wide GPU budget shared by every trainer in this checkout.

Windows/WDDM has no hard utilization-percent quota. This governor combines a
global CUDA work/cooldown schedule with feedback from whole-device NVML usage.
It is a measured utilization target, NOT an instantaneous hardware ceiling.
"""
import ctypes
import os
from pathlib import Path
import random
import sqlite3
import time
from contextlib import contextmanager, nullcontext
import psutil

TARGET=.65
DATABASE=Path(__file__).resolve().parents[1]/'local-results'/'resource-control'/'gpu-budget.sqlite'
_budget=None

class TrainingStopped(Exception):pass

class Nvml:
    def __init__(self,index=0):
        self.lib=ctypes.WinDLL('nvml.dll') if os.name=='nt' else ctypes.CDLL('libnvidia-ml.so.1')
        self.check(self.lib.nvmlInit_v2())
        self.handle=ctypes.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index),ctypes.byref(self.handle)))
    @staticmethod
    def check(result):
        if result:raise RuntimeError(f'NVML GPU monitoring failed (error {result}); training has not been allowed to run ungoverned')
    def utilization(self):
        class Rates(ctypes.Structure):_fields_=[('gpu',ctypes.c_uint),('memory',ctypes.c_uint)]
        rates=Rates();self.check(self.lib.nvmlDeviceGetUtilizationRates(self.handle,ctypes.byref(rates)))
        return rates.gpu

class GpuBudget:
    def __init__(self,device,stop_path=None,*,database=DATABASE,monitor=None):
        import torch
        self.device=device;self.stop_path=Path(stop_path) if stop_path else None
        self.pid=os.getpid();self.created=psutil.Process().create_time();self.wait_seconds=0.;self.work_seconds=0.;self.depth=0
        self.monitor=monitor or Nvml(torch.device(device).index or 0)
        database=Path(database);database.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(database,timeout=10,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, owner INTEGER, created REAL, ready REAL, sampled REAL, utilization REAL)')
        self.db.execute('INSERT OR IGNORE INTO budget VALUES (1,0,0,0,0,0)')
        self.db.execute('CREATE TABLE IF NOT EXISTS waiters (ticket INTEGER PRIMARY KEY AUTOINCREMENT, pid INTEGER, created REAL)')
    @staticmethod
    def alive(pid,created):
        try:return psutil.Process(pid).create_time()==created
        except psutil.Error:return False
    def acquire(self):
        began=time.monotonic();ticket=None
        try:
            while True:
                if self.stop_path and self.stop_path.exists():raise TrainingStopped()
                self.db.execute('BEGIN IMMEDIATE')
                try:
                    now=time.time()
                    if ticket is None:ticket=self.db.execute('INSERT INTO waiters(pid,created) VALUES (?,?)',(self.pid,self.created)).lastrowid
                    owner,created,ready,sampled,util=self.db.execute('SELECT owner,created,ready,sampled,utilization FROM budget WHERE id=1').fetchone()
                    if owner and not self.alive(owner,created):owner=0;self.db.execute('UPDATE budget SET owner=0 WHERE id=1')
                    for entry,pid,birth in self.db.execute('SELECT ticket,pid,created FROM waiters').fetchall():
                        if not self.alive(pid,birth):self.db.execute('DELETE FROM waiters WHERE ticket=?',(entry,))
                    if now-sampled>=.2:
                        util=self.monitor.utilization();self.db.execute('UPDATE budget SET sampled=?,utilization=? WHERE id=1',(now,util))
                    first=self.db.execute('SELECT MIN(ticket) FROM waiters').fetchone()[0]
                    allowed=not owner and now>=ready and util<100*TARGET and ticket==first
                    if allowed:self.db.execute('UPDATE budget SET owner=?,created=? WHERE id=1',(self.pid,self.created))
                    self.db.execute('COMMIT')
                except BaseException:
                    self.db.execute('ROLLBACK');raise
                if allowed:
                    self.wait_seconds+=time.monotonic()-began;return ticket
                time.sleep(.004+random.random()*.004)
        except BaseException:
            if ticket:self.db.execute('DELETE FROM waiters WHERE ticket=?',(ticket,))
            raise
    @contextmanager
    def work(self):
        import torch
        if self.depth:
            self.depth+=1
            try:yield
            finally:self.depth-=1
            return
        ticket=self.acquire();began=time.monotonic()
        self.depth=1
        try:
            yield
        finally:
            try:torch.cuda.synchronize(self.device)
            finally:
                self.depth=0;elapsed=time.monotonic()-began;self.work_seconds+=elapsed
                # Aim below 65% to leave room for sampling latency and the desktop.
                duty=TARGET-.05;cooldown=elapsed*(1-duty)/duty
                self.db.execute('BEGIN IMMEDIATE')
                try:
                    self.db.execute('UPDATE budget SET owner=0,ready=? WHERE id=1',(time.time()+cooldown,))
                    self.db.execute('DELETE FROM waiters WHERE ticket=?',(ticket,));self.db.execute('COMMIT')
                except BaseException:self.db.execute('ROLLBACK');raise
    def status(self):
        utilization=self.db.execute('SELECT utilization FROM budget WHERE id=1').fetchone()[0]
        return dict(targetPercent=100*TARGET,measuredGpuPercent=utilization,waitSeconds=self.wait_seconds,
                    workSeconds=self.work_seconds,mode='shared cooperative duty budget + whole-device NVML feedback',hardLimit=False)

def configure(device,stop_path=None):
    global _budget
    import torch
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA 학습을 사용할 수 없습니다. CUDA 지원 PyTorch와 NVIDIA 드라이버를 확인하세요. CPU로 자동 전환하지 않습니다.')
        torch.cuda.set_device(device);_budget=GpuBudget(device,stop_path)
    else:_budget=None
    os.environ['ANS_TORCH_DEVICE']=str(device)
    return _budget

def gpu_work():return _budget.work() if _budget else nullcontext()

def budget_status():return _budget.status() if _budget else None
