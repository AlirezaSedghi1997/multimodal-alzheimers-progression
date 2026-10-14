"""Bounded resource policy. No writes outside the authoritative project."""
from __future__ import annotations
import ctypes
import os
from pathlib import Path
import subprocess
import sys


def configure(root, threads=4):
    root = Path(root).resolve()
    cache = root/'.runtime/xyz_fusion'
    for key, child in {'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','MPLCONFIGDIR':'mpl',
                       'TORCH_HOME':'torch','TORCHINDUCTOR_CACHE_DIR':'inductor',
                       'CUDA_CACHE_PATH':'cuda','NUMBA_CACHE_DIR':'numba',
                       'JOBLIB_TEMP_FOLDER':'joblib'}.items():
        p = cache/child; p.mkdir(parents=True, exist_ok=True); os.environ[key] = str(p)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'; sys.dont_write_bytecode = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
        os.environ[key] = str(threads)


def available_ram():
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [
                (x,ctypes.c_ulonglong) for x in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
        m=Memory(); m.length=ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            raise OSError('MEMORY_QUERY_FAILED')
        return m.available
    return os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')


def peak_rss():
    if os.name=='nt':
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
                ('peak','working','paged_peak','paged','nonpaged_peak','nonpaged','pagefile','pagefile_peak')]
        c=Counters();c.cb=ctypes.sizeof(c)
        kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=ctypes.c_void_p
        fn=ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
        if not fn(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):raise OSError('RSS_QUERY_FAILED')
        return int(c.peak)
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def gpu_guard():
    import csv
    r = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],
                       capture_output=True,text=True,check=True)
    active=[]
    for row in csv.reader(r.stdout.splitlines()):
        if len(row)!=2 or not row[0].strip().isdigit():
            if any(x.strip() for x in row):raise RuntimeError('GPU_PROCESS_QUERY_UNPARSEABLE')
            continue
        pid=int(row[0]);name=row[1].strip().replace('\\','/').lower()
        # Windows WDDM reports this installed desktop UI as a compute app (10 MiB
        # in the observed preflight). Do not mistake it for a research worker.
        codex_ui=('/windowsapps/openai.codex_' in name and name.endswith('/app/chatgpt.exe'))
        if pid!=os.getpid() and not codex_ui:active.append(pid)
    if active:
        raise RuntimeError('FOREIGN_GPU_WRITER_ACTIVE_DO_NOT_STOP_IT')
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if not r.stdout.splitlines() or float(r.stdout.splitlines()[0].strip())<6000:
        raise RuntimeError('GPU_FREE_MEMORY_BELOW_6000_MIB')


def cpu_workers(policy):
    by_ram = max(1, int((available_ram()-4*1024**3)//(policy['ram_gib_per_worker']*1024**3)))
    by_cpu = max(1, (os.cpu_count() or 2)//policy['itk_threads_per_worker'])
    return max(1, min(policy['workers_max'], by_ram, by_cpu))
