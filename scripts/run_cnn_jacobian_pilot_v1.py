"""Versioned pilot only; no automatic launch of a full-cohort experiment."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','fit','evaluate','all'],default='prepare');p.add_argument('--config',type=Path,default=ROOT/'configs/cnn_jacobian_pilot_v1_2.json');a=p.parse_args();cfg=json.loads(a.config.read_text())
 out=(ROOT/cfg['output_root']).resolve()
 if not out.is_relative_to(ROOT/'results') or not out.name.startswith('cnn_jacobian_pilot_'):raise ValueError('ISOLATED_PILOT_ROOT_REQUIRED')
 for field in ['parent_root','cache_root']:
  parent=(ROOT/cfg[field]).resolve()
  if out==parent or out.is_relative_to(parent) or parent.is_relative_to(out):raise ValueError('PARENT_OUTPUT_OVERLAP')
 if cfg['full_dataset_run_automatic'] or cfg['image_size']!=64 or cfg['fit_n']+cfg['selection_n']+cfg['audit_n']!=cfg['pilot_n']:raise ValueError('UNSUPPORTED_PROTOCOL_CHANGE')
 runtime=ROOT/'.runtime/cnn_jacobian_pilot_v1';runtime.mkdir(parents=True,exist_ok=True)
 for k in ['TMP','TEMP','TMPDIR']:os.environ[k]=str(runtime)
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']:os.environ[k]=str(cfg['cpu_threads'])
 from scripts.xyz_fusion_v1.store import writer_lock
 from scripts.cnn_jacobian_pilot_v1.prepare import prepare
 from scripts.cnn_jacobian_pilot_v1.training import fit_all,evaluate
 with writer_lock(out):
  if a.mode in ['prepare','all']:prepare(ROOT,cfg)
  if a.mode in ['fit','all']:fit_all(ROOT,cfg)
  if a.mode in ['evaluate','all']:evaluate(ROOT,cfg)
if __name__=='__main__':main()
