"""V2 image trainer: train-only augmentation, minimum optimization budget, exact batch resume."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from src.models.backbones.amuse_x_hybrid3d import AMUSEXHybrid3DEncoder
from scripts.xyz_fusion_v1.loss import discrete_time_survival_nll, survival_from_logits
from scripts.xyz_fusion_v1.store import atomic_json,digest,file_sha
from scripts.xyz_fusion_v1.runtime import available_ram
from scripts.xyz_fusion_v1.statistics import ibs_score


def augment(image,policy):
    if policy.get('modality','mri')!='mri' or policy.get('test_disable_augmentation',False):
        return image
    # Small integer translations preserve signal amplitudes. No reflection,
    # test-time augmentation, PET intensity transform, or synthetic subjects.
    shifts=torch.randint(-2,3,(3,),device='cpu').tolist()
    result=torch.roll(image,tuple(shifts),(2,3,4))
    for dim,shift in zip((2,3,4),shifts):
        if shift:
            sl=[slice(None)]*5;sl[dim]=slice(0,shift) if shift>0 else slice(shift,None)
            result[tuple(sl)]=0
    mask=result!=0
    return result+torch.randn_like(result)*.01*mask


class MRISurvival(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.encoder=AMUSEXHybrid3DEncoder(out_dim=policy['latent_dim'],hybrid_variant=policy['hybrid_variant'])
        self.head=nn.Linear(policy['latent_dim'],len(policy['hazard_edges_days'])-1)

    def forward(self,image):
        return self.head(self.encoder(image))


def deterministic(seed,threads):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False


def save_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


class ImageBank:
    def __init__(self,path,max_gib):
        mapped=np.load(path,mmap_mode='r',allow_pickle=False)
        self.mode='MEMMAP'
        if mapped.nbytes < min(max_gib*1024**3,available_ram()-6*1024**3):
            self.array=np.array(mapped); self.mode='RAM_ONCE'
        else:
            self.array=mapped

    def batch(self,rows,device):
        value=torch.from_numpy(np.array(self.array[np.asarray(rows,dtype=int)],copy=True))
        if device.type=='cuda':
            value=value.pin_memory()
        return value.to(device,non_blocking=device.type=='cuda')


@torch.no_grad()
def predict(model,bank,frame,policy,device,grid):
    model.eval(); result=[]
    for start in range(0,len(frame),policy['microbatch_size']):
        rows=frame.cache_row.iloc[start:start+policy['microbatch_size']].to_numpy()
        with torch.autocast(device.type,enabled=policy['amp'] and device.type=='cuda'):
            logits=model(bank.batch(rows,device))
        result.append(survival_from_logits(logits.float()).cpu().numpy())
    boundary=np.concatenate(result)
    edges=np.asarray(policy['hazard_edges_days'])
    return np.array([np.interp(grid,edges,np.r_[1.,row]) for row in boundary])


def train_expert(bank,train,val,policy,grid,out,binding,lr,seed,*,fixed_epochs=None,interrupt_after_steps=None):
    """Only train/inner-validation enter this function. Outer rows are not accepted."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if val is not None and set(train.patient_hash)&set(val.patient_hash):
        raise ValueError('MRI_INNER_SUBJECT_OVERLAP')
    if not len(train) or train.patient_hash.duplicated().any():
        raise ValueError('MRI_TRAIN_MEMBERSHIP_INVALID')
    local=digest({'parent':binding,'train':sorted(train.patient_hash),'val':[] if val is None else sorted(val.patient_hash),
                  'policy':policy,'lr':lr,'seed':seed,'fixed_epochs':fixed_epochs})
    path=out/'progress.pt'; locked=out/'locked.pt'; receipt=out/'locked.json'
    deterministic(seed,policy['cpu_threads'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MRISurvival(policy).to(device)
    # Train-only marginal hazard initialization; no validation/test outcomes.
    edges_np=np.asarray(policy['hazard_edges_days'],float)
    init=[]
    for lo,hi in zip(edges_np[:-1],edges_np[1:]):
        risk=int((train.time_days>lo).sum())
        events=int(((train.time_days>lo)&(train.time_days<=hi)&train.event.eq(1)).sum())
        init.append(np.clip((events+.5)/(risk+1),1e-4,1-1e-4))
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor(np.log(np.asarray(init)/(1-np.asarray(init))),device=device,dtype=torch.float32))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=policy['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=policy['amp'] and device.type=='cuda')
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['binding']!=local or r['sha256']!=file_sha(locked):
            raise ValueError('MRI_LOCKED_EXPERT_MISMATCH')
        cp=torch.load(locked,map_location='cpu',weights_only=False)
        model.load_state_dict(cp['model'],strict=True)
        return model,r
    cp={'binding':local,'epoch':0,'cursor':0,'order':None,'steps':0,'patience':0,
        'best':float('inf'),'best_epoch':0,'best_state':None,'history':[],
        'loss_sum':0.,'seen':0,'terminal':False}
    if path.exists():
        cp=torch.load(path,map_location='cpu',weights_only=False)
        if cp['binding']!=local:
            raise ValueError('MRI_TRAIN_CONTRACT_MISMATCH')
        model.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler'])
        # Optimizer state belongs to its parameter's device, RNG state always to CPU.
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v) and k!='step': state[k]=v.to(device)
        random.setstate(cp['python_rng']);np.random.set_state(cp['numpy_rng']);torch.set_rng_state(cp['torch_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']])
        print(json.dumps({'event':'MRI_TRAIN_RESUMED','task':out.name,'epoch':cp['epoch']+1,
                          'optimizer_steps_completed':cp['steps'],'samples_completed_in_epoch':cp['cursor']}),flush=True)
    cp.setdefault('amp_overflow_retries',0);cp.setdefault('amp_retry_current_batch',0)
    def commit():
        cp.update(model=cpu_state(model),optimizer=copy.deepcopy(opt.state_dict()),scaler=scaler.state_dict(),
                  python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                  numerical_runtime='AMP_SAME_BATCH_REPLAY_V1')
        save_torch(path,cp)
    max_epochs=fixed_epochs or policy['epochs']; batch=policy['batch_size']; micro=policy['microbatch_size']
    edges=torch.tensor(policy['hazard_edges_days'],device=device,dtype=torch.float64)
    while cp['epoch']<max_epochs and not cp['terminal']:
        start_time=time.perf_counter(); model.train()
        if cp['order'] is None:
            cp['order']=np.random.permutation(len(train));cp['cursor']=0
        while cp['cursor']<len(train):
            if cp['amp_retry_current_batch']>=8:raise ValueError('MRI_AMP_NONFINITE_RETRY_LIMIT')
            ids=cp['order'][cp['cursor']:cp['cursor']+batch];opt.zero_grad(set_to_none=True)
            # Replay the same effective batch after scale overflow: restore RNG
            # and running buffers, but retain GradScaler's reduced scale. Do not
            # advance the subject cursor, loss denominator or optimizer count.
            replay=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
                    torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                    {k:v.detach().clone() for k,v in model.named_buffers()})
            batch_loss_sum=0.
            for offset in range(0,len(ids),micro):
                part=ids[offset:offset+micro]; rows=train.iloc[part]
                with torch.autocast(device.type,enabled=policy['amp'] and device.type=='cuda'):
                    logits=model(augment(bank.batch(rows.cache_row.to_numpy(),device),policy))
                loss=discrete_time_survival_nll(logits.float(),torch.tensor(rows.event.to_numpy(),device=device),
                                               torch.tensor(rows.time_days.to_numpy(),device=device),edges)
                if not torch.isfinite(loss): raise ValueError('MRI_LOSS_NONFINITE')
                scaler.scale(loss*len(part)/len(ids)).backward()
                batch_loss_sum+=float(loss.detach())*len(part)
            scaler.unscale_(opt)
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            if not gradients:raise ValueError('MRI_NO_PARAMETER_GRADIENTS')
            finite=bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
            if not finite:
                if not scaler.is_enabled():raise ValueError('MRI_GRADIENT_NONFINITE_WITHOUT_AMP')
                previous_scale=scaler.get_scale()
                scaler.step(opt)  # GradScaler skips the update when unscale found inf/NaN.
                scaler.update()
                reduced_scale=scaler.get_scale()
                if not (np.isfinite(reduced_scale) and 0<reduced_scale<previous_scale):
                    raise ValueError('MRI_AMP_SCALE_DID_NOT_BACK_OFF')
                random.setstate(replay[0]);np.random.set_state(replay[1]);torch.set_rng_state(replay[2].cpu())
                if device.type=='cuda':torch.cuda.set_rng_state_all([r.cpu() for r in replay[3]])
                with torch.no_grad():
                    for name,value in model.named_buffers():value.copy_(replay[4][name])
                cp['amp_overflow_retries']+=1;cp['amp_retry_current_batch']+=1
                commit()  # Persist the lower scale at the SAME incomplete batch.
                print(json.dumps({'event':'AMP_GRADIENT_OVERFLOW_RETRY','task':out.name,'epoch':cp['epoch']+1,
                                  'optimizer_step_pending':cp['steps']+1,'retry':cp['amp_retry_current_batch'],
                                  'scale_before':previous_scale,'scale_after':reduced_scale,
                                  'batch_dropped':False}),flush=True)
                continue
            nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            scaler.step(opt);scaler.update()
            cp['loss_sum']+=batch_loss_sum;cp['seen']+=len(ids)
            cp['cursor']+=len(ids);cp['steps']+=1;cp['amp_retry_current_batch']=0
            if cp['steps']%policy['checkpoint_every_optimizer_steps']==0: commit()
            if interrupt_after_steps is not None and cp['steps']>=interrupt_after_steps:
                commit();return None,{'status':'TEST_INTERRUPTION','steps':cp['steps']}
        score=None
        if val is not None:
            score=ibs_score(train,val,predict(model,bank,val,policy,device,grid),grid)
            if cp['epoch']+1>=policy['minimum_epochs'] and score<cp['best']-1e-8:
                cp['best']=score;cp['best_epoch']=cp['epoch']+1;cp['best_state']=cpu_state(model);cp['patience']=0
            elif cp['epoch']+1>=policy['minimum_epochs']: cp['patience']+=1
        else:
            cp['best_epoch']=cp['epoch']+1;cp['best_state']=cpu_state(model)
        cp['history'].append({'epoch':cp['epoch']+1,'train_nll':cp['loss_sum']/cp['seen'],'inner_ipcw_ibs':score})
        cp['epoch']+=1;cp['cursor']=0;cp['order']=None;cp['loss_sum']=0.;cp['seen']=0
        cp['terminal']=cp['epoch']>=max_epochs or (val is not None and cp['epoch']>=policy['minimum_epochs'] and cp['patience']>=policy['patience'])
        commit()
        elapsed=time.perf_counter()-start_time
        print(json.dumps({'event':'MRI_EPOCH_COMMITTED','task':out.name,'epoch':cp['epoch'],'max_epochs':max_epochs,
                          'train_nll':cp['history'][-1]['train_nll'],'inner_ibs':score,'seconds':elapsed,
                          'task_eta_upper_seconds':elapsed*(max_epochs-cp['epoch']),
                          'checkpoint':str(path),'resume_step':cp['steps'],'terminal':cp['terminal']}),flush=True)
    if cp['best_state'] is None: raise ValueError('MRI_NO_VALID_LOCKABLE_MODEL')
    model.load_state_dict(cp['best_state'],strict=True)
    save_torch(locked,{'binding':local,'model':cpu_state(model),'best_epoch':cp['best_epoch'],
                       'ancestry_subjects':sorted(train.patient_hash),'selection_subjects':[] if val is None else sorted(val.patient_hash)})
    r={'binding':local,'sha256':file_sha(locked),'best_epoch':cp['best_epoch'],
       'inner_ibs':None if val is None else cp['best'],'history':cp['history'],'terminal':True,
       'parent_contract':binding,'train_subjects':sorted(train.patient_hash),
       'selection_subjects':[] if val is None else sorted(val.patient_hash)}
    atomic_json(receipt,r)
    return model,r


