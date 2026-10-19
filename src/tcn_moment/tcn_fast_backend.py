"""Numerically checked execution backend for the frozen E05C TCN protocol."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.io_utils import atomic_torch_save, atomic_write_json, sha256_file
from tcn_moment.tcn_async import train_epoch, validation
from tcn_moment.tcn_resident import make_resident_loader
from tcn_moment.training_utils import (seed_everything, resume_training_checkpoint,
                                      save_training_checkpoint)


def run_case(p, f, fraction, condition, data, deadline):
    directory = pilot.root(p) / pilot.case_name(fraction, condition)
    directory.mkdir(exist_ok=True)
    if (directory / 'result.json').exists():
        return pilot.read(directory / 'result.json')
    seed_everything(torch, p['seed'])
    d, v = data['train'], data['validation']
    indices = pilot.stratified_train_subset_indices(d['y'], d['ids'], fraction, p['seed'])
    ids = d['ids'][indices]
    np.save(directory / 'train_ids.npy', ids)
    generator = torch.Generator().manual_seed(p['seed'])
    train = make_resident_loader(d['x'][indices], d['mask'][indices], d['y'][indices],
                                 p['batch_size'], 0, shuffle=True, generator=generator)
    val = make_resident_loader(v['x'], v['mask'], v['y'], p['evaluation_batch_size'], 0, shuffle=False)
    device = torch.device('cuda')
    model = pilot.build_model(condition).to(device)
    initial_hash = pilot.state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=3)
    history, selected_f1, stale, start = [], -1., 0, 1
    checkpoint = directory / 'checkpoint_latest.pt'
    metadata = {'protocol_sha256':f['protocol_sha256'], 'condition':condition, 'fraction':fraction,
                'train_ids_sha256':pilot.train_subset_sha256(ids)}
    if checkpoint.exists():
        start, history, selected_f1, stale = resume_training_checkpoint(
            torch=torch, path=checkpoint, model=model, optimizer=optimizer, data_generator=generator,
            device=torch.device('cpu'), scheduler=scheduler, expected_metadata=metadata)
    peak_f1 = max((row['macro_f1'] for row in history), default=-1.)
    revision = {'backend':'resident_async_v1', 'resume_start_epoch':start,
                'previous_committed_epochs':len(history), 'recorded_at':pilot.now(),
                'training_scope_changed':False, 'precision':'fp32_tf32_disabled',
                'source_sha256':{name:sha256_file(Path('src/tcn_moment')/name)
                                 for name in ('tcn_resident.py','tcn_async.py','tcn_fast_backend.py')}}
    if (directory/'execution_backend.json').exists():
        previous = pilot.read(directory/'execution_backend.json')
        revision['previous_invocations'] = previous.get('previous_invocations',[])+[
            {k:v for k,v in previous.items() if k!='previous_invocations'}]
    atomic_write_json(directory/'execution_backend.json',revision)
    for epoch in range(start,p['epochs']+1):
        if time.monotonic() >= deadline:
            raise TimeoutError('Frozen suite wall-time budget reached; partial results retained')
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        lr = float(optimizer.param_groups[0]['lr'])
        train_loss = train_epoch(model,train,optimizer,device,p['gradient_clip_norm'])
        metrics, probabilities = validation(model,val,device)
        scheduler.step(metrics['macro_f1'])
        metrics.update(epoch=epoch,train_loss=train_loss,learning_rate_used=lr,
                       learning_rate_next=float(optimizer.param_groups[0]['lr']),
                       optimizer_steps=len(train),epoch_seconds=time.monotonic()-started,
                       peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/1024**2)
        history.append(metrics)
        if metrics['macro_f1'] > peak_f1:
            peak_f1 = metrics['macro_f1']
            atomic_torch_save(torch,model.state_dict(),directory/'model_peak.pt')
            pilot.save_predictions(directory/'validation_peak.csv',v['ids'],v['y'],probabilities)
        if metrics['macro_f1'] > selected_f1 + p['early_stopping_min_delta']:
            selected_f1, stale = metrics['macro_f1'], 0
            atomic_torch_save(torch,model.state_dict(),directory/'model_selected.pt')
            pilot.save_predictions(directory/'validation_selected.csv',v['ids'],v['y'],probabilities)
        else:
            stale += 1
        save_training_checkpoint(torch=torch,path=checkpoint,epoch=epoch,model=model,optimizer=optimizer,
            history=history,best_macro_f1=selected_f1,epochs_without_improvement=stale,
            data_generator=generator,scheduler=scheduler,metadata=metadata)
        atomic_write_json(directory/'history.json',history)
        atomic_write_json(directory/'status.json',{'status':'running','epoch':epoch,'updated_at':pilot.now(),
                                                 'test_evaluated':False,'execution_backend':'resident_async_v1'})
        print(f"{pilot.case_name(fraction,condition)} epoch={epoch} f1={100*metrics['macro_f1']:.4f} seconds={metrics['epoch_seconds']:.1f} backend=resident_async",flush=True)
    metrics, probabilities = validation(model,val,device)
    pilot.save_predictions(directory/'validation_endpoint.csv',v['ids'],v['y'],probabilities)
    peak_epoch = max(history,key=lambda row:row['macro_f1'])['epoch']
    selected_epoch = next(row['epoch'] for row in history if row['macro_f1']==selected_f1)
    result = {**metadata,'test_evaluated':False,'train_count':len(ids),'validation_count':len(v['ids']),
              'initial_model_sha256':initial_hash,'total_parameters':sum(t.numel() for t in model.parameters()),
              'epochs':len(history),'optimizer_steps':sum(row['optimizer_steps'] for row in history),
              'selected_epoch':selected_epoch,'selected_macro_f1':selected_f1,
              'peak_epoch':peak_epoch,'peak_macro_f1':peak_f1,'endpoint_metrics':metrics,
              'training_seconds':sum(row['epoch_seconds'] for row in history),
              'execution_backend':'resident_async_v1',
              'output_sha256':{q.name:sha256_file(q) for q in sorted(directory.iterdir())
                              if q.suffix in {'.pt','.csv','.npy'} or q.name=='history.json'}}
    if result['optimizer_steps'] != len(history)*math.ceil(len(ids)/p['batch_size']):
        raise ValueError('Training update budget mismatch')
    atomic_write_json(directory/'result.json',result)
    atomic_write_json(directory/'status.json',{'status':'completed','updated_at':pilot.now(),'test_evaluated':False,
                                             'execution_backend':'resident_async_v1'})
    del model, optimizer, scheduler, train, val
    torch.cuda.empty_cache()
    return result
