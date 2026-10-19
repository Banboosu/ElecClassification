"""Synthetic numerical equivalence and speed benchmark for resident TCN data."""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from tcn_moment import tcn_async
from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.tcn_resident import make_resident_loader
from tcn_moment.train_tcn import make_loader
from tcn_moment.training_utils import seed_everything, resume_training_checkpoint


def run(condition, loader_factory, x, mask, y, epochs, *, asynchronous=False, checkpoint=None):
    pilot.configure()
    seed_everything(torch, 1234)
    generator = torch.Generator().manual_seed(1234)
    torch.cuda.synchronize()
    preload_started = time.perf_counter()
    train = loader_factory(x, mask, y, 32, 0, shuffle=True, generator=generator)
    val = loader_factory(x[:259], mask[:259], y[:259], 64, 0, shuffle=False)
    model = pilot.build_model(condition).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=3)
    if checkpoint is not None:
        resume_training_checkpoint(torch=torch,path=checkpoint,model=model,optimizer=optimizer,
                                   data_generator=generator,device=torch.device('cpu'),scheduler=scheduler)
    torch.cuda.synchronize()
    setup = time.perf_counter() - preload_started
    history, timings, logits = [], [], []
    for epoch in range(epochs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if asynchronous:
            train_loss = tcn_async.train_epoch(model, train, optimizer, torch.device('cuda'))
            metrics, probability = tcn_async.validation(model, val, torch.device('cuda'))
        else:
            model.train()
            total = 0.
            for batch_x, batch_mask, batch_y in train:
                batch_x, batch_mask, batch_y = batch_x.cuda(), batch_mask.cuda(), batch_y.cuda()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch_x, batch_mask)
                loss = torch.nn.functional.cross_entropy(prediction, batch_y)
                if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
                    raise FloatingPointError('Synthetic loss/logits nonfinite')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Synthetic gradients nonfinite')
                optimizer.step()
                total += float(loss.detach()) * len(batch_y)
            train_loss = total/len(y)
            metrics, probability = pilot.validation(model, val, torch.device('cuda'))
        scheduler.step(metrics['macro_f1'])
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        history.append({'train_loss': train_loss, 'validation': metrics})
        logits.append(probability)
    payload = {'model_hash':pilot.state_hash(model), 'optimizer':copy.deepcopy(optimizer.state_dict()),
               'history':history, 'probability':logits, 'shuffle_rng':generator.get_state().clone(),
               'cpu_rng':torch.get_rng_state().clone(), 'cuda_rng':torch.cuda.get_rng_state().clone(),
               'seconds':timings, 'setup_seconds':setup}
    del optimizer, model, scheduler, train, val
    torch.cuda.empty_cache()
    return payload


def compare(a,b):
    checks = {'model_weights_equal':a['model_hash']==b['model_hash'],
              'all_epoch_metrics_equal':a['history']==b['history'],
              'all_validation_probabilities_equal':all(np.array_equal(x,y) for x,y in zip(a['probability'],b['probability'],strict=True))}
    for key in ('shuffle_rng','cpu_rng','cuda_rng'):
        checks[key+'_equal']=torch.equal(a[key],b[key])
    def equal(x,y):
        if isinstance(x,torch.Tensor):return torch.equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,list):return len(x)==len(y) and all(equal(xx,yy) for xx,yy in zip(x,y))
        return x==y
    checks['optimizer_state_equal']=equal(a['optimizer'],b['optimizer'])
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows',type=int,default=4099)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--asynchronous', action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(20261019)
    x=rng.normal(3,1,(args.rows,1024)).astype(np.float32)
    lengths=rng.integers(18,817,args.rows)
    mask=(np.arange(1024)[None,:]<lengths[:,None]).astype(np.float32)
    x*=mask
    y=np.arange(args.rows,dtype=np.int64)%3
    rows=[]
    for condition in ('current_fixed50','legacy_masked50','legacy_unmasked50'):
        baseline=run(condition,make_loader,x,mask,y,args.epochs)
        resident=run(condition,make_resident_loader,x,mask,y,args.epochs,asynchronous=args.asynchronous)
        checks=compare(baseline,resident)
        row={'condition':condition,'checks':checks,'baseline_seconds':baseline['seconds'],
             'resident_seconds':resident['seconds'], 'baseline_setup_seconds':baseline['setup_seconds'],
             'resident_setup_seconds':resident['setup_seconds'],
             'speedup_including_first_epoch':sum(baseline['seconds'])/sum(resident['seconds']),
             'speedup_after_first_epoch':sum(baseline['seconds'][1:])/sum(resident['seconds'][1:])}
        rows.append(row)
        atomic_write_json(args.output,{'status':'passed' if all(all(r['checks'].values()) for r in rows) else 'failed',
            'synthetic_only':True,'asynchronous_checks':args.asynchronous,'project_samples_used':0,'rows':args.rows,'epochs':args.epochs,
            'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(), 'conditions':rows})
        print(row,flush=True)
        if not all(checks.values()):
            raise AssertionError('Resident numerical equivalence failed; do not apply')


if __name__=='__main__':main()
