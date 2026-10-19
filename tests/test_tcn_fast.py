"""Exact batching/RNG, checkpoint continuation and fail-stop checks for fast TCN."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tcn_moment import tcn_async
from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.tcn_resident import ResidentLoader
from tcn_moment.train_tcn import make_loader
from tcn_moment.training_utils import seed_everything, save_training_checkpoint, resume_training_checkpoint


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(8,12),nn.ReLU(),nn.Dropout(.3),nn.Linear(12,3))
    def forward(self,x,mask):return self.net(x[:,0,:]*mask)


def reference_train(model,loader,optimizer):
    model.train();total=0.
    for x,mask,y in loader:
        optimizer.zero_grad(set_to_none=True)
        logits=model(x,mask);loss=nn.functional.cross_entropy(logits,y)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():raise FloatingPointError()
        loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError()
        optimizer.step();total+=float(loss.detach())*len(y)
    return total/len(loader.dataset)


class FastTCNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        rng=np.random.default_rng(42)
        self.x=rng.normal(size=(65,8)).astype(np.float32)
        self.mask=np.ones_like(self.x)
        self.y=np.arange(65,dtype=np.int64)%3

    def test_batches_tail_and_all_rng_match_multiple_epochs(self):
        for shuffle in (False,True):
            outputs=[]
            for resident in (False,True):
                seed_everything(torch,42)
                generator=torch.Generator().manual_seed(42) if shuffle else None
                loader=(ResidentLoader(self.x,self.mask,self.y,32,shuffle=shuffle,generator=generator,device='cpu')
                        if resident else make_loader(self.x,self.mask,self.y,32,0,shuffle=shuffle,generator=generator))
                batches=[]
                for _ in range(3):batches.extend(tuple(t.clone() for t in batch) for batch in loader)
                outputs.append((batches,torch.get_rng_state(),generator.get_state() if generator else None))
            for a,b in zip(outputs[0][0],outputs[1][0],strict=True):
                self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
            self.assertEqual([len(batch[2]) for batch in outputs[0][0]],[32,32,1]*3)
            self.assertTrue(torch.equal(outputs[0][1],outputs[1][1]))
            if shuffle:self.assertTrue(torch.equal(outputs[0][2],outputs[1][2]))

    def test_optimized_epoch_matches_reference(self):
        results=[]
        for fast in (False,True):
            seed_everything(torch,42);gen=torch.Generator().manual_seed(42)
            loader=(ResidentLoader(self.x,self.mask,self.y,32,shuffle=True,generator=gen,device='cpu')
                    if fast else make_loader(self.x,self.mask,self.y,32,0,shuffle=True,generator=gen))
            model=Toy();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            losses=[]
            for _ in range(3):
                losses.append(tcn_async.train_epoch(model,loader,optimizer,torch.device('cpu')) if fast else reference_train(model,loader,optimizer))
            val=(ResidentLoader(self.x,self.mask,self.y,64,shuffle=False,device='cpu')
                 if fast else make_loader(self.x,self.mask,self.y,64,0,shuffle=False))
            metrics,prob=(tcn_async.validation(model,val,torch.device('cpu')) if fast else pilot.validation(model,val,torch.device('cpu')))
            results.append((pilot.state_hash(model),losses,metrics,prob,torch.get_rng_state(),gen.get_state()))
        self.assertEqual(results[0][:3],results[1][:3])
        np.testing.assert_array_equal(results[0][3],results[1][3])
        self.assertTrue(torch.equal(results[0][4],results[1][4]))
        self.assertTrue(torch.equal(results[0][5],results[1][5]))

    def test_reference_checkpoint_resumes_exactly_on_fast_backend(self):
        seed_everything(torch,42);gen=torch.Generator().manual_seed(42)
        loader=make_loader(self.x,self.mask,self.y,32,0,shuffle=True,generator=gen)
        model=Toy();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        reference_train(model,loader,optimizer)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt'
            save_training_checkpoint(torch=torch,path=path,epoch=1,model=model,optimizer=optimizer,
                history=[],best_macro_f1=.5,epochs_without_improvement=0,data_generator=gen)
            reference_loss=reference_train(model,loader,optimizer)
            expected=pilot.state_hash(model)
            restored=Toy();opt=torch.optim.AdamW(restored.parameters(),lr=.001)
            g=torch.Generator().manual_seed(123)
            fast=ResidentLoader(self.x,self.mask,self.y,32,shuffle=True,generator=g,device='cpu')
            start,_,_,_=resume_training_checkpoint(torch=torch,path=path,model=restored,optimizer=opt,
                                                   data_generator=g,device=torch.device('cpu'))
            self.assertEqual(start,2)
            loss=tcn_async.train_epoch(restored,fast,opt,torch.device('cpu'))
            self.assertEqual(loss,reference_loss)
            self.assertEqual(pilot.state_hash(restored),expected)
            self.assertTrue(torch.equal(g.get_state(),gen.get_state()))

    def test_nonfinite_inputs_abort_before_update_on_cpu(self):
        model=Toy();before=pilot.state_hash(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        x=self.x.copy();x[0,0]=np.nan
        loader=ResidentLoader(x,self.mask,self.y,32,shuffle=False,device='cpu')
        with self.assertRaises(RuntimeError):tcn_async.train_epoch(model,loader,optimizer,torch.device('cpu'))
        self.assertEqual(pilot.state_hash(model),before)
        self.assertEqual(len(optimizer.state),0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_nonfinite_assertion_aborts_isolated_process(self):
        code="import torch;torch._assert_async(torch.tensor(False,device='cuda'),'TCN nonfinite smoke');torch.cuda.synchronize()"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=20)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('assert',result.stderr.lower())


if __name__=='__main__':unittest.main()
