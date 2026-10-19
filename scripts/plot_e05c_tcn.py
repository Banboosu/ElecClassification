"""Plot saved N1 diagnostics or completed replication; no inference/training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = {'current_fixed50':'Current masked', 'legacy_masked50':'Legacy masked',
         'legacy_unmasked50':'Legacy unmasked'}
COLORS = {'current_fixed50':'#287da8', 'legacy_masked50':'#e78712',
          'legacy_unmasked50':'#26865a'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--mode',choices=('diagnostics','replication'),required=True)
    args = parser.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    if args.mode == 'diagnostics':
        s = json.loads((args.analysis/'diagnostics/summary.json').read_text())
        if s['test_evaluated'] or len(s['padding']) != 18:
            raise ValueError('Require complete validation-only diagnostics')
        fig,axes = plt.subplots(2,2,figsize=(10.5,7.4))
        for col,fraction in enumerate((.1,1.)):
            for condition in NAMES:
                rows = [r for r in s['padding'] if r['fraction']==fraction and r['condition']==condition]
                rows.sort(key=lambda r:r['width'])
                baseline = rows[0]['metrics']['macro_f1']
                axes[0,col].plot([r['width'] for r in rows],
                                 [100*(r['metrics']['macro_f1']-baseline) for r in rows],
                                 marker='o',label=NAMES[condition],color=COLORS[condition],alpha=.85)
                axes[1,col].plot([r['width'] for r in rows],[r['changed_predictions'] for r in rows],
                                 marker='o',color=COLORS[condition],alpha=.85)
            axes[0,col].set_title(f'{100*fraction:g}% training labels, seed 43')
            axes[0,col].set_ylim(-3.5, .6)
            axes[1,col].set_xlabel('Padded width (valid observations unchanged)')
            axes[1,col].set_ylim(-4,110)
            for axis in axes[:,col]:axis.set_xticks([1024,1280,1536])
        axes[0,0].set_ylabel('Validation Macro-F1 change (pp)')
        axes[1,0].set_ylabel('Changed predictions / 3510')
        handles,labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
        footer='Fixed saved checkpoints; reused validation; no new test evaluation. Overlapping masked curves are unchanged.'
        output=Path('docs/figures/e05c_length_padding_20261019.svg')
    else:
        s=json.loads((args.analysis/'summary.json').read_text())
        if s['audit_status']!='passed' or s['conditions']!=36 or s['test_evaluated']:
            raise ValueError('Require completed36-condition audit')
        fig,axes=plt.subplots(1,3,figsize=(12,4.6),sharey=True)
        for axis,fraction in zip(axes,(.1,.4,1.)):
            for i,condition in enumerate(NAMES):
                rows=sorted([r for r in s['per_condition'] if r['fraction']==fraction and r['condition']==condition],key=lambda r:r['seed'])
                scores=[100*r['selected']['macro_f1'] for r in rows]
                axis.scatter(i+np.linspace(-.08,.08,4),scores,color=COLORS[condition],s=25,alpha=.8)
                axis.errorbar(i,np.mean(scores),yerr=np.std(scores,ddof=1),fmt='D',color='black',capsize=4,markersize=5)
            for seed in (42,44,45,46):
                scores=[100*next(r for r in s['per_condition'] if r['fraction']==fraction and r['seed']==seed and r['condition']==c)['selected']['macro_f1'] for c in NAMES]
                axis.plot(range(3),scores,color='#777777',alpha=.25,linewidth=.8,zorder=0)
            axis.set_title(f'{100*fraction:g}% training labels')
            axis.set_xticks(range(3),['Current\nmasked','Legacy\nmasked','Legacy\nunmasked'])
        axes[0].set_ylabel('Validation Macro-F1 (%)')
        footer='Four overlapping random splits (42/44/45/46); dots: individual seeds; diamonds: mean +/- sample SD. Pilot43 excluded.'
        output=Path('docs/figures/e05c_tcn_replication_20261019.svg')
    for axis in np.asarray(axes).ravel():
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',alpha=.2)
    fig.text(.5,.015,footer,ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94))
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__=='__main__':main()
