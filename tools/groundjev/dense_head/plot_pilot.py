"""Standalone scientific loss and DEV-AP figure; no training environment mutation."""
import json
from pathlib import Path
import sys

RUN=Path('/data3/user/LocateMOT-runs/G2_Spatial_DenseHead_20261026')
# Optional plotting libraries are isolated from the running torch environments.
sys.path.insert(0,str(RUN/'plot_dependencies'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
    for variant,style in [('', '--'),('spatial_prior_v2','-')]:
      for arm,color in [('g0','#377eb8'),('g2','#e66101')]:
        directory=RUN/variant/arm
        label=arm.upper()+(' global' if not variant else ' spatial prior')
        rows=[json.loads(line) for line in (directory/'loss.jsonl').open()]
        steps=np.array([r['step'] for r in rows]);loss=np.array([r['loss'] for r in rows])
        axes[0].plot(steps,loss,color=color,alpha=.15,lw=.6)
        smooth=np.convolve(loss,np.ones(30)/30,mode='valid')
        axes[0].plot(steps[29:],smooth,color=color,ls=style,label=label)
        checks=[json.loads(p.read_text()) for p in sorted(directory.glob('dev_step*.json'))]
        for dataset,axis in [('sportsmot',axes[1]),('animaltrack',axes[2])]:
            axis.plot([r['step'] for r in checks],
                      [100*r['metrics_at_0.5'][dataset]['AP50'] for r in checks],
                      marker='o',ls=style,color=color,label=label,ms=4)
    axes[0].set(title='Optimizer loss (not a performance claim)',ylabel='Hungarian detection loss')
    axes[1].set(title='SportsMOT internal DEV',ylabel='Class-agnostic AP50 (%)')
    axes[2].set(title='AnimalTrack internal DEV',ylabel='Class-agnostic AP50 (%)')
    for axis in axes:axis.set_xlabel('Optimizer step');axis.grid(alpha=.2);axis.legend(frameon=False)
    fig.suptitle('Frozen native spatial features + matched parallel object decoder; DEV was in G2 TRAIN',fontsize=11)
    out=Path(__file__).resolve().parents[3]/'docs/GROUNDJEV_DENSE_HEAD/PILOT_CURVES.png'
    fig.savefig(out,dpi=180)
    fig.savefig(RUN/'PILOT_CURVES.svg')
    print(str(out))


if __name__=='__main__':main()
