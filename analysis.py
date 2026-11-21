import json
import itertools
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from data import CLASSES, Images, read_image
from models import Classifier


def plot_matrix(matrix, folder):
    fig, ax = plt.subplots(figsize=(9,8))
    im = ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues')
    ax.set_xticks(range(10),CLASSES,rotation=60,ha='right')
    ax.set_yticks(range(10),CLASSES)
    ax.set(xlabel='Prediction',ylabel='True class')
    fig.colorbar(im,ax=ax)
    fig.tight_layout()
    fig.savefig(folder/'confusion.png')
    plt.close(fig)


def direction(frame,a,b):
    subset = frame[frame.truth==a]
    count = int((subset.prediction==b).sum())
    return count,count/len(subset)


def compare(args,info,fingerprint):
    val = pd.concat([pd.read_csv(args.out/f'A_{s}'/'validation.csv') for s in args.seeds])
    veg = ['AnnualCrop','PermanentCrop','Pasture','HerbaceousVegetation']
    pair = max(itertools.combinations(veg,2),key=lambda ab:direction(val,*ab)[1]+direction(val,*reversed(ab))[1])
    rows = []
    for seed in args.seeds:
        for group in args.groups:
            folder = args.out/f'{group}_{seed}'
            if json.loads((folder/'metrics.json').read_text())['manifest'] != fingerprint:
                raise ValueError('Split mismatch')
            frame = pd.read_csv(folder/'predictions.csv')
            for a,b in [('Industrial','Residential'),pair]:
                for src,dst in [(a,b),(b,a)]:
                    rows.append((seed,group,src,dst,*direction(frame,src,dst)))
    pd.DataFrame(rows,columns=['seed','group','truth','confused_with','count','rate']).to_csv(args.out/'confused_pairs.csv',index=False)
    return pair


@torch.inference_mode()
def interventions(args,info,fingerprint,seed,pair,device):
    ds = Images(args.data,info,'test')
    subset = [i for i,p in enumerate(ds.paths) if Path(p).parent.name in {'Industrial','Residential',*pair}]
    loader = DataLoader(Subset(ds,subset),batch_size=min(args.batch,32))
    frames = [pd.read_csv(args.out/f'{g}_{seed}'/'predictions.csv').set_index('path') for g in ('A','C')]
    examples = {}
    for i in subset:
        path = ds.paths[i]
        correct = tuple(f.at[path,'truth']==f.at[path,'prediction'] for f in frames)
        case = {(False,True):'corrected',(True,False):'regressed',(False,False):'both_wrong'}.get(correct)
        if case:
            examples.setdefault(case,path)
    rows = []
    accuracy_rows = []
    for group in ('A','C'):
        totals = {mode: [0, 0] for mode in ('clean','high','low','random')}
        folder = args.out/f'{group}_{seed}'
        ckpt = torch.load(folder/'best.pt',map_location=device,weights_only=True)
        if ckpt['manifest'] != fingerprint:
            raise ValueError('Checkpoint and split mismatch')
        model = Classifier(group).to(device)
        model.load_state_dict(ckpt['state'])
        model.eval()
        rng = torch.Generator().manual_seed(seed)
        for x,y,paths in loader:
            x,y = x.to(device),y.to(device)
            logits,attention = model(x,explain=True)
            idx = torch.arange(len(y),device=device)
            correct = logits.argmax(1)==y
            totals['clean'][0] += int(correct.sum())
            totals['clean'][1] += len(y)
            for mode in ('high','low','random'):
                drops,flips = [],[]
                for _ in range(args.random_masks if mode=='random' else 1):
                    rank = torch.rand(len(y),64,generator=rng).to(device) if mode=='random' else attention
                    chosen = rank.topk(8,largest=mode!='low').indices
                    mask = torch.zeros_like(rank).scatter_(1,chosen,1).reshape(-1,1,8,8)
                    mask = mask.repeat_interleave(8,2).repeat_interleave(8,3)
                    masked = model(x*(1-mask))
                    drops.append(logits[idx,y]-masked[idx,y])
                    totals[mode][0] += int((masked.argmax(1)==y).sum())
                    totals[mode][1] += len(y)
                    flips.append((masked.argmax(1)!=y).float())
                drop = torch.stack(drops).mean(0).tolist()
                flip = torch.stack(flips).mean(0).tolist()
                for i,path in enumerate(paths):
                    rows.append((group,path,CLASSES[y[i].item()],mode,drop[i],flip[i] if correct[i] else np.nan))
                    if mode=='high' and path in examples.values():
                        case = next(k for k,v in examples.items() if v==path)
                        fig,axes = plt.subplots(1,3,figsize=(9,4))
                        img = np.asarray(read_image(args.data/path))
                        axes[0].imshow(img)
                        axes[1].imshow(img)
                        axes[1].imshow(attention[i].cpu().reshape(8,8),extent=(-.5,63.5,63.5,-.5),alpha=.5,cmap='inferno',interpolation='nearest')
                        hidden = mask[i,0].cpu().numpy()[:,:,None]
                        axes[2].imshow((img*(1-hidden)+255*np.array(info['mean'])*hidden).astype('uint8'))
                        for ax in axes:
                            ax.axis('off')
                        fig.suptitle(f"{case} | {path} | {group}: {CLASSES[logits[i].argmax().item()]} | masked: {CLASSES[masked[i].argmax().item()]} | drop {drop[i]:.2f}")
                        fig.tight_layout()
                        fig.savefig(folder/f'{case}.png')
                        plt.close(fig)
        accuracy_rows.extend((group,mode,hits,n,hits/n) for mode,(hits,n) in totals.items())
    pd.DataFrame(accuracy_rows,columns=['group','condition','correct','predictions','accuracy']).to_csv(
        args.out/f'occlusion_accuracy_{seed}.csv',index=False)
    frame = pd.DataFrame(rows,columns=['group','path','truth','mask','logit_drop','flip_if_correct'])
    frame.to_csv(args.out/f'occlusion_{seed}.csv',index=False)
    frame.groupby(['group','truth','mask'])[['logit_drop','flip_if_correct']].mean().to_csv(args.out/f'occlusion_summary_{seed}.csv')
    ax = frame.groupby(['group','mask']).logit_drop.mean().unstack().plot.bar(ylabel='Mean true-class logit drop')
    ax.figure.tight_layout()
    ax.figure.savefig(args.out/f'occlusion_{seed}.png')
    plt.close(ax.figure)
