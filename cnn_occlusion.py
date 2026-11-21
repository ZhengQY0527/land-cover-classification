"""Run the CNN occlusion control on the exact seed-42 A/C test subset."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from data import CLASSES, Images
from models import Classifier


def main():
    base = Path(__file__).resolve().parent
    out = base/'results'
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["B","G"], default="B")
    group = parser.parse_args().group
    folder = out/"B_42" if group=="B" else out/"lightweight"/"G_42"
    manifest = (out/'manifest.json').read_bytes()
    ds = Images(base/'data', json.loads(manifest), 'test')
    subset_paths = set(pd.read_csv(out/'occlusion_42.csv').path)
    indices = [i for i,p in enumerate(ds.paths) if p in subset_paths]
    assert len(indices) == len(subset_paths)
    loader = DataLoader(Subset(ds, indices), batch_size=32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(folder/'best.pt', map_location=device, weights_only=True)
    assert checkpoint['manifest'] == hashlib.sha256(manifest).hexdigest()
    model = Classifier(group).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['state'])
    rng = torch.Generator().manual_seed(42)
    rows = []
    for x,y,paths in loader:
        x,y = x.to(device),y.to(device)
        grid = model.project(model.local(x)).requires_grad_()
        logits = model.head(grid.mean((2,3)))
        pred = logits.argmax(1)
        grad, = torch.autograd.grad(logits.gather(1,pred[:,None]).sum(), grid)
        cam = (grad.mean((2,3),keepdim=True)*grid.detach()).sum(1).relu().flatten(1)
        with torch.no_grad():
            for mode in ('clean','high','low','random'):
                repeats = 20 if mode=='random' else 1
                hits = torch.zeros_like(y)
                drop = torch.zeros_like(y,dtype=x.dtype)
                for _ in range(repeats):
                    masked = logits
                    if mode!='clean':
                        if mode=='random':
                            rank = torch.rand(len(y),64,generator=rng).to(device)
                            chosen = rank.topk(8).indices
                        else:
                            # Stable ordering gives disjoint extremes even when ReLU ties at zero.
                            order = cam.argsort(dim=1,stable=True)
                            chosen = order[:,-8:] if mode=='high' else order[:,:8]
                        mask = torch.zeros_like(cam).scatter_(1,chosen,1).reshape(-1,1,8,8)
                        masked = model(x*(1-mask.repeat_interleave(8,2).repeat_interleave(8,3)))
                    hits += masked.argmax(1)==y
                    drop += (logits-masked).gather(1,y[:,None]).squeeze(1)
                for i,path in enumerate(paths):
                    rows.append((path,CLASSES[y[i].item()],mode,pred[i].item()==y[i].item(),
                                 hits[i].item(),repeats,drop[i].item()/repeats))
    df = pd.DataFrame(rows,columns=['path','truth','mode','original_correct','hits','trials','logit_drop'])
    df.to_csv(folder/'occlusion_images.csv',index=False)
    for keys,name in [(['mode'],'occlusion_accuracy'),(['truth','mode'],'occlusion_by_class')]:
        summary = df.groupby(keys).agg(correct=('hits','sum'),predictions=('trials','sum'),
                                     mean_logit_drop=('logit_drop','mean'))
        summary['accuracy'] = summary.correct/summary.predictions
        summary.to_csv(folder/f'{name}.csv')
        print(summary)


if __name__ == '__main__':
    main()
