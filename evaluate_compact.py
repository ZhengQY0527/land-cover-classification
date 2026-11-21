"""Compare the two-layer fusion model against the seed-42 reference runs."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from data import Images
from models import Classifier


@torch.inference_mode()
def main():
    root = Path(__file__).resolve().parent
    results = root/'results'
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--lightweight', action='store_true')
    mode.add_argument('--serial', action='store_true')
    args = parser.parse_args()
    serial = args.serial
    compact = results/'compact'
    destination = results/'lightweight' if args.lightweight else (results/'serial' if serial else compact)
    manifest = (results/'manifest.json').read_bytes()
    ds = Images(root/'data', json.loads(manifest), 'test')
    focus = set(pd.read_csv(results/'occlusion_42.csv').path)
    loader = DataLoader(Subset(ds, [i for i,p in enumerate(ds.paths) if p in focus]), batch_size=32)
    models = {}
    rows = []
    for group in ('BG' if args.lightweight else ('ABCFI' if serial else 'ABCF')):
        base = {'F':compact, 'I':destination,'G':destination}.get(group, results)
        folder = base/f'{group}_42'
        checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
        assert checkpoint['manifest'] == hashlib.sha256(manifest).hexdigest()
        model = Classifier(group).eval().cuda()
        model.load_state_dict(checkpoint['state'])
        models[group] = model
        meta = json.loads((folder/'metrics.json').read_text())
        cm = pd.read_csv(folder/'confusion_counts.csv', index_col=0)
        def errors(a,b):
            return cm.at[a,b] + cm.at[b,a]
        rows.append(dict(model=group, params=meta['parameters'], test_accuracy=meta['accuracy'],
                         macro_f1=meta['macro_f1'], val_f1=pd.read_csv(folder/'history.csv').val_f1.max(),
                         buildings=errors('Industrial','Residential'),
                         vegetation=errors('PermanentCrop','HerbaceousVegetation')))
    frame = pd.DataFrame(rows).set_index('model')
    for group in ('' if args.lightweight else ('FI' if serial else 'F')):
        rng = torch.Generator().manual_seed(42)
        counts = {mode:0 for mode in ('clean','high','low','random')}
        for x,y,_ in loader:
            x,y = x.cuda(),y.cuda()
            logits,attention = models[group](x,explain=True)
            counts['clean'] += int((logits.argmax(1)==y).sum())
            for mode in ('high','low','random'):
                for _ in range(20 if mode=='random' else 1):
                    rank = torch.rand(len(y),64,generator=rng).cuda() if mode=='random' else attention
                    mask = torch.zeros_like(rank).scatter_(1,rank.topk(8,largest=mode!='low').indices,1)
                    mask = mask.reshape(-1,1,8,8).repeat_interleave(8,2).repeat_interleave(8,3)
                    counts[mode] += int((models[group](x*(1-mask)).argmax(1)==y).sum())
        for mode,correct in counts.items():
            frame.loc[group, mode+'_mask_accuracy'] = correct/(len(focus)*(20 if mode=='random' else 1))
    reference = pd.read_csv(results/'occlusion_accuracy_42.csv')
    for group in ('BG' if args.lightweight else 'AC'):
        if args.lightweight:
            base = destination if group=='G' else results
            frame.loc[group,'random_mask_accuracy'] = pd.read_csv(base/f'{group}_42/occlusion_accuracy.csv').set_index('mode').at['random','accuracy']
            continue
        frame.loc[group,'random_mask_accuracy'] = reference.query("group == @group and condition == 'random'").accuracy.item()
    frame.loc['B','random_mask_accuracy'] = pd.read_csv(results/'B_42/occlusion_accuracy.csv').set_index('mode').at['random','accuracy']
    image = ds[0][0][None].cuda()
    order = np.random.default_rng(42)
    times = {g:[] for g in models}
    for m in models.values():
        for _ in range(60):
            m(image)
    torch.cuda.synchronize()
    for _ in range(10):
        for g in order.permutation(list(models)):
            for _ in range(40):
                start = perf_counter()
                models[g](image)
                torch.cuda.synchronize()
                times[g].append((perf_counter()-start)*1000)
    for g,values in times.items():
        frame.loc[g,['median_ms','p95_ms']] = np.percentile(values,[50,95])
    frame.to_csv(destination/'comparison.csv')
    print(frame.to_string())
    print('FP32 batch=1',torch.cuda.get_device_name(),torch.__version__)


if __name__ == '__main__':
    main()
