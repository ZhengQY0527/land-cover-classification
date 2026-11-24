"""Compare spatial ViT J and its baseline C under identical seed-42 protocols."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from data import CLASSES, Images
from models import Classifier


@torch.inference_mode()
def main():
    root = Path(__file__).resolve().parent
    out = root/'results'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, default=out/'spatial')
    parser.add_argument('--groups', nargs='+', choices=['C','J','K','L'], default=['C','J'])
    args = parser.parse_args()
    work = args.work
    manifest = (out/'manifest.json').read_bytes()
    info = json.loads(manifest)
    focus = set(pd.read_csv(out/'occlusion_42.csv').path)
    rows, masks, predictions = [], [], []
    for group in args.groups:
        folder = out/'C_42' if group == 'C' else work/f'{group}_42'
        ck = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        assert ck['manifest'] == hashlib.sha256(manifest).hexdigest()
        model = Classifier(group).cuda().eval()
        model.load_state_dict(ck['state'])
        for split in ('val','test'):
            ds = Images(root/'data',info,split)
            data = []
            for x,y,paths in DataLoader(ds,batch_size=64):
                x = x.cuda()
                probs = []
                for k in range(4):
                    rotated = x.rot90(k,(2,3))
                    for view in (rotated,rotated.flip([3])):
                        probs.append(model(view).softmax(1))
                single = probs[0].argmax(1).tolist()
                eight = torch.stack(probs).mean(0).argmax(1).tolist()
                data.extend(zip(paths,y.tolist(),single,eight))
            frame = pd.DataFrame(data,columns=['path','truth','single','eight'])
            frame['group'],frame['split']=group,split
            predictions.append(frame)
            for mode in ('single','eight'):
                y,p = frame.truth,frame[mode]
                def pair(a,b):
                    i,j = CLASSES.index(a),CLASSES.index(b)
                    return int(((y==i)&(p==j)|(y==j)&(p==i)).sum())
                rows.append([group,split,mode,int((y==p).sum()),(y==p).mean(),f1_score(y,p,average='macro'),
                             pair('Industrial','Residential'),pair('HerbaceousVegetation','PermanentCrop')])
        subset = Subset(ds,[i for i,path in enumerate(ds.paths) if path in focus])
        counts = dict(clean=0,high=0,low=0,random=0)
        rng = torch.Generator().manual_seed(42)
        for x,y,_ in DataLoader(subset,batch_size=32):
            x,y = x.cuda(),y.cuda()
            logits,attention = model(x,explain=True)
            counts['clean'] += int((logits.argmax(1)==y).sum())
            for mode in ('high','low','random'):
                for _ in range(20 if mode=='random' else 1):
                    rank = torch.rand(len(y),64,generator=rng).cuda() if mode=='random' else attention
                    chosen = rank.topk(8,largest=mode!='low').indices
                    mask = torch.zeros_like(rank).scatter_(1,chosen,1).reshape(-1,1,8,8)
                    masked = x*(1-mask.repeat_interleave(8,2).repeat_interleave(8,3))
                    counts[mode] += int((model(masked).argmax(1)==y).sum())
        for mode,hits in counts.items():
            n=len(subset)*(20 if mode=='random' else 1)
            masks.append([group,mode,hits,n,hits/n])
    for name,records,columns in [('comparison',rows,['group','split','mode','correct','accuracy','macro_f1','building_errors','vegetation_errors']),
                                 ('occlusion',masks,['group','mask','correct','count','accuracy'])]:
        table=pd.DataFrame(records,columns=columns)
        table.to_csv(work/f'{name}.csv',index=False)
        print(table.to_string(index=False))
    pd.concat(predictions).to_csv(work/'view_predictions.csv',index=False)


if __name__=='__main__':
    main()
