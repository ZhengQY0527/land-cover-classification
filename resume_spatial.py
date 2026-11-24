"""Resume the interrupted J run; store full end-of-epoch state atomically."""
import ast
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from experiment import predict, score, plot_matrix
from models import Classifier
from data import Images, CLASSES


def save(obj, path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj,tmp)
    tmp.replace(path)


def main():
    base = Path(__file__).resolve().parent
    backup = base/'results/spatial_restart/paused_backup'
    out = base/'results/spatial_resumed'
    folder = out/'J_42'
    folder.mkdir(parents=True,exist_ok=True)
    raw = (backup/'manifest.json').read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    (out/'manifest.json').write_bytes(raw)
    info = json.loads(raw)
    loaders = {s:DataLoader(Images(base/'data',info,s),64,shuffle=s=='train') for s in ['train','val','test']}
    random.seed(42)
    torch.manual_seed(42)
    device='cuda'
    model=Classifier('J').to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.05)
    latest=folder/'last.pt'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        assert ck['manifest']==fingerprint
        model.load_state_dict(ck['state'])
        opt.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['cpu_rng'])
        torch.cuda.set_rng_state_all(ck['gpu_rng'])
        random.setstate(ck['python_rng'])
        history,best_state,best=ck['history'],ck['best_state'],ck['best']
    else:
        ck=torch.load(backup/'best.pt',map_location='cpu',weights_only=True)
        assert ck['manifest']==fingerprint and ck['group']=='J' and ck['seed']==42
        best_state=ck['state']
        model.load_state_dict(best_state)
        best_epoch=json.loads((backup/'status.json').read_text())['best']['epoch']
        old=[ast.literal_eval(s.split(' ',2)[2]) for s in (backup/'train.log').read_text().splitlines() if s.startswith('J 42 ')]
        history=old[:best_epoch]
        assert history[-1]['epoch']==best_epoch
        best=history[-1]['val_f1']
    for epoch in range(len(history),100):
        lr=.0003*max(.5*(1+np.cos(np.pi*(epoch-5+1)/95)),.001)
        opt.param_groups[0]['lr']=lr
        model.train()
        total=0.
        for x,y,_ in loaders['train']:
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x.to(device)),y.to(device))
            loss.backward()
            opt.step()
            total+=loss.item()*len(y)
        val=predict(model,loaders['val'],device)
        f1=score(val)
        if f1>best:
            best=f1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(dict(epoch=epoch+1,loss=total/len(loaders['train'].dataset),val_f1=f1))
        save(dict(state=model.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),
                  gpu_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),history=history,
                  best=best,best_state=best_state,manifest=fingerprint),latest)
        print(history[-1],flush=True)
    model.load_state_dict(best_state)
    save(dict(state=best_state,manifest=fingerprint,group='J',seed=42),folder/'best.pt')
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    predict(model,loaders['val'],device).to_csv(folder/'validation.csv',index=False)
    f=predict(model,loaders['test'],device)
    f.to_csv(folder/'predictions.csv',index=False)
    matrix=pd.DataFrame(confusion_matrix(f.truth,f.prediction,labels=CLASSES),index=CLASSES,columns=CLASSES)
    matrix.to_csv(folder/'confusion_counts.csv')
    rates=matrix.div(matrix.sum(axis=1),axis=0)
    rates.to_csv(folder/'confusion_rates.csv')
    plot_matrix(rates,folder)
    pd.DataFrame(dict(category=CLASSES,support=matrix.sum(axis=1).values,recall=rates.values.diagonal(),
                      f1=f1_score(f.truth,f.prediction,labels=CLASSES,average=None))).to_csv(folder/'per_class.csv',index=False)
    metrics=dict(group='J',seed=42,accuracy=(f.truth==f.prediction).mean(),macro_f1=score(f),
                 parameters=sum(p.numel() for p in model.parameters()),epochs=100,batch=64,
                 manifest=fingerprint,optimizer_reinitialized_after_epoch=58)
    (folder/'metrics.json').write_text(json.dumps(metrics))


if __name__=='__main__':
    main()
