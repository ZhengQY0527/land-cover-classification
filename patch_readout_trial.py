"""Fit M/N on the seed-42 split. Evaluate the validation-selected checkpoints."""
import hashlib
import json
import random

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, Subset

from data import CLASSES, Images
from fusion_trial import RESULTS, ROOT, masked
from models import Classifier

OUT = RESULTS / "patch_readout_42"


def block(x, rng, noise=False):
    view = x.clone()
    for image in view:
        h,w = (2,4) if torch.randint(2,(1,),generator=rng).item() else (4,2)
        r = int(torch.randint(9-h,(1,),generator=rng))*8
        c = int(torch.randint(9-w,(1,),generator=rng))*8
        region = image[:,r:r+8*h,c:c+8*w]
        if noise:
            region.copy_(torch.randn(region.shape,generator=rng).to(image.device)*0.6)
        else:
            region.zero_()
    return view


@torch.no_grad()
def cache(model, info, split):
    rng = torch.Generator().manual_seed(142 if split=="train" else 42)
    ds = Images(ROOT/"data",info,split)
    pieces = []
    for x,y,_ in DataLoader(ds,batch_size=48):
        x = x.cuda()
        rank = torch.rand(len(y),64,generator=rng).cuda()
        views = torch.cat([x,masked(x,rank),block(x,rng)])
        base,tokens = model(views,return_readout_inputs=True)
        pieces.append((base.reshape(3,len(y),10).cpu(),
                       tokens.reshape(3,len(y),64,128).half().cpu(), y[None]))
    return tuple(torch.cat(col,dim=1) for col in zip(*pieces))


def fit(model, train, val, group, digest):
    head = model.patch_readout
    opt = torch.optim.AdamW(head.parameters(),lr=.002)
    base, patches, y = train
    vb,vp,vy = val
    vb,vy = vb.cuda(),vy.cuda()
    clean_min = int((vb[0].argmax(-1)==vy[0]).sum())
    best = (-1,-1)
    records = []
    rng = torch.Generator().manual_seed(42)
    for epoch in range(13):
        if epoch:
            for ix in torch.randperm(y.numel(),generator=rng).split(128):
                logits = base[:,ix].cuda()
                truth = y[:,ix].cuda().expand(3,-1)
                patch = patches[:,ix].flatten(0,1).cuda().float()
                output = logits + head(patch).reshape_as(logits)
                loss = F.cross_entropy(output.flatten(0,1),truth.reshape(-1))
                teacher = (logits[0]/2).softmax(-1).detach().repeat(2,1)
                loss += .5*4*F.kl_div((output[1:].flatten(0,1)/2).log_softmax(-1),teacher,reduction="batchmean")
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            head.scale.fill_(1.)
            delta = torch.cat([head(chunk.cuda().float()) for chunk in vp.flatten(0,1).split(256)]).reshape_as(vb)
            for scale in (0.,.125,.25,.5,.75,1.):
                hits = ((vb+scale*delta).argmax(-1)==vy).sum(1).tolist()
                records.append([epoch,scale,*hits])
                key = (hits[1]+hits[2],hits[0])
                if hits[0]>=clean_min and key>best:
                    best = key
                    head.scale.fill_(scale)
                    torch.save(dict(state=model.state_dict(),epoch=epoch,scale=scale,manifest=digest),OUT/f"{group}.pt")
            head.scale.fill_(1.)
    pd.DataFrame(records,columns=["epoch","scale","clean","random","block"]).to_csv(OUT/f"{group}_validation.csv",index=False)
    print(group,"selected validation counts",best,flush=True)


@torch.no_grad()
def test(models, info):
    ds = Images(ROOT/"data",info,"test")
    focus = set(pd.read_csv(RESULTS/"occlusion_42.csv").path)
    rows = []
    # All models have exactly the same frozen backbone.
    def predict(x,y,paths,condition,rep=0):
        base,tok = models["M"](x,return_readout_inputs=True)
        for group,model in models.items():
            pred = (base+model.patch_readout(tok)).argmax(1).tolist()
            rows.extend(zip(paths,y.tolist(),[group]*len(y),[condition]*len(y),[rep]*len(y),pred))
    block_rng = torch.Generator().manual_seed(42)
    noise_rng = torch.Generator().manual_seed(42)
    for x,y,paths in DataLoader(ds,batch_size=64):
        x = x.cuda()
        predict(x,y,paths,"clean")
        predict(block(x,block_rng),y,paths,"block")
        predict(block(x,noise_rng,True),y,paths,"noise_block")
    subset = Subset(ds,[i for i,p in enumerate(ds.paths) if p in focus])
    assert len(subset)==1650
    rng = torch.Generator().manual_seed(42)
    for x,y,paths in DataLoader(subset,batch_size=32):
        x=x.cuda()
        _,rank=models["M"](x,explain=True)
        for high in (True,False):
            predict(masked(x,rank,high),y,paths,"high" if high else "low")
        for rep in range(20):
            predict(masked(x,torch.rand(len(y),64,generator=rng).cuda()),y,paths,"random",rep)
    frame = pd.DataFrame(rows,columns=["path","truth","group","condition","rep","prediction"])
    frame.to_csv(OUT/"predictions.csv",index=False)
    summary=[]
    for (g,c),d in frame.groupby(["group","condition"]):
        hits = d.truth==d.prediction
        summary.append([g,c,len(d),hits.sum(),hits.mean(),f1_score(d.truth,d.prediction,average="macro",labels=sorted(d.truth.unique()))])
        if c in ("block","clean"):
            pd.DataFrame(confusion_matrix(d.truth,d.prediction),index=CLASSES,columns=CLASSES).to_csv(OUT/f"{g}_{c}_confusion.csv")
    table=pd.DataFrame(summary,columns=["group","condition","count","correct","accuracy","macro_f1"])
    table.to_csv(OUT/"test.csv",index=False)
    print(table,flush=True)


def main():
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(3)
    OUT.mkdir(exist_ok=True)
    raw=(RESULTS/"manifest.json").read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    info=json.loads(raw)
    ck=torch.load(RESULTS/"C_42/best.pt",map_location="cpu",weights_only=True)
    assert ck["manifest"]==digest
    models={}
    for g in ("M","N"):
        model=Classifier(g).cuda().eval()
        missing,extra=model.load_state_dict(ck["state"],strict=False)
        assert not extra and all(k.startswith("patch_readout.") for k in missing)
        models[g]=model
    models["N"].patch_readout.gate.load_state_dict(models["M"].patch_readout.gate.state_dict())
    train=cache(models["M"],info,"train")
    val=cache(models["M"],info,"val")
    print("Feature extraction complete",flush=True)
    for g,m in models.items():
        fit(m,train,val,g,digest)
        saved=torch.load(OUT/f"{g}.pt",map_location="cpu",weights_only=True)
        m.load_state_dict(saved["state"])
        print(g,saved["epoch"],saved["scale"],sum(p.numel() for p in m.parameters()),flush=True)
    test(models,info)


if __name__=="__main__":
    main()
