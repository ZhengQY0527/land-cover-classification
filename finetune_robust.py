"""Fine-tune C end-to-end with clean/occluded pairs and a frozen clean teacher."""
import hashlib
import json
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import Images
from patch_readout_trial import ROOT,RESULTS,OUT,block,masked
from models import Classifier


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--interpolate",action="store_true")
    interpolate=parser.parse_args().interpolate
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(3)
    raw=(RESULTS/"manifest.json").read_bytes()
    info=json.loads(raw)
    digest=hashlib.sha256(raw).hexdigest()
    ck=torch.load(RESULTS/"C_42/best.pt",map_location="cpu",weights_only=True)
    assert ck["manifest"]==digest
    model=Classifier("C").cuda().eval()
    model.load_state_dict(ck["state"])
    sets={}
    for split in ("train","val"):
        features=[]
        rng=torch.Generator().manual_seed(42)
        with torch.no_grad():
            for x,y,_ in DataLoader(Images(ROOT/"data",info,split),batch_size=96):
                x=x.cuda()
                if split=="val":
                    x=torch.cat([x,block(x,rng)])
                target=model(x)
                features.append((x.half().cpu(),y.repeat(2 if split=="val" else 1),target.cpu()))
        sets[split]=[torch.cat(part) for part in zip(*features)]
    images,labels,teacher=sets["train"]
    vi,vy,vt=sets["val"]
    # Within each original loader batch, clean examples precede the masked ones.
    clean=torch.cat([torch.arange(2).repeat_interleave(min(96,len(vi)//2-i))==0 for i in range(0,len(vi)//2,96)])
    clean_min=int((vt.argmax(1)[clean]==vy[clean]).sum())
    opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    records=[]
    dest=OUT/("Q_blend.pt" if interpolate else "Q.pt")
    best=(-1,-1)
    for epoch in range(2 if interpolate else 6):
        if epoch:
            # Keep the running BN statistics of the original clean-image backbone.
            model.eval()
            for ix in torch.randperm(len(labels)).split(96):
                x=images[ix].cuda().float()
                y=labels[ix].cuda()
                soft=(teacher[ix].cuda()/2).softmax(1)
                n=len(x)//2
                corrupted=x.clone()
                # Continuous locations, varying size/aspect, three types of fill.
                for j in range(n):
                    h=random.randint(12,27)
                    w=random.randint(12,27)
                    r=random.randint(0,64-h)
                    c=random.randint(0,64-w)
                    region=corrupted[j,:,r:r+h,c:c+w]
                    fill=random.randrange(3)
                    if fill==0:
                        region.zero_()
                    elif fill==1:
                        region.normal_(0,.45)
                    else:
                        region.copy_(x[(j+1)%len(x),:,r:r+h,c:c+w])
                rank=torch.rand(len(x)-n,64,device=x.device)
                corrupted[n:]=masked(x[n:],rank)
                logits=model(torch.cat([x,corrupted]))
                loss=F.cross_entropy(logits,y.repeat(2)) + 2*F.kl_div((logits/2).log_softmax(1),soft.repeat(2,1),reduction="batchmean")
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
        with torch.no_grad():
            if interpolate and epoch:
                tuned={k:t.clone() for k,t in model.state_dict().items()}
                blends=[]
                for alpha in (0.,.025,.05,.1,.2,.35,.5,.75,1.):
                    model.load_state_dict({k:torch.lerp(ck["state"][k].to(v.device),v,alpha) if v.is_floating_point() else v for k,v in tuned.items()})
                    pred=torch.cat([model(chunk.cuda().float()).argmax(1).cpu() for chunk in vi.split(96)])
                    hit=pred==vy
                    a,b=int(hit[clean].sum()),int(hit[~clean].sum())
                    blends.append([alpha,a,b])
                    if alpha==1.:
                        torch.save(dict(state=model.state_dict(),manifest=digest,epoch=1),OUT/"Q_full_step.pt")
                    if a>=clean_min and (b,a)>=best:
                        best=(b,a)
                        torch.save(dict(state=model.state_dict(),manifest=digest,alpha=alpha),dest)
                pd.DataFrame(blends,columns=["alpha","clean","block"]).to_csv(OUT/"Q_blend_validation.csv",index=False)
                print(blends,flush=True)
                break
            pred=torch.cat([model(chunk.cuda().float()).argmax(1).cpu() for chunk in vi.split(96)])
            hit=pred==vy
            a,b=int(hit[clean].sum()),int(hit[~clean].sum())
            records.append([epoch,a,b])
            if a>=clean_min and (b,a)>best:
                best=(b,a)
                torch.save(dict(group="C",state=model.state_dict(),manifest=digest,epoch=epoch),dest)
            print(epoch,a,b,flush=True)
            if not interpolate:
                pd.DataFrame(records,columns=["epoch","clean","block"]).to_csv(OUT/"Q_validation.csv",index=False)
    print("selected",best,flush=True)


if __name__=="__main__":
    main()
