"""Input-side repair for the single-backbone L classifier, calibrated on validation."""
import hashlib
import json
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from data import Images, CLASSES
from models import Classifier
from patch_readout_trial import OUT, ROOT, RESULTS, masked, block


class Repair(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode=mode
        self.backbone=Classifier("L")

    def forward(self,x):
        patches=x.unfold(2,8,8).unfold(3,8,8).permute(0,2,3,1,4,5)
        valid=(patches.amax((-2,-1))-patches.amin((-2,-1))).amax(-1)>1e-6
        if valid.all():
            return self.backbone(x)
        # Nearest visible grid cells; average tied cells or use the first.
        row=torch.arange(8,device=x.device)
        ij=torch.cartesian_prod(row,row).float()
        dist=torch.cdist(ij,ij,p=1)[None].masked_fill(~valid.reshape(-1,1,64),float("inf"))
        pixels=patches.reshape(-1,64,192)
        if self.mode=="O":
            take=dist.argmin(-1)
            fixed=pixels.gather(1,take.unsqueeze(-1).expand_as(pixels))
        else:
            weights=(dist==dist.amin(-1,keepdim=True)).to(x.dtype)
            fixed=(weights/weights.sum(-1,keepdim=True))@pixels
        image=fixed.reshape(-1,8,8,3,8,8).permute(0,3,1,4,2,5).reshape_as(x)
        return self.backbone(image)


@torch.no_grad()
def main():
    raw=(RESULTS/"manifest.json").read_bytes()
    info=json.loads(raw)
    ck=torch.load(RESULTS/"calibrated_readout_42/L_42/best.pt",map_location="cpu",weights_only=True)
    assert ck["manifest"]==hashlib.sha256(raw).hexdigest()
    models={g:Repair(g).cuda().eval() for g in ("O","P")}
    for m in models.values():
        m.backbone.load_state_dict(ck["state"])
    rows=[]
    def run(x,y,paths,split,condition,rep=0):
        for g,m in models.items():
            preds=m(x).argmax(1).tolist()
            rows.extend(zip(paths,y.tolist(),[g]*len(y),[split]*len(y),[condition]*len(y),[rep]*len(y),preds))
    for split in ("val","test"):
        ds=Images(ROOT/"data",info,split)
        rng=torch.Generator().manual_seed(42)
        noise_rng=torch.Generator().manual_seed(42)
        for x,y,paths in DataLoader(ds,batch_size=64):
            x=x.cuda()
            run(x,y,paths,split,"clean")
            run(block(x,rng),y,paths,split,"block")
            if split=="test":
                run(block(x,noise_rng,True),y,paths,split,"noise_block")
        if split=="val":
            val=pd.DataFrame(rows,columns=["path","truth","model","split","condition","rep","prediction"])
            scores=val.assign(hit=val.truth==val.prediction).groupby(["model","condition"]).hit.sum().unstack()
            eligible=scores[scores.clean>=3959]
            chosen=eligible.block.idxmax() if len(eligible) else "L"
            (OUT/"repair_selection.json").write_text(json.dumps(dict(selected=chosen,val=scores.to_dict())),encoding="utf8")
            print(scores,chosen,flush=True)
    focus=set(pd.read_csv(RESULTS/"occlusion_42.csv").path)
    ds=Subset(ds,[i for i,p in enumerate(ds.paths) if p in focus])
    rng=torch.Generator().manual_seed(42)
    for x,y,paths in DataLoader(ds,batch_size=32):
        x=x.cuda()
        _,rank=models["P"].backbone(x,explain=True)
        for high in (True,False):
            run(masked(x,rank,high),y,paths,"test","high" if high else "low")
        for rep in range(20):
            run(masked(x,torch.rand(len(y),64,generator=rng).cuda()),y,paths,"test","random",rep)
    frame=pd.DataFrame(rows,columns=["path","truth","model","split","condition","rep","prediction"])
    frame.to_csv(OUT/"repair_predictions.csv",index=False)
    summary=[]
    for (g,s,c),part in frame.groupby(["model","split","condition"]):
        hits=part.truth==part.prediction
        summary.append([g,s,c,len(part),int(hits.sum()),hits.mean(),f1_score(part.truth,part.prediction,average="macro",labels=sorted(part.truth.unique()))])
        if s=="test" and c in ("block","clean"):
            pd.DataFrame(confusion_matrix(part.truth,part.prediction),index=CLASSES,columns=CLASSES).to_csv(OUT/f"{g}_{c}_confusion.csv")
    report=pd.DataFrame(summary,columns=["group","split","condition","count","correct","accuracy","macro_f1"])
    report.to_csv(OUT/"repair.csv",index=False)
    print(report,flush=True)


if __name__=="__main__":
    main()
