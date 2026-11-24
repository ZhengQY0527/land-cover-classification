"""Benchmark Q and C on paired test views, including unaligned texture occlusion."""
import hashlib
import json
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader,Subset
from data import CLASSES,Images
from patch_readout_trial import ROOT,RESULTS,OUT,Classifier,block,masked


@torch.no_grad()
def main():
    raw=(RESULTS/"manifest.json").read_bytes()
    ds=Images(ROOT/"data",json.loads(raw),"test")
    models={}
    for name,path in [("C",RESULTS/"C_42/best.pt"),("Q",OUT/"Q.pt"),
                      ("Q-blend",OUT/"Q_blend.pt"),("Q-full-step",OUT/"Q_full_step.pt")]:
        ck=torch.load(path,map_location="cpu",weights_only=True)
        assert ck["manifest"]==hashlib.sha256(raw).hexdigest()
        model=Classifier("C").cuda().eval()
        model.load_state_dict(ck["state"])
        models[name]=model
    rows=[]
    def predict(x,y,paths,mode,rep=0):
        for g,m in models.items():
            rows.extend(zip(paths,y.tolist(),[g]*len(y),[mode]*len(y),[rep]*len(y),m(x).argmax(1).tolist()))
    rngs=[torch.Generator().manual_seed(42) for _ in range(3)]
    donors=torch.randperm(len(ds),generator=rngs[2]).tolist()
    offset=0
    for x,y,paths in DataLoader(ds,batch_size=64):
        x=x.cuda()
        predict(x,y,paths,"clean")
        predict(block(x,rngs[0]),y,paths,"block")
        predict(block(x,rngs[1],True),y,paths,"noise_block")
        pasted=x.clone()
        for j in range(len(x)):
            r,c=torch.randint(0,43,(2,),generator=rngs[2]).tolist()
            donor=ds[donors[offset+j]][0]
            pasted[j,:,r:r+22,c:c+22]=donor[:,r:r+22,c:c+22]
        offset+=len(x)
        predict(pasted,y,paths,"texture")
    focus=set(pd.read_csv(RESULTS/"occlusion_42.csv").path)
    subset=Subset(ds,[i for i,p in enumerate(ds.paths) if p in focus])
    rng=torch.Generator().manual_seed(42)
    for x,y,paths in DataLoader(subset,batch_size=32):
        x=x.cuda()
        _,rank=models["C"](x,explain=True)
        for high in (True,False):
            predict(masked(x,rank,high),y,paths,"high" if high else "low")
        for rep in range(20):
            predict(masked(x,torch.rand(len(y),64,generator=rng).cuda()),y,paths,"random",rep)
    df=pd.DataFrame(rows,columns=["path","truth","group","condition","rep","prediction"])
    df.to_csv(OUT/"Q_predictions.csv",index=False)
    results=[]
    for (g,c),d in df.groupby(["group","condition"]):
        hits=d.truth==d.prediction
        results.append([g,c,len(d),hits.sum(),hits.mean(),f1_score(d.truth,d.prediction,average="macro",labels=sorted(d.truth.unique()))])
        if g!="C" and c in ("clean","block"):
            pd.DataFrame(confusion_matrix(d.truth,d.prediction),index=CLASSES,columns=CLASSES).to_csv(OUT/f"{g}_{c}_confusion.csv")
    table=pd.DataFrame(results,columns=["group","condition","count","correct","accuracy","macro_f1"])
    table.to_csv(OUT/"Q_test.csv",index=False)
    print(table)


if __name__=="__main__":
    main()
