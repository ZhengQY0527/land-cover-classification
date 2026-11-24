"""Out-of-pattern check: one contiguous 2x4 or 4x2 block per test image."""
import hashlib
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from data import Images
from models import Classifier
from fusion_trial import Gate, ROOT, RESULTS


@torch.no_grad()
def main():
    raw = (RESULTS/"manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    folders = dict(B="B_42",C="C_42",J="spatial_resumed/J_42",K="readout_42/K_42",L="calibrated_readout_42/L_42")
    models = {}
    for group,folder in folders.items():
        ck = torch.load(RESULTS/folder/"best.pt",map_location="cpu",weights_only=True)
        assert ck["manifest"] == digest
        models[group] = Classifier(group).cuda().eval()
        models[group].load_state_dict(ck["state"])
    gate = Gate().cuda().eval()
    ck = torch.load(RESULTS/"fusion_42/gate.pt",map_location="cpu",weights_only=True)
    assert ck["manifest"] == digest
    gate.load_state_dict(ck["state"])
    rng = torch.Generator().manual_seed(42)
    ds = Images(ROOT/"data",json.loads(raw),"test")
    rows = []
    for x,y,paths in DataLoader(ds,batch_size=64):
        for image in x:
            h,w = (2,4) if torch.randint(2,(1,),generator=rng).item() else (4,2)
            r = int(torch.randint(9-h,(1,),generator=rng))*8
            c = int(torch.randint(9-w,(1,),generator=rng))*8
            image[:,r:r+8*h,c:c+8*w] = 0
        x = x.cuda()
        probs = {g:m(x).softmax(1) for g,m in models.items()}
        p = torch.stack([probs[g] for g in ("B","C","J")],1)
        probs.update(equal=p.mean(1),gate=gate(p))
        for g,s in probs.items():
            rows.extend(zip(paths,y.tolist(),[g]*len(y),s.argmax(1).tolist()))
    df = pd.DataFrame(rows,columns=["path","truth","group","prediction"])
    out = RESULTS/"fusion_42"
    df.to_csv(out/"block_predictions.csv",index=False)
    records = []
    for name,part in df.groupby("group"):
        hits = part.truth==part.prediction
        records.append([name,hits.sum(),len(part),hits.mean(),f1_score(part.truth,part.prediction,average="macro")])
    table = pd.DataFrame(records,columns=["model","hits","count","accuracy","macro_f1"])
    table.to_csv(out/"block.csv",index=False)
    print(table)


if __name__ == "__main__":
    main()
