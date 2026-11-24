"""Calibrate L on validation only, enforcing C's clean validation accuracy."""
import hashlib
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data import Images
from fusion_trial import ROOT, RESULTS, masked
from models import Classifier


@torch.no_grad()
def main():
    raw = (RESULTS/"manifest.json").read_bytes()
    ck = torch.load(RESULTS/"readout_42/K_42/best.pt",map_location="cpu", weights_only=True)
    assert ck["manifest"] == hashlib.sha256(raw).hexdigest()
    model = Classifier("L").cuda().eval()
    missing, unexpected = model.load_state_dict(ck["state"],strict=False)
    assert set(missing)=={"readout_scale","readout_threshold"} and not unexpected
    ds = Images(ROOT/"data",json.loads(raw),"val")
    rows = []
    generator = torch.Generator().manual_seed(42)
    for x,y,_ in DataLoader(ds,batch_size=64):
        x = x.cuda()
        rank = torch.rand(len(y),64,generator=generator).cuda()
        logits, pooled = model(torch.cat([x,masked(x,rank)]), return_readout_inputs=True)
        rows.append((logits,model.local_readout(pooled),y.cuda().repeat(2),torch.arange(2,device=x.device).repeat_interleave(len(y))))
    base, correction, y, mode = [torch.cat(col) for col in zip(*rows)]
    conf = base.softmax(1).amax(1,keepdim=True)
    records = []
    for threshold in (.5,.7,.8,.9,.95,.98,.99,1.):
        for scale in (0.,.125,.25,.5,.75,1.):
            hits = (base + correction*(conf<=threshold)*scale).argmax(1)==y
            records.append([threshold,scale,*[int(hits[mode==i].sum()) for i in (0,1)]])
    frame = pd.DataFrame(records,columns=["threshold","scale","clean","random"])
    choices = frame[frame.clean >= frame.iloc[0].clean]
    best = choices.sort_values(["random","clean","scale"],ascending=[False,False,True]).iloc[0]
    model.readout_threshold.fill_(float(best.threshold))
    model.readout_scale.fill_(float(best.scale))
    out = RESULTS/"calibrated_readout_42"/"L_42"
    out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=model.state_dict(),manifest=ck["manifest"],epoch=ck["epoch"]),out/"best.pt")
    frame.to_csv(out/"validation_grid.csv",index=False)
    print(best,flush=True)


if __name__ == "__main__":
    main()
