
import argparse, json, numpy as np, torch, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score

class SeqSet(Dataset):
    def __init__(self, X, y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i], self.y[i]

class LSTMCls(nn.Module):
    def __init__(self, C, H=128, K=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=C, hidden_size=H, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(2*H, K)
    def forward(self, x):
        o,_ = self.lstm(x); o=o.mean(dim=1); return self.fc(o)

def metrics(y_true, y_pred):
    return dict(acc=float(accuracy_score(y_true,y_pred)),
                f1=float(f1_score(y_true,y_pred,average="macro")))

def eval_loader_preds(model, dl, dev):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for xb,yb in dl:
            logits = model(xb.to(dev))
            ps += list(torch.argmax(logits,dim=1).cpu().numpy())
            ys += list(yb.numpy())
    return np.array(ys), np.array(ps)

def main(args):
    arr = np.load(args.cache, allow_pickle=True)
    X, y, meta = arr["X"], arr["y"], arr["meta"]
    subj = np.array([m[1] for m in meta])  # subject id
    subjects = sorted(np.unique(subj))
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    K = len(np.unique(y))

    # class weight
    cls, cnt = np.unique(y, return_counts=True)
    w = cnt.sum() / (cnt * len(cnt))
    weight = torch.tensor(w, dtype=torch.float32).to(dev)

    folds=[]; pred_rows=[]
    for s in subjects:
        te = (subj == s)
        tr = ~te
        if te.sum()==0 or tr.sum()==0: continue

        full_tr = SeqSet(X[tr], y[tr])
        val_size = max(1, int(0.2 * len(full_tr)))
        train_size = len(full_tr) - val_size
        tr_set, va_set = random_split(full_tr, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
        te_set = SeqSet(X[te], y[te])
        tr_dl = DataLoader(tr_set, batch_size=64, shuffle=True)
        va_dl = DataLoader(va_set, batch_size=128)
        te_dl = DataLoader(te_set, batch_size=128)

        model=LSTMCls(C=X.shape[-1], K=K).to(dev)
        opt=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit=nn.CrossEntropyLoss(weight=weight)

        best_va=None; best_state=None; patience=8; bad=0; max_epochs=25
        for ep in range(max_epochs):
            model.train()
            for xb,yb in tr_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss=crit(model(xb), yb); loss.backward(); opt.step()
            ys_va, ps_va = eval_loader_preds(model, va_dl, dev)
            va_m = metrics(ys_va, ps_va)
            if (best_va is None) or (va_m["f1"] > best_va["f1"]):
                best_va = va_m; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience: break

        model.load_state_dict({k:v.to(dev) for k,v in best_state.items()})
        ys_te, ps_te = eval_loader_preds(model, te_dl, dev)
        te_m = metrics(ys_te, ps_te)
        folds.append({"subject":str(s), **te_m})

        # te 인덱스와 매칭하여 trial/subject 저장
        te_idx = np.where(te)[0]
        for idx, yt, yp in zip(te_idx, ys_te, ps_te):
            pred_rows.append({
                "subject": str(s),
                "trial": str(meta[idx][2]),
                "true": int(yt),
                "pred": int(yp)
            })

    out={"lstm_loso":{"folds":folds,
                      "mean":{"acc":float(np.mean([f["acc"] for f in folds])) if folds else 0.0,
                              "f1" :float(np.mean([f["f1"]  for f in folds])) if folds else 0.0}}}
    print(json.dumps(out, indent=2))

    import os, json as _json
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/dl_results_loso.json","w") as f: _json.dump(out,f,indent=2)
    pd.DataFrame(pred_rows).to_csv(f"{args.out}/preds_lstm.csv", index=False)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache", default="out/cache/arrays_skill_stride1_subjid.npz")
    ap.add_argument("--out", default="out/dl/lstm_skill_stride1_subjid_fixed")
    main(ap.parse_args())
