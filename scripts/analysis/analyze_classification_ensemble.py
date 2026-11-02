from pathlib import Path
import argparse, csv, json
import numpy as np

CLASSES = ('NORMAL','BACTERIAL','VIRAL')
PROBS = ('prob_normal','prob_bacterial','prob_viral')

def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    required = {'path','true_class',*PROBS}
    if not rows or not required.issubset(rows[0]): raise ValueError(f'CSV missing columns: {path}')
    keyed = {}
    for row in rows:
        true = row['true_class'].strip().upper()
        if true not in CLASSES: raise ValueError(f'Unexpected class {true}')
        # A and B3 may have different extensions (.jpeg vs .png)
        key = (true, Path(row['path'].replace('\\','/')).stem.casefold())
        if key in keyed: raise ValueError(f'Duplicate key {key} in {path}')
        probs = np.asarray([float(row[c]) for c in PROBS], dtype=np.float64)
        if not np.all(np.isfinite(probs)) or np.any(probs < 0) or abs(probs.sum()-1)>0.02: raise ValueError(f'Invalid probabilities {key}')
        keyed[key] = probs/probs.sum()
    return keyed

def metrics(y, pred):
    cm = np.zeros((3,3), dtype=int)
    for a,b in zip(y,pred): cm[a,b]+=1
    f1=[]; recall=[]
    for i in range(3):
        tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp
        precision=tp/(tp+fp) if tp+fp else 0.
        rec=tp/(tp+fn) if tp+fn else 0.
        f1.append(2*precision*rec/(precision+rec) if precision+rec else 0.)
        recall.append(rec)
    return {'accuracy': float(np.mean(y==pred)), 'macro_f1':float(np.mean(f1)), 'recall':dict(zip(CLASSES, map(float,recall))), 'confusion_matrix':cm.tolist()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path.cwd())
    args=ap.parse_args()
    root=args.root.resolve()
    folder=root/'results'/'classification'/'ablation_uniform_evaluation'
    a=read_csv(folder/'A_original_predictions.csv')
    b=read_csv(folder/'B3_alpha050_predictions.csv')
    if set(a)!=set(b):
        raise ValueError(f'Image mismatch: only A={len(set(a)-set(b))}, only B3={len(set(b)-set(a))}')
    keys=sorted(a)
    if len(keys)!=875: raise ValueError(f'Expected 875 validation samples, got {len(keys)}')
    y=np.asarray([CLASSES.index(k[0]) for k in keys]); pa=np.stack([a[k] for k in keys]); pb=np.stack([b[k] for k in keys])
    results=[]
    for w in (0.,.25,.5,.75,1.):
        pred=np.argmax(w*pa+(1-w)*pb,axis=1)
        results.append({'weight_A':w,'weight_B3':1-w, **metrics(y,pred)})
    # Descriptive sensitivity only; do not claim a new test result or choose the best on test.
    out=root/'results'/'classification'/'ensemble_weight_sensitivity_validation'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'weight_sensitivity.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=['weight_A','weight_B3','accuracy','macro_f1','recall_normal','recall_bacterial','recall_viral'])
        wr.writeheader()
        for r in results:
            wr.writerow({'weight_A':r['weight_A'],'weight_B3':r['weight_B3'],'accuracy':r['accuracy'],'macro_f1':r['macro_f1'],**{'recall_'+k.lower():v for k,v in r['recall'].items()}})
    (out/'summary.json').write_text(json.dumps({'sample_count':len(keys),'dataset':'validation','results':results,'warning':'Exploratory weight sensitivity on the same validation set. Best result is optimistically biased; do not call it independent test accuracy.'},indent=2),encoding='utf-8')
    print('ENSEMBLE WEIGHT SENSITIVITY (VALIDATION ONLY)')
    for r in results:
        print(f"A={r['weight_A']:.2f} B3={r['weight_B3']:.2f} accuracy={r['accuracy']:.4%} macro_F1={r['macro_f1']:.4f} viral_recall={r['recall']['VIRAL']:.4%}")
    print('Saved:',out)
    print('No training or test evaluation performed.')

if __name__=='__main__': main()
