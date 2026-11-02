"""Validation-only exploratory calibration + gated CNN analysis.
No model loading, training, test access, or modification of source files.
Requires: numpy, scikit-learn, scipy.
"""
import argparse, csv, json, re
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CLASSES=('NORMAL','BACTERIAL','VIRAL')

def load(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def parse(rows, is_cnn=False):
    d={}
    for r in rows:
        label=(r['actual'] if is_cnn else r['true_class']).strip().upper()
        if label not in CLASSES: raise ValueError(f'Unknown class: {label}')
        stem=Path(r['file'].replace('\\','/')).stem if is_cnn else r['image_stem']
        key=(label,stem.lower())
        if key in d: raise ValueError(f'Duplicate identity: {key}')
        fields=('score_NORMAL','score_BACTERIAL','score_VIRAL') if is_cnn else ('prob_normal','prob_bacterial','prob_viral')
        p=np.array([float(r[k]) for k in fields],dtype=float)
        if not np.isfinite(p).all() or np.min(p)<0 or np.max(p)>1 or abs(p.sum()-1)>.002: raise ValueError(f'Invalid probabilities: {key}')
        declared=(r['predicted'] if is_cnn else r['predicted_class']).strip().upper()
        if declared != CLASSES[int(p.argmax())]: raise ValueError(f'Predicted label mismatch: {key}')
        d[key]=p
    return d

def group_id(key):
    cls,stem=key
    m=re.match(r'(person\d+)',stem,re.I)
    if m: return 'person:'+m.group(1).lower()
    # Non-person identifiers cannot establish patient identity.
    return 'unverified:'+cls+':'+stem

def temperature(p,t):
    z=np.log(np.maximum(p,1e-12))/t
    z-=z.max(axis=1,keepdims=True)
    q=np.exp(z)
    return q/q.sum(axis=1,keepdims=True)

def ece(p,y,bins=10):
    conf=p.max(axis=1); pred=p.argmax(axis=1); edges=np.linspace(0,1,bins+1)
    return float(sum(np.mean((conf>=edges[i]) & (conf<(edges[i+1] if i<bins-1 else 1.00001))) * abs(np.mean(pred[mask]==y[mask])-np.mean(conf[mask])) for i in range(bins) if np.any(mask:=((conf>=edges[i]) & (conf<(edges[i+1] if i<bins-1 else 1.00001))))))

def metrics(p,y):
    pred=p.argmax(axis=1)
    return {'n':int(len(y)), 'correct':int((pred==y).sum()), 'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)), 'recall':{c:float(np.mean(pred[y==i]==i)) if np.any(y==i) else None for i,c in enumerate(CLASSES)},'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()), 'brier':float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))), 'ece10':ece(p,y), 'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist()}

def gate(e,c,cutoff):
    out=e.copy()
    # Fixed hypothesis: consider CNN only when ensemble predicts VIRAL,
    # CNN predicts BACTERIAL, and ensemble confidence is below cutoff.
    mask=(e.argmax(axis=1)==2)&(c.argmax(axis=1)==1)&(e.max(axis=1)<cutoff)
    out[mask]=.8*e[mask]+.2*c[mask]
    return out,int(mask.sum())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    root=args.root.resolve()
    ep=root/'results/classification/ablation_uniform_evaluation/ensemble_50_50/ensemble_predictions.csv'
    cp=root/'experiments/classification_error_analysis/validation/baseline_cnn_predictions.csv'
    e=parse(load(ep)); c=parse(load(cp),True)
    if set(e)!=set(c): raise ValueError(f'Identity mismatch: ensemble_only={len(set(e)-set(c))}, cnn_only={len(set(c)-set(e))}')
    keys=sorted(e); y=np.array([CLASSES.index(k[0]) for k in keys]); pe=np.stack([e[k] for k in keys]); pc=np.stack([c[k] for k in keys]); groups=np.array([group_id(k) for k in keys])
    if len(keys)!=875 or [int(np.sum(y==i)) for i in range(3)]!=[240,417,218]: raise ValueError('Unexpected validation cohort; stop.')
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=args.seed)
    fit_idx,hold_idx=next(cv.split(pe,y,groups))
    if set(groups[fit_idx]) & set(groups[hold_idx]): raise ValueError('Group overlap')
    yf,yh=y[fit_idx],y[hold_idx]
    if min(np.bincount(yf,minlength=3))<20 or min(np.bincount(yh,minlength=3))<20: raise ValueError('Bad class distribution; try another seed')
    base_fit=metrics(pe[fit_idx],yf); base_hold=metrics(pe[hold_idx],yh)
    opt=minimize_scalar(lambda t: metrics(temperature(pe[fit_idx],t),yf)['nll'],bounds=(.5,5),method='bounded',options={'xatol':1e-5})
    temp=float(opt.x)
    cal_fit=metrics(temperature(pe[fit_idx],temp),yf); cal_hold=metrics(temperature(pe[hold_idx],temp),yh)
    # Very small pre-declared gate candidate set; selection FIT only.
    candidates=[('no_gate',None)] + [(f'gate_{v:.2f}',v) for v in (.55,.65,.75)]
    fit_results=[]
    for name,cut in candidates:
        p,n=(pe[fit_idx],0) if cut is None else gate(pe[fit_idx],pc[fit_idx],cut)
        m=metrics(p,yf)
        fit_results.append({'name':name,'cutoff':cut,'changed_eligible':n,'metrics':m})
    # Select by fit macro F1, then accuracy; prefer no_gate on ties.
    best=max(enumerate(fit_results),key=lambda im:(im[1]['metrics']['macro_f1'],im[1]['metrics']['accuracy'],-im[0]))[1]
    ph,nh=(pe[hold_idx],0) if best['cutoff'] is None else gate(pe[hold_idx],pc[hold_idx],best['cutoff'])
    selected_hold=metrics(ph,yh)
    print('VALIDATION-ONLY EXPLORATORY STUDY; NOT AN INDEPENDENT TEST')
    print(f'Validated aligned identities: {len(keys)}; groups: {len(set(groups))}')
    print(f'Fit: {len(fit_idx)} {dict(zip(CLASSES,np.bincount(yf,minlength=3).tolist()))}; Holdout: {len(hold_idx)} {dict(zip(CLASSES,np.bincount(yh,minlength=3).tolist()))}')
    print('Patient grouping uses personNNN where available; other filenames may not identify patients.')
    print('\nTEMPERATURE SCALING (fit NLL only)')
    print(f'Fit temperature: {temp:.4f}')
    for name,m in [('fit baseline',base_fit),('fit calibrated',cal_fit),('holdout baseline',base_hold),('holdout calibrated',cal_hold)]:
        print(f'  {name:19s} accuracy={m["accuracy"]*100:.2f}% macroF1={m["macro_f1"]:.4f} NLL={m["nll"]:.4f} Brier={m["brier"]:.4f} ECE10={m["ece10"]:.4f}')
    print('Temperature scaling preserves class ranking and therefore accuracy.')
    print('\nGATED CNN RULE (selection on fit subset ONLY)')
    for item in fit_results:
        m=item['metrics']; print(f'  {item["name"]:12s} eligible={item["changed_eligible"]:3d} accuracy={m["accuracy"]*100:.2f}% macroF1={m["macro_f1"]:.4f} BactR={m["recall"]["BACTERIAL"]*100:.2f}% ViralR={m["recall"]["VIRAL"]*100:.2f}%')
    print(f'Chosen on fit: {best["name"]}')
    print(f'Holdout baseline: {base_hold["correct"]}/{base_hold["n"]} acc={base_hold["accuracy"]*100:.2f}% macroF1={base_hold["macro_f1"]:.4f} BactR={base_hold["recall"]["BACTERIAL"]*100:.2f}% ViralR={base_hold["recall"]["VIRAL"]*100:.2f}%')
    print(f'Holdout selected: {selected_hold["correct"]}/{selected_hold["n"]} acc={selected_hold["accuracy"]*100:.2f}% macroF1={selected_hold["macro_f1"]:.4f} BactR={selected_hold["recall"]["BACTERIAL"]*100:.2f}% ViralR={selected_hold["recall"]["VIRAL"]*100:.2f}% eligible={nh}')
    out=root/'results/classification/calibration_decision_validation_v1'
    if out.exists(): raise FileExistsError(f'Refusing to overwrite: {out}')
    out.mkdir(parents=True)
    report={'warning':'Exploratory validation reused in previous experiments; holdout not pristine. No test accessed.', 'seed':args.seed,'temperature':temp,'n_groups':len(set(groups)),'fit_n':len(fit_idx),'holdout_n':len(hold_idx),'fit_baseline':base_fit,'fit_calibrated':cal_fit,'holdout_baseline':base_hold,'holdout_calibrated':cal_hold,'gate_fit_candidates':fit_results,'selected_gate':best['name'],'holdout_selected_gate':selected_hold,'holdout_gate_eligible':nh}
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'split_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['true_class','image_stem','group_id','split']);fs=set(fit_idx.tolist())
        for i,k in enumerate(keys):w.writerow([k[0],k[1],groups[i],'fit' if i in fs else 'holdout'])
    print(f'\nSaved: {out}');print('No training, no test access, no existing files overwritten.')

if __name__=='__main__':main()
