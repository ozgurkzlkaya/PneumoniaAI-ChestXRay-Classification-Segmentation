"""Validation-only Stage 2 edge occlusion sensitivity. No training, no test access."""
import argparse, csv, json
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

SIZE=224

def read_image(path):
    raw=tf.io.read_file(str(path))
    im=tf.io.decode_image(raw,channels=3,expand_animations=False)
    im.set_shape((None,None,3))
    return tf.image.resize(tf.cast(im,tf.float32),(SIZE,SIZE)).numpy()

def make_variant(images, kind, neutral):
    if kind=='original':return images
    n=images.shape[1]; w=int(round(n*float(kind.split('_')[1])/100))
    mask=np.ones((n,n,1),dtype=np.float32)
    mask[:w,:,:]=0;mask[-w:,:,:]=0;mask[:,:w,:]=0;mask[:,-w:,:]=0
    return images*mask+neutral*(1-mask)

def metrics(y, p):
    pred=(p>=.5).astype(int)
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'bacterial_recall':float(recall_score(y,pred,pos_label=0,zero_division=0)),
            'viral_recall':float(recall_score(y,pred,pos_label=1,zero_division=0)),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--root',default='.')
    a.add_argument('--previous-run',required=True)
    a.add_argument('--error-run',required=True)
    a.add_argument('--batch-size',type=int,default=16)
    a.add_argument('--neutral',type=float,default=127.5)
    a.add_argument('--output')
    a=a.parse_args()
    root=Path(a.root).resolve(); model_path=(root/a.previous_run/'stage2_selected.keras').resolve()
    csvpath=(root/a.error_run/'per_image_predictions.csv').resolve()
    if not model_path.is_file() or not csvpath.is_file():raise FileNotFoundError(f'Model or CSV missing: {model_path}, {csvpath}')
    with csvpath.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    rows=[r for r in rows if r['true_label'].upper() in ('BACTERIAL','VIRAL')]
    if not rows:raise RuntimeError('No pneumonia rows in validation CSV')
    paths=[];labels=[];expected=[]
    for r in rows:
        p=Path(r['relative_path'])
        if not p.is_absolute():p=root/p
        p=p.resolve()
        if 'validation' not in [s.lower() for s in p.parts]:raise RuntimeError('Non-validation path: '+str(p))
        if not p.is_file():raise FileNotFoundError(p)
        paths.append(p);labels.append(1 if r['true_label'].upper()=='VIRAL' else 0)
        expected.append(float(r['stage2_baseline_viral_prob']))
    y=np.asarray(labels,dtype=np.int32);expected=np.asarray(expected)
    print('Loading:',model_path,flush=True)
    model=tf.keras.models.load_model(model_path,compile=False)
    variants=['original','edge_5','edge_10','edge_15','edge_20']
    probabilities={k:[] for k in variants}
    for start in range(0,len(paths),a.batch_size):
        imgs=np.stack([read_image(p) for p in paths[start:start+a.batch_size]],axis=0)
        for k in variants:
            x=make_variant(imgs,k,a.neutral)
            pr=np.asarray(model(x,training=False))
            if pr.shape[1]!=2:raise RuntimeError('Stage2 output must have 2 classes')
            probabilities[k].extend(pr[:,1].tolist())
        print(f'Processed {min(start+a.batch_size,len(paths))}/{len(paths)}',flush=True)
    probabilities={k:np.asarray(v) for k,v in probabilities.items()}
    err=np.abs(probabilities['original']-expected)
    print(f'CSV baseline consistency: max_abs={err.max():.6f}, mean_abs={err.mean():.6f}, over_0.025={int((err>.025).sum())}',flush=True)
    if err.max()>.025:
        raise RuntimeError('Original prediction does not match CSV; aborting before results. Check CSV/model/split correspondence.')
    orig=probabilities['original'];orig_pred=orig>=.5
    report={'split':'validation','count':len(paths),'neutral_pixel':a.neutral,'note':'Edge occlusion is an out-of-distribution sensitivity probe, NOT a lung segmentation or clinical localization measure.',
            'baseline_csv_max_abs_difference':float(err.max()),'variants':{}}
    for k,p in probabilities.items():
        m=metrics(y,p)
        m.update({'mean_abs_viral_probability_change':float(np.mean(np.abs(p-orig))),
                  'median_abs_viral_probability_change':float(np.median(np.abs(p-orig))),
                  'prediction_flip_count':int(np.sum((p>=.5)!=orig_pred)),
                  'prediction_flip_fraction':float(np.mean((p>=.5)!=orig_pred))})
        report['variants'][k]=m
        print(f"{k:10} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} Brec={m['bacterial_recall']:.4f} Vrec={m['viral_recall']:.4f} flips={m['prediction_flip_count']} mean_dP={m['mean_abs_viral_probability_change']:.4f}",flush=True)
    out=Path(a.output).resolve() if a.output else root/'experiments'/'two_stage_classification'/f'edge_sensitivity_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'per_image_results.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['relative_path','true_label','csv_baseline_viral_prob']+[f'{k}_viral_prob' for k in variants]+[f'{k}_flipped' for k in variants[1:]])
        for i,p in enumerate(paths):
            w.writerow([str(p.relative_to(root)), 'VIRAL' if y[i] else 'BACTERIAL',float(expected[i])]+[float(probabilities[k][i]) for k in variants]+[int((probabilities[k][i]>=.5)!=orig_pred[i]) for k in variants[1:]])
    print('Saved:',out,flush=True)
if __name__=='__main__':main()
