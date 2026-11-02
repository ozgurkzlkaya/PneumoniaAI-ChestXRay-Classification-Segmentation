import argparse, csv, json, math, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES=['NORMAL','BACTERIAL','VIRAL']

def find_backbone(model):
    candidates=[l for l in model.layers if isinstance(l,tf.keras.Model)]
    if not candidates: raise RuntimeError('Ic ice backbone bulunamadi. model.summary() ciktisini paylasin.')
    return next((l for l in candidates if 'mobilenet' in l.name.lower()),candidates[0])

def find_conv(backbone):
    convs=[l for l in backbone.layers if isinstance(l,tf.keras.layers.Conv2D)]
    if not convs: raise RuntimeError('Backbone icinde Conv2D bulunamadi')
    return convs[-1]

def make_cam_model(model):
    backbone=find_backbone(model); conv=find_conv(backbone)
    extractor=tf.keras.Model(backbone.input,[conv.output,backbone.output])
    inp=tf.keras.Input(shape=model.input_shape[1:]); x=inp; activation=None; seen=False
    for layer in model.layers:
        if isinstance(layer,tf.keras.layers.InputLayer): continue
        if layer is backbone:
            activation,x=extractor(x,training=False); seen=True
        else:
            x=layer(x,training=False)
    if not seen: raise RuntimeError('Backbone baglanamadi')
    return tf.keras.Model(inp,[activation,x]),conv.name,backbone.name

def image_input(path, size):
    img=Image.open(path).convert('RGB').resize((size[1],size[0]))
    arr=np.asarray(img,dtype=np.float32)
    # MobileNetV2 pipeline generally expects raw [0,255] if preprocess is in model or dataset.
    # Script tests alternative preprocessing when requested.
    return arr

def cam_predict(cam_model, image, mode, class_idx):
    x=image.copy()
    if mode=='mobilenet': x=tf.keras.applications.mobilenet_v2.preprocess_input(x)
    elif mode=='unit': x=x/255.0
    x=tf.convert_to_tensor(x[None],tf.float32)
    with tf.GradientTape() as tape:
        act,pred=cam_model(x,training=False)
        if pred.shape[-1]==1:
            score=pred[:,0] if class_idx==1 else 1.0-pred[:,0]
        else: score=pred[:,class_idx]
    grad=tape.gradient(score,act)
    if grad is None: raise RuntimeError('Gradient hesaplanamadi; model graph kontrol edilmeli')
    weights=tf.reduce_mean(grad,axis=(1,2),keepdims=True)
    heat=tf.nn.relu(tf.reduce_sum(act*weights,axis=-1))[0]
    heat=heat/(tf.reduce_max(heat)+1e-8)
    return np.asarray(heat),np.asarray(pred[0])

def find_csv(folder):
    p=Path(folder)/'per_image_predictions.csv'
    if not p.exists(): raise FileNotFoundError(f'CSV bulunamadi: {p}')
    return p

def choose_samples(csvpath, limit, seed):
    with open(csvpath,encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    required={'true_label','baseline_050_pred','stage2_baseline_viral_prob','relative_path'}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError('CSV sutunlari uyumsuz: '+str(sorted(rows[0]) if rows else 'EMPTY'))
    buckets={k:[] for k in ('BACTERIAL_correct','VIRAL_correct','BACTERIAL_to_VIRAL','VIRAL_to_BACTERIAL','boundary')}
    for r in rows:
        true=r['true_label'].strip().upper()
        pred=r['baseline_050_pred'].strip().upper()
        if true not in ('BACTERIAL','VIRAL'): continue
        if pred==true: buckets[f'{true}_correct'].append(r)
        elif pred in ('BACTERIAL','VIRAL'): buckets[f'{true}_to_{pred}'].append(r)
        p=float(r['stage2_baseline_viral_prob'])
        if abs(p-0.5)<=0.04: buckets['boundary'].append(r)
    rng=random.Random(seed)
    selected=[]
    seen=set()
    for group,items in buckets.items():
        rng.shuffle(items)
        count=0
        for r in items:
            path=r['relative_path']
            if path in seen: continue
            seen.add(path); selected.append((group,r)); count+=1
            if count>=limit:break
        print(f'{group}: available={len(items)} selected={count}')
    if not selected:raise RuntimeError('Hicbir uygun BACTERIAL/VIRAL ornegi bulunamadi; islem durduruldu.')
    return selected

def get_path(r,root):
    for k in ('relative_path','path','image_path','filepath','file_path','image','filename'):
        v=r.get(k)
        if v:
            p=Path(v)
            if p.exists():return p
            if (root/p).exists():return root/p
    # Dataset split must be validation; do not silently use other splits.
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--previous-run',required=True)
    ap.add_argument('--error-run',required=True)
    ap.add_argument('--model',choices=['baseline','augmentation'],default='baseline')
    ap.add_argument('--improvement-run')
    ap.add_argument('--input-mode',choices=['raw','mobilenet','unit'],default='mobilenet')
    ap.add_argument('--per-group',type=int,default=5)
    ap.add_argument('--output')
    args=ap.parse_args()
    root=Path(args.root).resolve(); prev=(root/args.previous_run).resolve(); err=(root/args.error_run).resolve()
    if args.model=='baseline': ckpt=prev/'stage2_selected.keras'
    else:
        if not args.improvement_run: ap.error('--improvement-run gerekli')
        ckpt=(root/args.improvement_run/'low_lr_intensity_aug'/'best.keras').resolve()
    if not ckpt.exists():raise FileNotFoundError(ckpt)
    from datetime import datetime
    out=Path(args.output).resolve() if args.output else root/'experiments'/'two_stage_classification'/f'gradcam_pneumonia_{args.model}_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(parents=True,exist_ok=True)
    print('Loading',ckpt)
    model=tf.keras.models.load_model(ckpt,compile=False)
    cam_model,conv_name,backbone_name=make_cam_model(model)
    print('Backbone:',backbone_name,'CAM layer:',conv_name,'input:',model.input_shape,'output:',model.output_shape)
    selected=choose_samples(find_csv(err),args.per_group,42)
    size=(int(model.input_shape[1]),int(model.input_shape[2])); saved=[]; skipped=[]
    for i,(group,r) in enumerate(selected):
        path=get_path(r,root)
        if path is not None and 'validation' not in [part.lower() for part in path.parts]:
            raise RuntimeError('Validation disi goruntu secildi: '+str(path))
        if path is None:
            skipped.append({'group':group,'reason':'path_not_found','row':r});continue
        try:
            img=image_input(path,size)
            heat,pred=cam_predict(cam_model,img,args.input_mode,1) # VIRAL: stage2 class 1
            expected=float(r['stage2_baseline_viral_prob']) if args.model=='baseline' else float(r['stage2_augmentation_viral_prob'])
            observed=float(pred[1])
            if abs(observed-expected)>0.025:
                raise RuntimeError(f'Preprocessing / model uyumsuzlugu: CSV={expected:.4f}, CAM={observed:.4f}. --input-mode raw veya mobilenet deneyin; tutarlilik saglanmadan yorumlamayin.')
            heat=tf.image.resize(heat[...,None],size).numpy()[...,0]
            fig,axs=plt.subplots(1,3,figsize=(10,3.5))
            axs[0].imshow(img.astype('uint8'));axs[0].set_title('Original X-Ray')
            axs[1].imshow(heat,cmap='jet',vmin=0,vmax=1);axs[1].set_title('Grad-CAM (VIRAL)')
            axs[2].imshow(img.astype('uint8'));axs[2].imshow(heat,cmap='jet',alpha=.38,vmin=0,vmax=1);axs[2].set_title('Overlay')
            for ax in axs:ax.axis('off')
            fig.suptitle(f'{group} | {path.name} | output={np.round(pred,3).tolist()}',fontsize=9)
            fig.tight_layout()
            name=f'{i:03d}_{group}_{path.stem[:35]}.png';fig.savefig(out/name,dpi=130);plt.close(fig)
            saved.append({'group':group,'image':str(path),'file':name,'probabilities':pred.tolist()})
        except Exception as e:
            if 'Preprocessing / model uyumsuzlugu' in str(e): raise
            skipped.append({'group':group,'image':str(path),'reason':str(e)})
    with open(out/'summary.json','w',encoding='utf8') as f:json.dump({'model':str(ckpt),'input_mode':args.input_mode,'backbone':backbone_name,'cam_layer':conv_name,'saved':saved,'skipped':skipped,'warning':'Exploratory visualization only; no clinical localization claim; no test accessed.'},f,indent=2,ensure_ascii=False)
    print('Saved',len(saved),'Skipped',len(skipped),'Output:',out)
    if skipped: print('Skipped details:',skipped[:5])
    if not saved: print('CSV yol/etiket kolonlarini gormek icin ilk satiri ve kolonlari paylasin.')

if __name__=='__main__':main()
