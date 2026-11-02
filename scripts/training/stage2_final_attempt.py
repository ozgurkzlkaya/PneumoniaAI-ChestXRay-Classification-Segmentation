"""One controlled Stage2 head-only focal-loss experiment. Validation only; no test access."""
import argparse, csv, json, random
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

SIZE=(224,224)
CLASSES=('NORMAL','BACTERIAL','VIRAL')

def gather(folder):
    paths=[]; labels=[]
    for k,c in enumerate(CLASSES):
        d=folder/c
        if not d.is_dir(): raise FileNotFoundError(d)
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in ('.jpeg','.jpg','.png'):
                paths.append(str(p.resolve())); labels.append(k)
    return np.array(paths),np.array(labels,dtype=np.int32)

def dataset(paths,labels,batch,shuffle=False):
    def decode(path,label):
        im=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False)
        im.set_shape((None,None,3))
        return tf.image.resize(tf.cast(im,tf.float32),SIZE),label
    ds=tf.data.Dataset.from_tensor_slices((paths,labels))
    if shuffle:ds=ds.shuffle(len(paths),seed=42,reshuffle_each_iteration=True)
    return ds.map(decode,num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)

@tf.keras.utils.register_keras_serializable(package='PneumoniaAI')
class SparseFocal(tf.keras.losses.Loss):
    def __init__(self,gamma=1.5,name='sparse_focal',**kwargs):
        super().__init__(name=name,**kwargs);self.gamma=gamma
    def call(self,y_true,y_pred):
        y=tf.cast(tf.reshape(y_true,[-1]),tf.int32)
        p=tf.clip_by_value(tf.gather(y_pred,y,batch_dims=1),1e-7,1.-1e-7)
        return -tf.pow(1.-p,self.gamma)*tf.math.log(p)
    def get_config(self):return {**super().get_config(),'gamma':self.gamma}

class Selection(tf.keras.callbacks.Callback):
    def __init__(self,val_ds,y,folder,base_f1,base_rec):
        super().__init__();self.ds=val_ds;self.y=y;self.folder=folder
        self.base_f1=base_f1;self.base_rec=base_rec;self.best=-1;self.history=[]
    def on_epoch_end(self,epoch,logs=None):
        p=self.model.predict(self.ds,verbose=0)[:,1];pred=(p>=.5).astype(int)
        f=float(f1_score(self.y,pred,average='macro')); rec=recall_score(self.y,pred,labels=[0,1],average=None,zero_division=0)
        acc=float(accuracy_score(self.y,pred))
        # Require no class recall to fall more than 3 percentage points from baseline.
        eligible=bool(np.all(rec>=self.base_rec-.03))
        row={'epoch':epoch+1,'val_acc':acc,'val_macro_f1':f,'bacterial_recall':float(rec[0]),'viral_recall':float(rec[1]),'eligible':eligible}
        self.history.append(row)
        print('CHECK',json.dumps(row),flush=True)
        if eligible and f>self.best:
            self.best=f;self.model.save(self.folder/'candidate_best.keras')
        (self.folder/'epoch_metrics.json').write_text(json.dumps(self.history,indent=2))

def metrics(y,p):
    pred=np.argmax(p,axis=1);rec=recall_score(y,pred,labels=[0,1],average=None,zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),
            'bacterial_recall':float(rec[0]),'viral_recall':float(rec[1]),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def combined(y,p1,p2):
    q=np.stack([p1[:,0],p1[:,1]*p2[:,0],p1[:,1]*p2[:,1]],axis=1)
    pred=q.argmax(axis=1)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),
            'recall':dict(zip(CLASSES,map(float,recall_score(y,pred,labels=[0,1,2],average=None,zero_division=0)))),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--previous-run',required=True)
    ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--batch-size',type=int,default=16)
    args=ap.parse_args();tf.keras.utils.set_random_seed(42);random.seed(42);np.random.seed(42)
    root=Path(args.root).resolve();prev=(root/args.previous_run).resolve()
    stage1=prev/'stage1_selected.keras';stage2=prev/'stage2_selected.keras'
    for p in [stage1,stage2]:
        if not p.is_file():raise FileNotFoundError(p)
    train_p,train_y=gather(root/'dataset'/'train');val_p,val_y=gather(root/'dataset'/'validation')
    train_keep=train_y!=0;val_keep=val_y!=0
    tr_y=(train_y[train_keep]==2).astype('int32');v_y=(val_y[val_keep]==2).astype('int32')
    tr_ds=dataset(train_p[train_keep],tr_y,args.batch_size,True)
    v_ds=dataset(val_p[val_keep],v_y,args.batch_size)
    all_val_ds=dataset(val_p,np.zeros(len(val_p),dtype=np.int32),args.batch_size)
    out=root/'experiments'/'two_stage_classification'/f'stage2_final_attempt_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(parents=True);print('Output:',out,flush=True)
    m2=tf.keras.models.load_model(stage2,compile=False)
    base_prob=m2.predict(v_ds,verbose=0);base=metrics(v_y,base_prob)
    print('STAGE2 BASELINE',json.dumps(base),flush=True)
    m1=tf.keras.models.load_model(stage1,compile=False)
    p1=m1.predict(all_val_ds,verbose=0)
    all_base_p2=m2.predict(all_val_ds,verbose=0)
    base_combined=combined(val_y,p1,all_base_p2)
    print('THREE-CLASS BASELINE',json.dumps(base_combined),flush=True)
    backbone=next((l for l in m2.layers if isinstance(l,tf.keras.Model) and 'mobilenetv2' in l.name.lower()),None)
    if backbone is None:raise RuntimeError('MobileNetV2 backbone not found; refusing to train')
    backbone.trainable=False
    # Head-only adaptation: no architectural changes, no backbone fine-tuning.
    m2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),loss=SparseFocal(1.5),metrics=['accuracy'])
    counts=np.bincount(tr_y,minlength=2);weights={i:float(len(tr_y)/(2*counts[i])) for i in range(2)}
    cb=Selection(v_ds,v_y,out,base['macro_f1'],np.array([base['bacterial_recall'],base['viral_recall']]))
    m2.fit(tr_ds,validation_data=v_ds,epochs=args.epochs,class_weight=weights,
           callbacks=[cb,tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=False)],verbose=2)
    report={'stage2_baseline':base,'three_class_baseline':base_combined,'experiment':'frozen MobileNetV2 head-only sparse focal gamma=1.5, Adam 2e-5, class weights',
            'selection':'best validation macro-F1 with each class recall no more than 3pp below Stage2 baseline',
            'note':'Validation used repeatedly for model selection; exploratory, not independent evidence. Test untouched.'}
    best=out/'candidate_best.keras'
    if best.exists():
        candidate=tf.keras.models.load_model(best,compile=False)
        cp=candidate.predict(v_ds,verbose=0);report['stage2_candidate']=metrics(v_y,cp)
        cpall=candidate.predict(all_val_ds,verbose=0);report['three_class_candidate']=combined(val_y,p1,cpall)
        b=base_combined;c=report['three_class_candidate']
        report['promote_recommended']=bool(c['macro_f1']>=b['macro_f1']+.005 and
             c['recall']['NORMAL']>=b['recall']['NORMAL']-.01 and
             c['recall']['BACTERIAL']>=b['recall']['BACTERIAL']-.03 and
             c['recall']['VIRAL']>=b['recall']['VIRAL']-.03)
    else:report['promote_recommended']=False;report['reason']='No candidate satisfied class-recall guardrails'
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('FINAL SUMMARY',json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    print('Saved:',out,flush=True)
if __name__=='__main__':main()
