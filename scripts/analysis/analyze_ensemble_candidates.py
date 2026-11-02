import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np

CLASSES = ('NORMAL', 'BACTERIAL', 'VIRAL')
IDX = {x: i for i, x in enumerate(CLASSES)}

FILES = {
    'A': ('results/classification/ablation_uniform_evaluation/A_original_predictions.csv', 'ablation'),
    'B3_025': ('results/classification/ablation_uniform_evaluation/B3_alpha025_predictions.csv', 'ablation'),
    'B3_050': ('results/classification/ablation_uniform_evaluation/B3_alpha050_predictions.csv', 'ablation'),
    'CNN': ('experiments/classification_error_analysis/validation/baseline_cnn_predictions.csv', 'old'),
    'MobileNet_transfer': ('experiments/classification_error_analysis/validation/mobilenetv2_predictions.csv', 'old'),
    'MobileNet_finetuned': ('experiments/classification_error_analysis/validation/finetuned_mobilenetv2_predictions.csv', 'old'),
    'Contrast_control': ('experiments/contrast_augmentation_training/robustness_comparison/per_image_probabilities.csv', 'control'),
    'Contrast_aug': ('experiments/contrast_augmentation_training/robustness_comparison/per_image_probabilities.csv', 'contrast'),
}

def canonical(path, true):
    p = str(path).replace('\\', '/').strip()
    return (true, Path(p).stem.lower())

def load(path, typ):
    out = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if typ == 'ablation':
                truth, pathstr = r['true_class'], r['path']
                probs = [r['prob_normal'], r['prob_bacterial'], r['prob_viral']]
                pred = r['predicted_class']
            elif typ == 'old':
                truth, pathstr = r['actual'], r['file']
                probs = [r['score_NORMAL'], r['score_BACTERIAL'], r['score_VIRAL']]
                pred = r['predicted']
            else:
                truth, pathstr = r['actual'], r['file']
                probs = [r[f'{typ}_prob_normal'], r[f'{typ}_prob_bacterial'], r[f'{typ}_prob_viral']]
                pred = r[f'{typ}_prediction']
            if truth not in IDX or pred not in IDX:
                raise ValueError(f'Unknown class {truth!r} / {pred!r} in {path}')
            key = canonical(pathstr, truth)
            if key in out:
                raise ValueError(f'Duplicate identity {key} in {path}')
            arr = np.array([float(x) for x in probs], dtype=np.float64)
            if not np.isfinite(arr).all() or (arr < -1e-6).any() or abs(arr.sum()-1) > .005:
                raise ValueError(f'Invalid probabilities for {key} in {path}: {arr}')
            if int(arr.argmax()) != IDX[pred]:
                raise ValueError(f'Probability/prediction mismatch for {key} in {path}')
            out[key] = arr
    return out

def metrics(y, p):
    pred = p.argmax(axis=1)
    correct = pred == y
    recall, f1 = [], []
    for i in range(3):
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        fn = int(((pred != i) & (y == i)).sum())
        recall.append(tp / (tp+fn) if tp+fn else 0)
        f1.append(2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return int(correct.sum()), float(correct.mean()), float(np.mean(f1)), recall, correct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('.'))
    args = ap.parse_args()
    root = args.root.resolve()
    data = {}
    for name, (relative, typ) in FILES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        data[name] = load(path, typ)
        print(f'Loaded {name}: {len(data[name])} unique identities')
    keys = sorted(data['A'])
    if len(keys) != 875:
        raise ValueError(f'Expected 875 A identities, got {len(keys)}')
    for name, rows in data.items():
        if set(rows) != set(keys):
            missing, extra = set(keys)-set(rows), set(rows)-set(keys)
            raise ValueError(f'{name} identity mismatch: missing={len(missing)} extra={len(extra)}; examples {list(missing)[:3]} {list(extra)[:3]}')
    y = np.array([IDX[k[0]] for k in keys])
    print('\nClass counts:', dict(Counter(k[0] for k in keys)))
    p = {name: np.stack([rows[k] for k in keys]) for name, rows in data.items()}
    base = (p['A']+p['B3_050'])/2
    models = {'Ensemble_50_50': base, **p}
    print('\nMODEL VALIDATION METRICS')
    print(f'{"Model":<24} {"Correct":>8} {"Accuracy":>10} {"MacroF1":>9} {"NormalR":>9} {"BactR":>9} {"ViralR":>9}')
    for name, prob in models.items():
        n, acc, f1, recalls, _ = metrics(y, prob)
        print(f'{name:<24} {n:>8} {acc*100:>9.3f}% {f1:>9.5f} {recalls[0]*100:>8.2f}% {recalls[1]*100:>8.2f}% {recalls[2]*100:>8.2f}%')
    print('\nIDENTICAL PREDICTION / PROBABILITY CHECK')
    for name in p:
        if name == 'A':
            continue
        print(f'A vs {name:<22}: same argmax={int((p["A"].argmax(1)==p[name].argmax(1)).sum())}/875; max abs probability diff={np.max(np.abs(p["A"]-p[name])):.9g}')
    aok = metrics(y,p['A'])[-1]
    bok = metrics(y,p['B3_050'])[-1]
    baseok = metrics(y,base)[-1]
    bothwrong = ~aok & ~bok
    print(f'\nA & B3_050 both wrong: {int(bothwrong.sum())}; ensemble wrong: {int((~baseok).sum())}')
    print('ADDITIONAL MODEL RECOVERY (descriptive; not a deployable oracle)')
    for name in p:
        if name in ('A','B3_050','Contrast_control'):
            continue
        ok = metrics(y,p[name])[-1]
        print(f'{name:<24} fixes_both_wrong={int((bothwrong & ok).sum()):>3} fixes_ensemble_wrong={int((~baseok & ok).sum()):>3} loses_ensemble_correct={int((baseok & ~ok).sum()):>3}')
    print('\nPRE-SPECIFIED 3-MODEL ENSEMBLES: 40% A + 40% B3_050 + 20% C')
    print(f'{"Third model":<24} {"Correct":>8} {"Accuracy":>10} {"MacroF1":>9} {"BactR":>9} {"ViralR":>9} {"Delta":>7}')
    base_n = int(baseok.sum())
    for name in ('B3_025','CNN','MobileNet_transfer','MobileNet_finetuned','Contrast_aug'):
        prob = .4*p['A']+.4*p['B3_050']+.2*p[name]
        n, acc, f1, r, _ = metrics(y, prob)
        print(f'{name:<24} {n:>8} {acc*100:>9.3f}% {f1:>9.5f} {r[1]*100:>8.2f}% {r[2]*100:>8.2f}% {n-base_n:>+7}')
    print('\nAnalysis only. No files written. No training or test inference.')

if __name__ == '__main__':
    main()
