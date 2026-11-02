from pathlib import Path
import argparse, csv, json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

CLASSES = ('NORMAL', 'BACTERIAL', 'VIRAL')
PROBS = ('prob_normal', 'prob_bacterial', 'prob_viral')

def load_predictions(path):
    result = {}
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        needed = {'path', 'true_class', *PROBS}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {needed - set(reader.fieldnames or [])}')
        for row in reader:
            key = (row['true_class'].strip().upper(), Path(row['path']).stem.lower())
            if key in result:
                raise ValueError(f'Duplicate image key in {path}: {key}')
            p = np.array([float(row[c]) for c in PROBS], dtype=float)
            if not np.isfinite(p).all() or (p < 0).any() or abs(p.sum()-1) > .02:
                raise ValueError(f'Invalid probabilities: {key} {p}')
            if key[0] not in CLASSES:
                raise ValueError(f'Unknown class: {key[0]}')
            result[key] = p
    return result

def main():
    parser = argparse.ArgumentParser(description='Read-only 50/50 A+B3 ensemble on matched validation images')
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    directory = args.root / 'results' / 'classification' / 'ablation_uniform_evaluation'
    a = load_predictions(directory / 'A_original_predictions.csv')
    b = load_predictions(directory / 'B3_alpha050_predictions.csv')
    if set(a) != set(b):
        only_a = list(set(a) - set(b))[:5]
        only_b = list(set(b) - set(a))[:5]
        raise ValueError(f'Dataset identity mismatch. A only: {only_a}; B3 only: {only_b}')
    keys = sorted(a)
    if len(keys) != 875:
        raise ValueError(f'Expected 875 matched validation images, found {len(keys)}')
    y_true = np.array([CLASSES.index(k[0]) for k in keys])
    scores = {
        'A_original': np.stack([a[k] for k in keys]),
        'B3_alpha050': np.stack([b[k] for k in keys]),
        'Ensemble_50_50': np.stack([(a[k]+b[k])/2 for k in keys]),
    }
    print(f'Validated matching images: {len(keys)}')
    results = []
    for name, p in scores.items():
        y_pred = p.argmax(axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        results.append({'model': name, 'n': len(keys), 'accuracy': accuracy, 'macro_f1': macro_f1})
        print(f'\n{name}: accuracy={accuracy:.4%}, macro F1={macro_f1:.5f}')
        print('Confusion matrix (NORMAL, BACTERIAL, VIRAL):')
        print(cm)
        print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0))
    out = directory / 'ensemble_50_50'
    if out.exists():
        raise FileExistsError(f'Output already exists; nothing overwritten: {out}')
    out.mkdir(parents=False)
    with (out / 'summary.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model','n','accuracy','macro_f1'])
        w.writeheader(); w.writerows(results)
    with (out / 'ensemble_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['true_class','image_stem','predicted_class',*PROBS])
        for key, p in zip(keys, scores['Ensemble_50_50']):
            w.writerow([key[0],key[1],CLASSES[p.argmax()],*map(float,p)])
    with (out / 'ensemble_confusion_matrix.csv').open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f);w.writerow(['true_class',*CLASSES]);
        cm=confusion_matrix(y_true,scores['Ensemble_50_50'].argmax(axis=1),labels=[0,1,2])
        for cls, row in zip(CLASSES,cm):w.writerow([cls,*map(int,row)])
    print(f'\nSaved new results: {out}')

if __name__ == '__main__':
    main()
