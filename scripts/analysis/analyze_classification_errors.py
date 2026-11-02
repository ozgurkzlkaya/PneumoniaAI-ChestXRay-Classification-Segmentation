"""PneumoniaAI: paired validation error analysis. No training or test evaluation."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

CLASSES = ('NORMAL', 'BACTERIAL', 'VIRAL')
PROBS = ('prob_normal', 'prob_bacterial', 'prob_viral')


def read_predictions(path):
    result = {}
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'path', 'true_class', *PROBS}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'Missing columns in {path}: {required - set(reader.fieldnames or [])}')
        for row in reader:
            true = row['true_class'].strip().upper()
            if true not in CLASSES:
                raise ValueError(f'Unknown class {true!r}')
            # JPEG and PNG are two representations of the same sample.
            key = (true, Path(row['path'].replace('\\', '/')).stem.lower())
            if key in result:
                raise ValueError(f'Duplicate image key: {key}')
            p = np.array([float(row[c]) for c in PROBS], dtype=float)
            if not np.isfinite(p).all() or np.any(p < 0) or abs(float(p.sum()) - 1) > 0.02:
                raise ValueError(f'Invalid probabilities for {key}: {p}')
            result[key] = (row['path'], p / p.sum())
    return result


def metrics(rows, name):
    correct = sum(r[f'{name}_correct'] for r in rows)
    recalls = {}
    for cls in CLASSES:
        group = [r for r in rows if r['true_class'] == cls]
        recalls[cls] = round(sum(r[f'{name}_correct'] for r in group) / len(group), 6) if group else None
    return {'accuracy': round(correct / len(rows), 6), 'correct': correct, 'recall': recalls}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    base = root / 'results' / 'classification' / 'ablation_uniform_evaluation'
    a = read_predictions(base / 'A_original_predictions.csv')
    b = read_predictions(base / 'B3_alpha050_predictions.csv')
    if set(a) != set(b):
        only_a, only_b = sorted(set(a)-set(b)), sorted(set(b)-set(a))
        raise ValueError(f'Image sets differ: only A={len(only_a)} {only_a[:3]}, only B3={len(only_b)} {only_b[:3]}')
    if len(a) != 875:
        raise ValueError(f'Expected 875 paired validation images, got {len(a)}')
    rows = []
    for (truth, stem) in sorted(a):
        apath, pa = a[(truth, stem)]
        bpath, pb = b[(truth, stem)]
        pe = (pa + pb) / 2
        ai, bi, ei = (int(np.argmax(p)) for p in (pa, pb, pe))
        record = {'true_class': truth, 'image_stem': stem, 'original_path': apath, 'masked_path': bpath,
                  'a_prediction': CLASSES[ai], 'b3_prediction': CLASSES[bi], 'ensemble_prediction': CLASSES[ei],
                  'a_correct': int(CLASSES[ai] == truth), 'b3_correct': int(CLASSES[bi] == truth),
                  'ensemble_correct': int(CLASSES[ei] == truth),
                  'a_confidence': round(float(pa[ai]), 6), 'b3_confidence': round(float(pb[bi]), 6),
                  'ensemble_confidence': round(float(pe[ei]), 6),
                  'a_true_probability': round(float(pa[CLASSES.index(truth)]), 6),
                  'b3_true_probability': round(float(pb[CLASSES.index(truth)]), 6),
                  'ensemble_true_probability': round(float(pe[CLASSES.index(truth)]), 6),
                  'models_disagree': int(ai != bi),
                  'error_category': ('both_correct' if ai == bi == CLASSES.index(truth) else
                                     'only_a_correct' if ai == CLASSES.index(truth) else
                                     'only_b3_correct' if bi == CLASSES.index(truth) else 'both_wrong')}
        rows.append(record)
    out = root / 'results' / 'classification' / 'paired_error_analysis_validation'
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out / 'paired_predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    for category in ('only_a_correct', 'only_b3_correct', 'both_wrong'):
        subset = [r for r in rows if r['error_category'] == category]
        with (out / f'{category}.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(subset)
    categories = Counter(r['error_category'] for r in rows)
    summary = {'n': len(rows), 'categories': dict(categories), 'models_disagree': sum(r['models_disagree'] for r in rows),
               'a': metrics(rows, 'a'), 'b3': metrics(rows, 'b3'), 'ensemble_50_50': metrics(rows, 'ensemble'),
               'ensemble_corrects_both_wrong': sum(r['ensemble_correct'] for r in rows if r['error_category']=='both_wrong'),
               'ensemble_loses_both_correct': sum(1-r['ensemble_correct'] for r in rows if r['error_category']=='both_correct'),
               'bacterial_as_viral': {name: sum(r[f'{name}_prediction']=='VIRAL' for r in rows if r['true_class']=='BACTERIAL') for name in ('a','b3','ensemble')},
               'viral_as_bacterial': {name: sum(r[f'{name}_prediction']=='BACTERIAL' for r in rows if r['true_class']=='VIRAL') for name in ('a','b3','ensemble')},
               'high_confidence_wrong_90pct': {name: sum(not r[f'{name}_correct'] and r[f'{name}_confidence']>=0.9 for r in rows) for name in ('a','b3','ensemble')},
               'note': 'Exploratory analysis of the same validation set used for model and ensemble selection; no independent test.'}
    with (out / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('PAIRED CLASSIFICATION ERROR ANALYSIS (VALIDATION ONLY)')
    print('Samples:', len(rows), '| Disagreements:', summary['models_disagree'])
    for name in ('a', 'b3', 'ensemble_50_50'):
        print(f'{name}: {summary[name]["correct"]}/{len(rows)} = {summary[name]["accuracy"]*100:.4f}%')
    print('Error categories:', dict(categories))
    print('BACTERIAL -> VIRAL:', summary['bacterial_as_viral'])
    print('VIRAL -> BACTERIAL:', summary['viral_as_bacterial'])
    print('Wrong predictions with confidence >=90%:', summary['high_confidence_wrong_90pct'])
    print('Saved:', out)
    print('No model training or test evaluation performed.')


if __name__ == '__main__':
    main()
