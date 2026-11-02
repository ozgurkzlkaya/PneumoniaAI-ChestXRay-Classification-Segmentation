"""Validation-only descriptive error, confidence, and model-disagreement analysis.

Run: python analyze_error_confidence.py --root .
Requires only Python standard library. No model inference, tuning, or test access.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ('NORMAL', 'BACTERIAL', 'VIRAL')
PROBS_E = ('prob_normal', 'prob_bacterial', 'prob_viral')
PROBS_C = ('score_NORMAL', 'score_BACTERIAL', 'score_VIRAL')
BINS = ((0, .5, '<50%'), (.5, .7, '50-70%'), (.7, .9, '70-90%'), (.9, 1.000001, '90-100%'))


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def parse_prob(row, fields, origin):
    values = [float(row[x]) for x in fields]
    if any(not (0 <= p <= 1) for p in values) or abs(sum(values) - 1) > .002:
        raise ValueError(f'Invalid probabilities in {origin}: {values}')
    return values


def identity(cls, stem):
    return cls, stem.lower()


def load_ensemble(path):
    records = {}
    for row in read_csv(path):
        cls = row['true_class'].strip().upper()
        if cls not in CLASSES:
            raise ValueError(f'Unexpected true class {cls}')
        key = identity(cls, row['image_stem'].strip())
        if key in records:
            raise ValueError(f'Duplicate ensemble key: {key}')
        probs = parse_prob(row, PROBS_E, path)
        pred = CLASSES[max(range(3), key=lambda i: probs[i])]
        if pred != row['predicted_class'].strip().upper():
            raise ValueError(f'Ensemble predicted_class mismatch: {key}')
        records[key] = probs
    return records


def load_cnn(path):
    records = {}
    for row in read_csv(path):
        cls = row['actual'].strip().upper()
        if cls not in CLASSES:
            raise ValueError(f'Unexpected CNN true class {cls}')
        stem = Path(row['file'].replace('\\', '/')).stem.lower()
        key = identity(cls, stem)
        if key in records:
            raise ValueError(f'Duplicate CNN key: {key}')
        probs = parse_prob(row, PROBS_C, path)
        pred = CLASSES[max(range(3), key=lambda i: probs[i])]
        if pred != row['predicted'].strip().upper():
            raise ValueError(f'CNN predicted_class mismatch: {key}')
        records[key] = probs
    return records


def pct(n, total):
    return round(100 * n / total, 2) if total else None


def summarize(rows):
    total = len(rows)
    correct = sum(r['ensemble_correct'] for r in rows)
    errors = [r for r in rows if not r['ensemble_correct']]
    print(f'Validation identities: {total}; ensemble correct: {correct}; errors: {len(errors)}')
    print(f'Ensemble accuracy: {pct(correct, total)}%')
    print('\nERROR CONFIDENCE BINS (confidence = largest ensemble probability)')
    for low, high, label in BINS:
        group = [r for r in rows if low <= r['ensemble_confidence'] < high]
        wrong = sum(not r['ensemble_correct'] for r in group)
        print(f'  {label:8s} total={len(group):3d} errors={wrong:3d} error_rate={pct(wrong,len(group))}%')
    high_wrong = [r for r in errors if r['ensemble_confidence'] >= .9]
    print(f'High-confidence errors (>=90%): {len(high_wrong)} / {len(errors)} ({pct(len(high_wrong),len(errors))}% of errors)')
    print('\nBACTERIAL <-> VIRAL ERRORS')
    for true, pred in [('BACTERIAL', 'VIRAL'), ('VIRAL', 'BACTERIAL')]:
        group = [r for r in rows if r['true_class'] == true and r['ensemble_prediction'] == pred]
        second = sum(r['ensemble_second_class'] == true for r in group)
        high = sum(r['ensemble_confidence'] >= .9 for r in group)
        margins = sorted(r['ensemble_margin'] for r in group)
        median = round(margins[len(margins)//2], 4) if margins else None
        print(f'  {true} -> {pred}: n={len(group)}, true_second={second}, high_confidence={high}, median_margin={median}')
    print('\nENSEMBLE vs CNN DISAGREEMENTS')
    disagreed = [r for r in rows if r['ensemble_prediction'] != r['cnn_prediction']]
    both_wrong = sum(not r['ensemble_correct'] and not r['cnn_correct'] for r in disagreed)
    e_only = sum(r['ensemble_correct'] and not r['cnn_correct'] for r in disagreed)
    c_only = sum(r['cnn_correct'] and not r['ensemble_correct'] for r in disagreed)
    print(f'  disagreement={len(disagreed)}, ensemble_only_correct={e_only}, cnn_only_correct={c_only}, both_wrong={both_wrong}')
    print('  By true class:')
    for cls in CLASSES:
        sub = [r for r in disagreed if r['true_class'] == cls]
        e = sum(r['ensemble_correct'] for r in sub)
        c = sum(r['cnn_correct'] for r in sub)
        print(f'    {cls:10s}: n={len(sub):3d}, ensemble_correct={e:3d}, cnn_correct={c:3d}')
    print('\nNo threshold optimization, model changes, or test access performed.')
    return {
        'n': total, 'correct': correct, 'errors': len(errors),
        'confidence_bins': [dict(label=label, n=sum(low <= r['ensemble_confidence'] < high for r in rows),
                                 errors=sum(low <= r['ensemble_confidence'] < high and not r['ensemble_correct'] for r in rows))
                            for low, high, label in BINS],
        'high_confidence_errors': len(high_wrong),
        'disagreements': len(disagreed), 'ensemble_only_correct_in_disagreements': e_only,
        'cnn_only_correct_in_disagreements': c_only, 'both_wrong_in_disagreements': both_wrong,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    root = args.root.resolve()
    ensemble_path = root / 'results/classification/ablation_uniform_evaluation/ensemble_50_50/ensemble_predictions.csv'
    cnn_path = root / 'experiments/classification_error_analysis/validation/baseline_cnn_predictions.csv'
    ens = load_ensemble(ensemble_path)
    cnn = load_cnn(cnn_path)
    if set(ens) != set(cnn):
        raise ValueError(f'Identity mismatch: ensemble-only={len(set(ens)-set(cnn))}, CNN-only={len(set(cnn)-set(ens))}')
    if len(ens) != 875:
        raise ValueError(f'Expected 875 validation identities; got {len(ens)}')
    rows = []
    for key in sorted(ens):
        cls, stem = key
        ep, cp = ens[key], cnn[key]
        ei = sorted(range(3), key=lambda i: ep[i], reverse=True)
        ci = max(range(3), key=lambda i: cp[i])
        er = CLASSES[ei[0]]
        cr = CLASSES[ci]
        rows.append({
            'true_class': cls, 'image_stem': stem,
            'ensemble_prediction': er, 'cnn_prediction': cr,
            'ensemble_correct': er == cls, 'cnn_correct': cr == cls,
            'ensemble_confidence': round(ep[ei[0]], 8),
            'ensemble_margin': round(ep[ei[0]] - ep[ei[1]], 8),
            'ensemble_second_class': CLASSES[ei[1]],
            'ensemble_true_class_probability': round(ep[CLASSES.index(cls)], 8),
            'cnn_confidence': round(cp[ci], 8),
            'cnn_true_class_probability': round(cp[CLASSES.index(cls)], 8),
            'models_disagree': er != cr,
            'ensemble_prob_normal': ep[0], 'ensemble_prob_bacterial': ep[1], 'ensemble_prob_viral': ep[2],
            'cnn_prob_normal': cp[0], 'cnn_prob_bacterial': cp[1], 'cnn_prob_viral': cp[2],
        })
    summary = summarize(rows)
    out = root / 'results/classification/error_confidence_validation_v1'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite: {out}')
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'per_image_diagnostics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
