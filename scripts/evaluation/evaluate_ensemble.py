"""Exploratory ensemble comparison using saved validation predictions only.

Run from PneumoniaProject root:
 python scripts/evaluation/evaluate_ensemble.py
No model training or test-set access. No weight tuning.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments' / 'classification_error_analysis' / 'validation'
OUTPUT = SOURCE / 'ensemble_analysis'
CLASSES = ['NORMAL', 'BACTERIAL', 'VIRAL']
FILES = {
    'Baseline CNN': 'baseline_cnn_predictions.csv',
    'MobileNetV2': 'mobilenetv2_predictions.csv',
    'Fine-Tuned MobileNetV2': 'finetuned_mobilenetv2_predictions.csv',
}


def metrics(y, pred):
    n = len(y)
    matrix = np.array([[int(np.sum((y == a) & (pred == b))) for b in CLASSES] for a in CLASSES])
    per_class = {}
    f1s = []
    for i, cls in enumerate(CLASSES):
        tp = matrix[i, i]
        recall = tp / matrix[i, :].sum() if matrix[i, :].sum() else 0.0
        precision = tp / matrix[:, i].sum() if matrix[:, i].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': int(matrix[i, :].sum())}
        f1s.append(f1)
    return {'n': n, 'accuracy': float(np.mean(y == pred)), 'macro_f1': float(np.mean(f1s)),
            'classes': per_class, 'confusion_matrix': matrix.tolist()}


def main():
    tables = {}
    required = ['file', 'actual', 'predicted'] + [f'score_{c}' for c in CLASSES]
    for name, filename in FILES.items():
        path = SOURCE / filename
        if not path.is_file():
            raise FileNotFoundError(f'Missing: {path}')
        df = pd.read_csv(path, dtype={'file': str})
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f'{filename}: missing columns {sorted(missing)}')
        df['file'] = df['file'].str.replace('\\', '/', regex=False).str.strip()
        if df['file'].duplicated().any():
            raise ValueError(f'{filename}: duplicate file paths')
        df = df.set_index('file').sort_index()
        for cls in CLASSES:
            df[f'score_{cls}'] = pd.to_numeric(df[f'score_{cls}'], errors='raise')
        scores = df[[f'score_{c}' for c in CLASSES]].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or (scores < -1e-5).any() or (scores > 1.00001).any():
            raise ValueError(f'{filename}: invalid score values')
        if not np.allclose(scores.sum(axis=1), 1, atol=0.002):
            raise ValueError(f'{filename}: class scores do not sum to 1')
        predicted_from_scores = np.array(CLASSES)[np.argmax(scores, axis=1)]
        if not np.array_equal(predicted_from_scores, df['predicted'].to_numpy()):
            raise ValueError(f'{filename}: predicted labels inconsistent with scores')
        tables[name] = df

    names = list(FILES)
    index = tables[names[0]].index
    actual = tables[names[0]]['actual'].to_numpy()
    if not set(actual).issubset(CLASSES):
        raise ValueError('Unexpected actual labels')
    for name in names[1:]:
        df = tables[name]
        if len(df) != len(index) or not df.index.equals(index):
            raise ValueError(f'File alignment mismatch: {name}')
        if not np.array_equal(df['actual'].to_numpy(), actual):
            raise ValueError(f'Actual label mismatch: {name}')

    scores = np.stack([tables[n][[f'score_{c}' for c in CLASSES]].to_numpy(dtype=float) for n in names], axis=0)
    individual = [tables[n]['predicted'].to_numpy() for n in names]
    average_scores = scores.mean(axis=0)
    avg_pred = np.array(CLASSES)[np.argmax(average_scores, axis=1)]

    # For 3 distinct votes, explicitly abstain; do not silently break ties.
    vote_pred = []
    for votes in zip(*individual):
        counts = {c: votes.count(c) for c in CLASSES}
        majority = [c for c in CLASSES if counts[c] >= 2]
        vote_pred.append(majority[0] if majority else 'ABSTAIN')
    vote_pred = np.array(vote_pred)
    covered = vote_pred != 'ABSTAIN'

    summary = []
    reports = {}
    for name, pred in zip(names, individual):
        result = metrics(actual, pred)
        reports[name] = result
        summary.append({'method': name, 'coverage': 1.0, 'accuracy_all': result['accuracy'],
                        'accuracy_covered': result['accuracy'], 'macro_f1_all': result['macro_f1'],
                        'recall_NORMAL': result['classes']['NORMAL']['recall'],
                        'recall_BACTERIAL': result['classes']['BACTERIAL']['recall'],
                        'recall_VIRAL': result['classes']['VIRAL']['recall'], 'abstentions': 0})

    avg_report = metrics(actual, avg_pred)
    reports['Equal mean softmax'] = avg_report
    summary.append({'method': 'Equal mean softmax', 'coverage': 1.0,
                    'accuracy_all': avg_report['accuracy'], 'accuracy_covered': avg_report['accuracy'],
                    'macro_f1_all': avg_report['macro_f1'],
                    'recall_NORMAL': avg_report['classes']['NORMAL']['recall'],
                    'recall_BACTERIAL': avg_report['classes']['BACTERIAL']['recall'],
                    'recall_VIRAL': avg_report['classes']['VIRAL']['recall'], 'abstentions': 0})

    vote_report = metrics(actual[covered], vote_pred[covered]) if covered.any() else None
    reports['Majority vote (covered only)'] = vote_report
    summary.append({'method': 'Majority vote (covered only)', 'coverage': float(covered.mean()),
                    'accuracy_all': None, 'accuracy_covered': vote_report['accuracy'] if vote_report else None,
                    'macro_f1_all': None,
                    'recall_NORMAL': vote_report['classes']['NORMAL']['recall'] if vote_report else None,
                    'recall_BACTERIAL': vote_report['classes']['BACTERIAL']['recall'] if vote_report else None,
                    'recall_VIRAL': vote_report['classes']['VIRAL']['recall'] if vote_report else None,
                    'abstentions': int((~covered).sum())})

    out = pd.DataFrame({'file': index, 'actual': actual})
    for name, pred in zip(names, individual):
        out[f'pred_{name.lower().replace(" ", "_").replace("-", "")}'] = pred
    out['mean_predicted'] = avg_pred
    out['mean_correct'] = avg_pred == actual
    out['vote_predicted'] = vote_pred
    out['vote_correct_when_covered'] = np.where(covered, vote_pred == actual, None)
    for i, c in enumerate(CLASSES):
        out[f'mean_score_{c}'] = average_scores[:, i]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT / 'ensemble_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(OUTPUT / 'ensemble_summary.csv', index=False, encoding='utf-8-sig')
    (OUTPUT / 'ensemble_reports.json').write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'Validated {len(actual)} aligned images across {len(names)} models.')
    print('\nExploratory validation results (not independent test performance):')
    for item in summary:
        acc = item['accuracy_covered']
        print(f"{item['method']}: accuracy_on_covered={acc:.2%} | coverage={item['coverage']:.2%} "
              f"| VIRAL_recall_on_covered={item['recall_VIRAL']:.2%} | abstentions={item['abstentions']}")
    print('\nNote: majority-vote accuracy excludes abstentions; do not compare it to full-coverage accuracy without coverage.')
    print(f'Reports saved to: {OUTPUT}')


if __name__ == '__main__':
    main()
