"""Pre-specified exploratory weighted ensembles on validation predictions; no training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments' / 'classification_error_analysis' / 'validation'
OUT = SOURCE / 'weighted_ensemble_analysis'
CLASSES = ['NORMAL', 'BACTERIAL', 'VIRAL']
FILES = ['baseline_cnn_predictions.csv', 'mobilenetv2_predictions.csv', 'finetuned_mobilenetv2_predictions.csv']
# Weights correspond to Baseline CNN, MobileNetV2, Fine-Tuned MobileNetV2.
# Predefined for exploratory comparison, not selected based on these results.
WEIGHTS = {
    'Equal (1/3,1/3,1/3)': [1/3, 1/3, 1/3],
    'Fine-Tuned only (0,0,1)': [0, 0, 1],
    'Two MobileNets (0,0.5,0.5)': [0, 0.5, 0.5],
    'Low baseline (0.1,0.3,0.6)': [0.1, 0.3, 0.6],
    'Balanced (0.2,0.3,0.5)': [0.2, 0.3, 0.5],
}

def main():
    frames = []
    for filename in FILES:
        p = SOURCE / filename
        df = pd.read_csv(p, dtype={'file': str})
        required = ['file', 'actual', 'predicted'] + [f'score_{c}' for c in CLASSES]
        if not set(required).issubset(df.columns):
            raise ValueError(f'Missing columns in {filename}')
        df['file'] = df['file'].str.replace('\\', '/', regex=False).str.strip()
        if df['file'].duplicated().any():
            raise ValueError(f'Duplicate files in {filename}')
        df = df.set_index('file').sort_index()
        if not df['actual'].isin(CLASSES).all():
            raise ValueError(f'Unexpected class in {filename}')
        probs = df[[f'score_{c}' for c in CLASSES]].to_numpy(dtype=float)
        if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1, atol=.002):
            raise ValueError(f'Invalid probabilities in {filename}')
        if not np.array_equal(np.array(CLASSES)[probs.argmax(axis=1)], df['predicted'].to_numpy()):
            raise ValueError(f'Predictions inconsistent with probabilities: {filename}')
        frames.append(df)
    idx = frames[0].index
    y = frames[0]['actual'].to_numpy()
    for df in frames[1:]:
        if not idx.equals(df.index) or not np.array_equal(y, df['actual'].to_numpy()):
            raise ValueError('Image alignment or labels differ across models')
    stack = np.stack([df[[f'score_{c}' for c in CLASSES]].to_numpy(dtype=float) for df in frames])
    summary = []
    detailed = pd.DataFrame({'file': idx, 'actual': y})
    for name, weights in WEIGHTS.items():
        p = np.tensordot(np.array(weights), stack, axes=(0, 0))
        pred = np.array(CLASSES)[p.argmax(axis=1)]
        cm = np.array([[( (y == a) & (pred == b)).sum() for b in CLASSES] for a in CLASSES])
        recalls = np.diag(cm) / cm.sum(axis=1)
        precisions = np.divide(np.diag(cm), cm.sum(axis=0), out=np.zeros(3), where=cm.sum(axis=0)!=0)
        f1 = np.divide(2*precisions*recalls, precisions+recalls, out=np.zeros(3), where=(precisions+recalls)!=0)
        row = {'method': name, 'accuracy': float(np.mean(pred==y)), 'macro_f1': float(f1.mean()),
               'recall_NORMAL': float(recalls[0]), 'recall_BACTERIAL': float(recalls[1]),
               'recall_VIRAL': float(recalls[2]), 'bacterial_to_viral': int(cm[1,2]),
               'viral_to_bacterial': int(cm[2,1]), 'weights': str(weights)}
        summary.append(row)
        detailed[name] = pred
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT/'weighted_summary.csv', index=False, encoding='utf-8-sig')
    detailed.to_csv(OUT/'weighted_predictions.csv', index=False, encoding='utf-8-sig')
    print(f'Validated {len(y)} paired validation images. No training or test-set access.')
    for r in summary:
        print(f"{r['method']}: accuracy={r['accuracy']:.2%}, macro_F1={r['macro_f1']:.2%}, "
              f"BACTERIAL_recall={r['recall_BACTERIAL']:.2%}, VIRAL_recall={r['recall_VIRAL']:.2%}, "
              f"B->V={r['bacterial_to_viral']}, V->B={r['viral_to_bacterial']}")
    print(f'Exploratory results saved to: {OUT}')
    print('Do not treat the highest validation score as independently confirmed improvement.')

if __name__ == '__main__':
    main()
