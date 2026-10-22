"""Step 57: exploratory paired comparison on validation predictions only.
No model loading, training, or test data. Requires numpy, pandas, scipy, sklearn.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments' / 'contrast_augmentation_training' / 'robustness_comparison'
OUT = SOURCE / 'paired_statistics'
CLASSES = ['NORMAL', 'BACTERIAL', 'VIRAL']
SEED = 42
N_BOOT = 3000

def metric_diff(y, c, a):
    c_ok = (c == y)
    a_ok = (a == y)
    out = {
        'accuracy_delta': float(a_ok.mean() - c_ok.mean()),
        'macro_f1_delta': float(f1_score(y, a, labels=CLASSES, average='macro', zero_division=0) -
                                f1_score(y, c, labels=CLASSES, average='macro', zero_division=0)),
    }
    for label in CLASSES:
        mask = (y == label)
        out[f'{label.lower()}_recall_delta'] = float(a_ok[mask].mean() - c_ok[mask].mean()) if mask.any() else np.nan
    return out

def percentile_ci(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return [float(v) for v in np.percentile(finite, [2.5, 97.5])] if len(finite) else [None, None]

def main():
    csv_path = SOURCE / 'per_image_predictions.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    expected = {'file', 'actual'} | {f'{arm}_{tag}_prediction' for arm in ('control','contrast') for tag in ('original','low','high')}
    expected |= {f'{arm}_{tag}_changed' for arm in ('control','contrast') for tag in ('low','high')}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if len(df) != 875 or df['file'].duplicated().any() or not df['actual'].isin(CLASSES).all():
        raise ValueError('Validation row count, unique filenames or class labels invalid.')
    rng = np.random.default_rng(SEED)
    y = df['actual'].to_numpy()
    # Stratified paired bootstrap: sample images within each true class, keeping model predictions paired.
    class_indices = [np.flatnonzero(y == name) for name in CLASSES]
    result = {'scope': 'Exploratory image-level validation only; no patient-cluster correction',
              'n_images': len(df), 'bootstrap_iterations': N_BOOT, 'bootstrap_seed': SEED,
              'bootstrap_method': 'stratified paired image resampling', 'conditions': {}}
    for tag in ('original','low','high'):
        c = df[f'control_{tag}_prediction'].to_numpy()
        a = df[f'contrast_{tag}_prediction'].to_numpy()
        if not np.isin(c, CLASSES).all() or not np.isin(a, CLASSES).all():
            raise ValueError(f'Invalid predicted class for {tag}')
        c_ok, a_ok = c == y, a == y
        control_only = int(np.sum(c_ok & ~a_ok))
        contrast_only = int(np.sum(~c_ok & a_ok))
        discordant = control_only + contrast_only
        p_value = float(binomtest(min(control_only, contrast_only), n=discordant, p=0.5).pvalue) if discordant else 1.0
        observed = metric_diff(y, c, a)
        samples = {k: [] for k in observed}
        stability = None
        if tag != 'original':
            c_changed = df[f'control_{tag}_changed'].to_numpy(dtype=int)
            a_changed = df[f'contrast_{tag}_changed'].to_numpy(dtype=int)
            if not np.isin(c_changed, [0,1]).all() or not np.isin(a_changed, [0,1]).all():
                raise ValueError('Changed flags must be 0 or 1')
            stability = {'control_changed': int(c_changed.sum()), 'contrast_changed': int(a_changed.sum()),
                         'change_rate_delta': float(a_changed.mean() - c_changed.mean())}
            samples['change_rate_delta'] = []
        for _ in range(N_BOOT):
            idx = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in class_indices])
            diffs = metric_diff(y[idx], c[idx], a[idx])
            for k, v in diffs.items():
                samples[k].append(v)
            if stability is not None:
                samples['change_rate_delta'].append(float(a_changed[idx].mean() - c_changed[idx].mean()))
        report = {
            'control_correct': int(c_ok.sum()), 'contrast_correct': int(a_ok.sum()),
            'control_only_correct': control_only, 'contrast_only_correct': contrast_only,
            'both_correct': int(np.sum(c_ok & a_ok)), 'both_wrong': int(np.sum(~c_ok & ~a_ok)),
            'mcnemar_exact_two_sided_p': p_value,
            'observed_deltas_contrast_minus_control': observed,
            'bootstrap_95pct_ci': {k: percentile_ci(v) for k, v in samples.items()},
        }
        if stability is not None:
            report['stability'] = stability
        result['conditions'][tag] = report
        print(f'\n{tag.upper()} | control_correct={report["control_correct"]} contrast_correct={report["contrast_correct"]}')
        print(f'  discordant: control_only={control_only} contrast_only={contrast_only}; exact McNemar p={p_value:.4f}')
        for key, delta in observed.items():
            lo, hi = report['bootstrap_95pct_ci'][key]
            print(f'  {key}: {delta*100:+.2f} pp; paired bootstrap 95% CI [{lo*100:+.2f}, {hi*100:+.2f}] pp')
        if stability is not None:
            lo, hi = report['bootstrap_95pct_ci']['change_rate_delta']
            print(f'  changed: control={stability["control_changed"]}, contrast={stability["contrast_changed"]}; '
                  f'rate delta={stability["change_rate_delta"]*100:+.2f} pp; CI [{lo*100:+.2f}, {hi*100:+.2f}] pp')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'paired_statistics.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSaved: {path}')
    print('CAUTION: Reused validation, image-level bootstrap, potential patient clustering, multiple exploratory comparisons.')

if __name__ == '__main__':
    main()
