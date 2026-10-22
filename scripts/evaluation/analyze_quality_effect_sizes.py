"""Step 42: descriptive effect sizes for fixed-size validation image metrics.
Read-only. Does not inspect test data or retrain/tune models.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / 'experiments/classification_error_analysis/validation/image_quality_analysis/standardized_256'
SOURCE = DIR / 'quality_per_image_256.csv'
FEATURES = ['brightness_256', 'contrast_256', 'sharpness_256']
SEED = 42
BOOTSTRAPS = 500


def probability_wrong_greater(wrong, correct):
    # Pairwise rank probability (ties count half); 0.5 indicates no separation.
    d = wrong[:, None] - correct[None, :]
    return float((np.count_nonzero(d > 0) + 0.5 * np.count_nonzero(d == 0)) / d.size)


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Missing file: {SOURCE}. Run Step 41 first.')
    df = pd.read_csv(SOURCE)
    required = ['actual', 'group', *FEATURES]
    if any(c not in df.columns for c in required):
        raise ValueError(f'Missing columns: {set(required) - set(df.columns)}')
    rng = np.random.default_rng(SEED)
    results = []
    for cls in ['NORMAL', 'BACTERIAL', 'VIRAL']:
        d = df[(df.actual == cls) & df.group.isin(['all_correct', 'all_wrong'])]
        for feat in FEATURES:
            correct = d.loc[d.group == 'all_correct', feat].dropna().to_numpy(dtype=float)
            wrong = d.loc[d.group == 'all_wrong', feat].dropna().to_numpy(dtype=float)
            if not len(correct) or not len(wrong):
                continue
            auc = probability_wrong_greater(wrong, correct)
            boot = np.empty(BOOTSTRAPS)
            for i in range(BOOTSTRAPS):
                w = rng.choice(wrong, len(wrong), replace=True)
                c = rng.choice(correct, len(correct), replace=True)
                boot[i] = probability_wrong_greater(w, c)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            results.append({
                'class': cls, 'feature': feat, 'n_correct': len(correct), 'n_wrong': len(wrong),
                'median_correct': float(np.median(correct)), 'median_wrong': float(np.median(wrong)),
                'P_wrong_greater': auc, 'bootstrap_95_low': float(lo), 'bootstrap_95_high': float(hi),
            })
    out = pd.DataFrame(results)
    path = DIR / 'descriptive_effect_sizes_256.csv'
    out.to_csv(path, index=False, encoding='utf-8-sig')
    print('DESCRIPTIVE RANK PROBABILITIES (wrong > correct; 0.50 = no separation)')
    print(out.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\nSaved:', path)
    print('CAUTION: Image-level bootstrap is exploratory; repeated images from one patient may be dependent.')
    print('These are NOT causal effects, prospective performance, or clinical-quality scores.')


if __name__ == '__main__':
    main()
