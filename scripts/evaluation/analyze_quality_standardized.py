"""Step 41: fixed-size image quality analysis by three-model correctness.
Read-only; no model inference, dataset modifications, or test-set access.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/classification_error_analysis/validation/shared_error_analysis/all_image_comparisons.csv'
OUT = ROOT / 'experiments/classification_error_analysis/validation/image_quality_analysis/standardized_256'
PRED = ['pred_baseline', 'pred_mobilenet', 'pred_finetuned']
SIZE = (256, 256)


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f'CSV bulunamadi: {SOURCE}')
    df = pd.read_csv(SOURCE)
    required = ['file', 'actual', *PRED]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'Eksik CSV sutunlari: {missing}; mevcut: {list(df.columns)}')

    rows = []
    missing_files = []
    for rec in df.to_dict('records'):
        file_name = str(rec['file'])
        path = Path(file_name)
        if not path.is_absolute():
            path = ROOT / path
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            missing_files.append(file_name)
            continue
        h, w = im.shape
        # Uniform pixel grid. Note: direct resizing changes the aspect ratio.
        standardized = cv2.resize(im, SIZE, interpolation=cv2.INTER_AREA if max(h, w) >= 256 else cv2.INTER_LINEAR)
        n_correct = sum(str(rec[p]).strip().upper() == str(rec['actual']).strip().upper() for p in PRED)
        group = 'all_correct' if n_correct == 3 else 'all_wrong' if n_correct == 0 else 'mixed'
        rows.append({
            'file': file_name, 'actual': rec['actual'], 'group': group, 'n_correct': n_correct,
            'original_width': w, 'original_height': h, 'original_aspect_ratio': w / h,
            'brightness_256': float(standardized.mean()),
            'contrast_256': float(standardized.std()),
            'sharpness_256': float(cv2.Laplacian(standardized, cv2.CV_64F).var()),
        })

    if missing_files:
        print('OKUNAMAYAN DOSYALAR:', len(missing_files))
        for file_name in missing_files[:10]:
            print(' ', file_name)
        raise RuntimeError('Eksik dosyalar var; eksik orneklerle sonuc uretmiyorum.')
    if len(rows) != len(df):
        raise RuntimeError('Satir sayisi uyusmuyor.')

    result = pd.DataFrame(rows)
    counts = result['group'].value_counts().to_dict()
    print('Processed:', len(result), 'Group counts:', counts)
    if len(result) == 875 and (counts.get('all_correct') != 589 or counts.get('all_wrong') != 110):
        print('WARNING: Expected all_correct=589, all_wrong=110; verify input CSV.')

    features = ['brightness_256', 'contrast_256', 'sharpness_256', 'original_width', 'original_height', 'original_aspect_ratio']
    summary = result.groupby(['actual', 'group'], observed=True).agg(
        n=('file', 'size'), **{f'median_{f}': (f, 'median') for f in features}
    ).reset_index()

    # Only compare strict groups, excluding mixed predictions.
    comparisons = []
    for cls in ['NORMAL', 'BACTERIAL', 'VIRAL']:
        subset = result[result.actual == cls]
        correct = subset[subset.group == 'all_correct']
        wrong = subset[subset.group == 'all_wrong']
        if correct.empty or wrong.empty:
            continue
        for feature in features:
            a = float(correct[feature].median())
            b = float(wrong[feature].median())
            comparisons.append({'actual': cls, 'feature': feature,
                                'n_all_correct': len(correct), 'n_all_wrong': len(wrong),
                                'median_all_correct': a, 'median_all_wrong': b,
                                'difference_wrong_minus_correct': b-a,
                                'percent_difference': (100*(b-a)/a) if a != 0 else np.nan})
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / 'quality_per_image_256.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUT / 'quality_summary_256.csv', index=False, encoding='utf-8-sig')
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(OUT / 'strict_group_comparison_256.csv', index=False, encoding='utf-8-sig')

    print('\nMEDIAN SUMMARY (256x256):')
    print(summary[['actual', 'group', 'n', 'median_brightness_256', 'median_contrast_256', 'median_sharpness_256']].to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print('\nSTRICT COMPARISON (all_wrong minus all_correct):')
    print(comparison_df[comparison_df.feature.isin(['brightness_256', 'contrast_256', 'sharpness_256'])].to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print('\nSaved to:', OUT)
    print('CAUTION: Observational analysis, not causal evidence. Resizing distorts original aspect ratios; Laplacian sharpness remains sensitive to texture/content.')

if __name__ == '__main__':
    main()
