from pathlib import Path
import pandas as pd
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/classification_error_analysis/validation/shared_error_analysis/all_image_comparisons.csv'
OUT = ROOT / 'experiments/classification_error_analysis/validation/image_quality_analysis'


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Missing input: {SOURCE}')
    df = pd.read_csv(SOURCE)
    print('CSV columns:', ', '.join(df.columns))
    file_col = next((c for c in ('file', 'filepath', 'path', 'image_path') if c in df.columns), None)
    actual_col = next((c for c in ('actual', 'true_label', 'label') if c in df.columns), None)
    if not file_col or not actual_col:
        raise ValueError('Expected file and actual columns; inspect printed CSV columns.')
    prediction_cols = ['pred_baseline', 'pred_mobilenet', 'pred_finetuned']
    missing = [c for c in prediction_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing prediction columns: {missing}')
    records = []
    unreadable = 0
    for _, row in df.iterrows():
        p = Path(str(row[file_col]))
        if not p.is_absolute():
            p = ROOT / p
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            print('Unreadable:', p)
            unreadable += 1
            continue
        h, w = im.shape
        correct = [str(row[c]) == str(row[actual_col]) for c in prediction_cols]
        records.append(dict(file=str(row[file_col]), actual=row[actual_col], width=w, height=h,
                            aspect_ratio=w/h, brightness=float(im.mean()), contrast=float(im.std()),
                            sharpness=float(cv2.Laplacian(im, cv2.CV_64F).var()),
                            all_correct=all(correct), all_wrong=not any(correct),
                            n_correct=sum(correct)))
    if not records:
        raise ValueError('No images processed')
    OUT.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(records)
    detail.to_csv(OUT/'image_quality_per_image.csv', index=False, encoding='utf-8-sig')
    summary = detail.groupby(['actual', 'all_wrong'], dropna=False).agg(
        n=('file','size'), median_brightness=('brightness','median'),
        median_contrast=('contrast','median'), median_sharpness=('sharpness','median'),
        median_width=('width','median'), median_height=('height','median'),
        median_aspect_ratio=('aspect_ratio','median')).reset_index()
    summary.to_csv(OUT/'quality_summary.csv', index=False, encoding='utf-8-sig')
    print(f'Processed {len(detail)} images; all_wrong={detail.all_wrong.sum()}, all_correct={detail.all_correct.sum()}, unreadable={unreadable}')
    if len(detail) != len(df):
        print(f'WARNING: Expected {len(df)} images, processed {len(detail)}. Check file paths before interpreting results.')
    if 'all_wrong' in df.columns and unreadable == 0:
        expected_wrong = df['all_wrong'].astype(str).str.lower().isin(['true', '1', 'yes']).sum()
        if int(detail.all_wrong.sum()) != int(expected_wrong):
            print(f'WARNING: all_wrong mismatch: recomputed={int(detail.all_wrong.sum())}, CSV={int(expected_wrong)}')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print('Saved to:', OUT)
    print('Caution: associations are not proof of causes; sharpness depends on image scale.')

if __name__ == '__main__':
    main()
