"""Compare predictions of three existing models on the same validation images.
Run from PneumoniaProject root: python scripts/evaluation/analyze_shared_errors.py
No training or test-set access.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments' / 'classification_error_analysis' / 'validation'
OUT = SOURCE / 'shared_error_analysis'
MODELS = {
    'baseline': 'baseline_cnn_predictions.csv',
    'mobilenet': 'mobilenetv2_predictions.csv',
    'finetuned': 'finetuned_mobilenetv2_predictions.csv',
}
CLASSES = ['NORMAL', 'BACTERIAL', 'VIRAL']


def read_predictions(name, filename):
    path = SOURCE / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing: {path}')
    df = pd.read_csv(path)
    required = {'file', 'actual', 'predicted', 'confidence'}
    if not required.issubset(df.columns):
        raise ValueError(f'{filename}: missing columns {sorted(required - set(df.columns))}')
    df['file'] = df['file'].astype(str).str.replace('\\', '/', regex=False)
    if df['file'].duplicated().any():
        raise ValueError(f'{filename}: duplicate file paths found')
    cols = ['file', 'actual', 'predicted', 'confidence']
    for label in CLASSES:
        col = f'score_{label}'
        if col in df.columns:
            cols.append(col)
    df = df[cols].rename(columns={
        'actual': f'actual_{name}',
        'predicted': f'pred_{name}',
        'confidence': f'confidence_{name}',
        **{f'score_{label}': f'score_{label}_{name}' for label in CLASSES},
    })
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    merged = None
    for name, filename in MODELS.items():
        df = read_predictions(name, filename)
        merged = df if merged is None else merged.merge(df, on='file', how='outer', validate='one_to_one', indicator=True)
        if '_merge' in merged.columns:
            if not (merged['_merge'] == 'both').all():
                examples = merged.loc[merged['_merge'] != 'both', 'file'].head(5).tolist()
                raise ValueError(f'Files do not match across models. Examples: {examples}')
            merged = merged.drop(columns='_merge')
    if not ((merged['actual_baseline'] == merged['actual_mobilenet']) &
            (merged['actual_baseline'] == merged['actual_finetuned'])).all():
        raise ValueError('Actual class labels differ between model CSV files.')
    merged = merged.rename(columns={'actual_baseline': 'actual'}).drop(
        columns=['actual_mobilenet', 'actual_finetuned'])
    for name in MODELS:
        merged[f'correct_{name}'] = merged[f'pred_{name}'] == merged['actual']
    correct_cols = [f'correct_{name}' for name in MODELS]
    merged['n_correct'] = merged[correct_cols].sum(axis=1)
    merged['all_wrong'] = merged['n_correct'] == 0
    merged['all_correct'] = merged['n_correct'] == 3
    merged['disagreement'] = merged[[f'pred_{name}' for name in MODELS]].nunique(axis=1) > 1
    merged['finetuned_only_correct'] = (merged['correct_finetuned'] &
                                      ~merged['correct_baseline'] & ~merged['correct_mobilenet'])
    merged['finetuned_only_wrong'] = (~merged['correct_finetuned'] &
                                    merged['correct_baseline'] & merged['correct_mobilenet'])
    merged = merged.sort_values(['actual', 'file']).reset_index(drop=True)
    merged.to_csv(OUT / 'all_image_comparisons.csv', index=False, encoding='utf-8-sig')
    merged.loc[merged['all_wrong']].to_csv(OUT / 'all_three_wrong.csv', index=False, encoding='utf-8-sig')
    merged.loc[merged['disagreement']].to_csv(OUT / 'model_disagreements.csv', index=False, encoding='utf-8-sig')
    merged.loc[merged['finetuned_only_correct']].to_csv(OUT / 'finetuned_only_correct.csv', index=False, encoding='utf-8-sig')
    summary = []
    for label in CLASSES:
        sub = merged[merged['actual'] == label]
        row = {'class': label, 'n': len(sub),
               'all_wrong': int(sub['all_wrong'].sum()),
               'all_correct': int(sub['all_correct'].sum()),
               'disagreement': int(sub['disagreement'].sum()),
               'finetuned_only_correct': int(sub['finetuned_only_correct'].sum())}
        for name in MODELS:
            row[f'recall_{name}'] = float(sub[f'correct_{name}'].mean()) if len(sub) else float('nan')
            row[f'{name}_to_BACTERIAL'] = int((sub[f'pred_{name}'] == 'BACTERIAL').sum())
            row[f'{name}_to_VIRAL'] = int((sub[f'pred_{name}'] == 'VIRAL').sum())
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / 'class_comparison.csv', index=False, encoding='utf-8-sig')
    print(f'Validated {len(merged)} paired images across three models.')
    print(f'All correct: {int(merged.all_correct.sum())}')
    print(f'All wrong: {int(merged.all_wrong.sum())}')
    print(f'Model disagreement: {int(merged.disagreement.sum())}')
    print(f'Only Fine-Tuned correct: {int(merged.finetuned_only_correct.sum())}')
    print(f'Only Fine-Tuned wrong: {int(merged.finetuned_only_wrong.sum())}')
    print('\nClass-wise recall:')
    for row in summary:
        print(f"{row['class']} (n={row['n']}): " + ' | '.join(
            f'{name}={row[f"recall_{name}"]:.2%}' for name in MODELS) +
            f" | all_wrong={row['all_wrong']} | disagreements={row['disagreement']}")
    print(f'\nSaved to: {OUT}')


if __name__ == '__main__':
    main()
