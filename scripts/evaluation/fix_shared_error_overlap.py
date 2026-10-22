from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments' / 'classification_error_analysis' / 'validation'
SRC = BASE / 'contrast_sensitivity_finetuned' / 'per_image_contrast_changes.csv'
SHARED = BASE / 'shared_error_analysis' / 'all_image_comparisons.csv'
OUT = BASE / 'contrast_sensitivity_finetuned' / 'transition_analysis'
OUT.mkdir(parents=True, exist_ok=True)

def key(path):
    # Match only class/filename, independent of absolute path, case, and slashes.
    s = str(path).strip().replace('\\', '/').lower()
    s = re.sub(r'/+', '/', s)
    parts = s.split('/')
    if len(parts) < 2:
        raise ValueError(f'Cannot identify class and filename: {path}')
    return '/'.join(parts[-2:])

a = pd.read_csv(SRC)
b = pd.read_csv(SHARED)
for label, frame, cols in [('contrast', a, ['file','actual','pred_original','pred_low','pred_high']),
                            ('historical', b, ['file','actual','all_wrong'])]:
    missing = set(cols) - set(frame.columns)
    if missing:
        raise ValueError(f'{label} missing columns: {missing}')
    frame['match_key'] = frame['file'].map(key)
    dup = frame['match_key'].duplicated(keep=False)
    if dup.any():
        print(frame.loc[dup, ['file','match_key']].head(10).to_string(index=False))
        raise ValueError(f'{label}: duplicate class/filename keys; no unsafe matching')

print('Example contrast path:', a['file'].iloc[0])
print('Example historical path:', b['file'].iloc[0])
print('Example normalized keys:', a['match_key'].iloc[0], '|', b['match_key'].iloc[0])
joined = a.merge(b[['match_key','actual','all_wrong']], on='match_key', how='left',
                 validate='one_to_one', suffixes=('', '_historical'), indicator=True)
print('Matched:', int((joined['_merge'] == 'both').sum()), '/', len(a))
if (joined['_merge'] != 'both').any() or len(a) != len(b):
    print('Unmatched examples:')
    print(joined.loc[joined['_merge'] != 'both', 'file'].head(10).to_string(index=False))
    raise ValueError('Incomplete matching; stop before calculating overlap')
if not (joined['actual'].str.upper() == joined['actual_historical'].str.upper()).all():
    raise ValueError('Label mismatch after matching; stop')

flag = joined['all_wrong'].astype(str).str.strip().str.lower()
if not flag.isin(['true','false','1','0','yes','no']).all():
    raise ValueError('Unrecognized all_wrong flag')
joined['shared_wrong'] = flag.isin(['true','1','yes'])
shared = joined[joined['shared_wrong']].copy()
print('All-three-wrong:', len(shared))
if len(shared) != 110:
    print('WARNING: expected 110 shared errors, got', len(shared))
for tag in ('low','high'):
    fixed = shared['pred_' + tag] == shared['actual']
    print(f'{tag}: fine-tuned correct after contrast change: {int(fixed.sum())}/{len(shared)}')
    shared.loc[fixed, ['file','actual','pred_original','pred_'+tag]].to_csv(
        OUT / f'shared_errors_fixed_{tag}.csv', index=False, encoding='utf-8-sig')
joined[['file','actual','shared_wrong','pred_original','pred_low','pred_high']].to_csv(
    OUT / 'matched_shared_error_overlap.csv', index=False, encoding='utf-8-sig')
print('Saved to:', OUT)
