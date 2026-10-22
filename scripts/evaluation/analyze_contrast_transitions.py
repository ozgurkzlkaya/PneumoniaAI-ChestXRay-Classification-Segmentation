from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments' / 'classification_error_analysis' / 'validation'
SRC = BASE / 'contrast_sensitivity_finetuned' / 'per_image_contrast_changes.csv'
SHARED = BASE / 'shared_error_analysis' / 'all_image_comparisons.csv'
OUT = BASE / 'contrast_sensitivity_finetuned' / 'transition_analysis'
CLASSES = ['NORMAL', 'BACTERIAL', 'VIRAL']

if not SRC.is_file():
    raise FileNotFoundError(f'Missing: {SRC}')
df = pd.read_csv(SRC)
required = {'file','actual','pred_original','pred_low','pred_high'}
if not required.issubset(df.columns):
    raise ValueError(f'Missing columns: {sorted(required-set(df.columns))}')
if len(df) != 875:
    print(f'WARNING: expected 875 validation images, found {len(df)}')
if df['file'].duplicated().any():
    raise ValueError('Duplicate file entries found')
for c in ['actual','pred_original','pred_low','pred_high']:
    if not df[c].isin(CLASSES).all():
        raise ValueError(f'Unexpected class value in {c}')

OUT.mkdir(parents=True, exist_ok=True)
summary = []
for tag in ['low', 'high']:
    col = f'pred_{tag}'
    change = df[col] != df['pred_original']
    fixed = (df['pred_original'] != df['actual']) & (df[col] == df['actual'])
    broken = (df['pred_original'] == df['actual']) & (df[col] != df['actual'])
    print(f'\n=== {tag.upper()} CONTRAST ===')
    print(f'changed={change.sum()}, fixed={fixed.sum()}, broken={broken.sum()}, net={fixed.sum()-broken.sum()}')
    for cls in CLASSES:
        d = df[df['actual'] == cls]
        f = ((d['pred_original'] != cls) & (d[col] == cls)).sum()
        b = ((d['pred_original'] == cls) & (d[col] != cls)).sum()
        print(f'{cls}: n={len(d)} fixed={f} broken={b} net={f-b}')
        summary.append({'condition':tag,'class':cls,'n':len(d),'fixed':int(f),'broken':int(b),'net':int(f-b)})
    matrix = pd.crosstab(df['pred_original'], df[col]).reindex(index=CLASSES,columns=CLASSES,fill_value=0)
    print('\nPrediction transitions (rows=original prediction; columns=modified prediction):')
    print(matrix.to_string())
    matrix.to_csv(OUT / f'prediction_transition_{tag}.csv',encoding='utf-8-sig')
    print('\nBACTERIAL actual, predicted BACTERIAL -> VIRAL:',int(((df['actual']=='BACTERIAL')&(df['pred_original']=='BACTERIAL')&(df[col]=='VIRAL')).sum()))
    print('VIRAL actual, predicted VIRAL -> BACTERIAL:',int(((df['actual']=='VIRAL')&(df['pred_original']=='VIRAL')&(df[col]=='BACTERIAL')).sum()))
    df.loc[change, ['file','actual','pred_original',col]].to_csv(OUT/f'changed_predictions_{tag}.csv',index=False,encoding='utf-8-sig')

pd.DataFrame(summary).to_csv(OUT/'class_changes.csv',index=False,encoding='utf-8-sig')
if SHARED.is_file():
    shared = pd.read_csv(SHARED)
    if not {'file','all_wrong'}.issubset(shared.columns):
        print('\nWARNING: shared-error CSV lacks file/all_wrong columns; overlap skipped')
    else:
        flags = shared[['file','all_wrong']].copy()
        flags['all_wrong'] = flags['all_wrong'].astype(str).str.strip().str.lower().isin(['true','1','yes'])
        joined = df.merge(flags,on='file',how='left',validate='one_to_one')
        if joined['all_wrong'].isna().any():
            print(f'WARNING: {joined["all_wrong"].isna().sum()} file paths did not match; overlap skipped')
        else:
            print('\n=== HISTORICAL SHARED ERRORS ===')
            print('All-three-wrong:',int(joined['all_wrong'].sum()))
            for tag in ['low','high']:
                group = joined[joined['all_wrong']]
                fixed = group['pred_'+tag] == group['actual']
                print(f'{tag}: fine-tuned model fixed {int(fixed.sum())} of {len(group)} shared errors')
                group.loc[fixed,['file','actual','pred_original','pred_'+tag]].to_csv(OUT/f'shared_errors_fixed_{tag}.csv',index=False,encoding='utf-8-sig')
else:
    print(f'\nWARNING: shared-error CSV not found at {SHARED}; overlap skipped')
print('\nSaved:',OUT)
print('NOTE: Validation-only exploratory analysis. Contrast transformations do not prove what visual cues the model uses.')
