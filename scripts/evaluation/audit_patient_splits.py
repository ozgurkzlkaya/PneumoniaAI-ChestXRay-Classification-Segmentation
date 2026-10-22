from pathlib import Path
from collections import defaultdict, Counter
import csv, re

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dataset'
OUT = ROOT / 'experiments' / 'classification_error_analysis' / 'patient_split_audit'
CLASSES = ('NORMAL', 'BACTERIAL', 'VIRAL')
SPLITS = ('train', 'validation', 'test')
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# 'person123_bacteria_...' and 'person123_virus_...' are potential IDs.
# NORMAL2-IM-... uses a different naming convention: do not infer patient IDs.
def get_id(filename):
    m = re.match(r'^(person\d+)_', filename, re.I)
    return m.group(1).lower() if m else None

rows = []
for split in SPLITS:
    for cls in CLASSES:
        folder = DATA / split / cls
        if not folder.is_dir():
            print(f'WARNING: missing folder {folder}')
            continue
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix.lower() in EXT:
                rows.append({'split': split, 'class': cls, 'patient_id': get_id(p.name) or '', 'file': str(p.relative_to(ROOT))})

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / 'image_patient_index.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=['split', 'class', 'patient_id', 'file']); w.writeheader(); w.writerows(rows)

by_id = defaultdict(list)
for r in rows:
    if r['patient_id']:
        by_id[r['patient_id']].append(r)

cross = []
for pid, group in sorted(by_id.items()):
    splits = sorted({r['split'] for r in group})
    if len(splits) > 1:
        cross.append({'patient_id': pid, 'splits': ','.join(splits), 'classes': ','.join(sorted({r['class'] for r in group})), 'n_images': len(group), 'sample_files': ' | '.join(r['file'] for r in group[:5])})
with (OUT / 'cross_split_patient_ids.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=['patient_id', 'splits', 'classes', 'n_images', 'sample_files']); w.writeheader(); w.writerows(cross)

print('Patient ID audit (filename heuristic; NOT a definitive patient linkage check)')
print('Images by split:', dict(Counter(r['split'] for r in rows)))
print('Images with recognizable personNNN ID:', sum(bool(r['patient_id']) for r in rows), '/', len(rows))
print('Distinct recognizable IDs:', len(by_id))
print('Potential IDs shared across splits:', len(cross))
for r in cross[:15]:
    print(' ', r['patient_id'], 'splits=', r['splits'], 'images=', r['n_images'])
print('Files saved in:', OUT)
print('IMPORTANT: Person IDs may be local to label/source. Same patient may have other filenames; NORMAL IDs not recognized. No images were moved or modified.')
