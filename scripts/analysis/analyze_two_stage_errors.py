"""Validation-only paired error analysis for PneumoniaAI two-stage models."""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

from train_two_stage import CLASSES, collect, predict_prob


def adjusted_viral(q, threshold):
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-7, 1 - 1e-7)
    return (q * (1 - threshold)) / (q * (1 - threshold) + (1 - q) * threshold)


def combine(p1, p2, threshold):
    viral = adjusted_viral(p2[:, 1], threshold)
    return np.column_stack((p1[:, 0], p1[:, 1] * (1 - viral), p1[:, 1] * viral))


def summarize(y, pred):
    report = classification_report(y, pred, labels=[0, 1, 2], target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        'recall': {name: float(report[name]['recall']) for name in CLASSES},
        'f1': {name: float(report[name]['f1-score']) for name in CLASSES},
    }


def gallery(rows, output, title, limit=24):
    if not rows:
        return 0
    w, h, gap, header = 240, 270, 12, 64
    ncols = 4
    nrows = (min(len(rows), limit) + ncols - 1) // ncols
    canvas = Image.new('RGB', (ncols * (w + gap) + gap, nrows * (h + gap) + gap + header), '#f3f5f8')
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 16), title, fill='#14213d')
    for k, row in enumerate(rows[:limit]):
        x = gap + (k % ncols) * (w + gap)
        y = gap + header + (k // ncols) * (h + gap)
        draw.rectangle((x, y, x+w, y+h), fill='white')
        try:
            with Image.open(row['_path']) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((w-12, 192))
                ox, oy = x+(w-im.width)//2, y+6+(192-im.height)//2
                canvas.paste(im, (ox, oy))
        except Exception as exc:
            draw.text((x+8, y+25), 'Image unavailable', fill='#a00')
        draw.text((x+7, y+202), f"True: {row['true_label']}", fill='#14213d')
        draw.text((x+7, y+220), f"0.50: {row['baseline_050_pred']} | 0.55: {row['baseline_055_pred']}", fill='#14213d')
        draw.text((x+7, y+238), f"Aug: {row['augmentation_050_pred']} | P2(V): {float(row['stage2_baseline_viral_prob']):.3f}", fill='#14213d')
        draw.text((x+7, y+253), Path(row['filename']).stem[:30], fill='#4c5564')
    canvas.save(output, optimize=True)
    return min(len(rows), limit)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='.')
    ap.add_argument('--previous-run', required=True)
    ap.add_argument('--improvement-run', required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--gallery-limit', type=int, default=24)
    args = ap.parse_args()
    if args.batch_size < 1 or args.gallery_limit < 1:
        ap.error('batch-size and gallery-limit must be positive')
    root = Path(args.root).resolve()
    previous = (root / args.previous_run).resolve()
    improvement = (root / args.improvement_run).resolve()
    models = {
        'stage1': previous / 'stage1_selected.keras',
        'baseline': previous / 'stage2_selected.keras',
        'augmentation': improvement / 'low_lr_intensity_aug' / 'best.keras',
    }
    for key, path in models.items():
        if not path.is_file():
            raise FileNotFoundError(f'{key}: {path}')
    paths, labels = collect(root / 'dataset', 'validation')
    out = root / 'experiments' / 'two_stage_classification' / datetime.now().strftime('error_analysis_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    print('Output:', out, flush=True)
    print('VALIDATION ONLY; no test data, no training, no model changes.', flush=True)
    probabilities = {}
    for key, path in models.items():
        print('Predicting:', key, flush=True)
        model = tf.keras.models.load_model(path, compile=False)
        probabilities[key] = predict_prob(model, paths, args.batch_size)
        del model
        tf.keras.backend.clear_session()
        if probabilities[key].shape != (len(paths), 2):
            raise ValueError(f'Unexpected {key} prediction shape {probabilities[key].shape}')
    p1, p2, p2_aug = (probabilities[k] for k in ('stage1', 'baseline', 'augmentation'))
    preds = {}
    probs = {}
    for name, p, t in [('baseline_050', p2, .5), ('baseline_055', p2, .55), ('augmentation_050', p2_aug, .5)]:
        probs[name] = combine(p1, p, t)
        preds[name] = np.argmax(probs[name], axis=1)
    stage1_true = (labels != 0).astype(int)
    stage1_pred = np.argmax(p1, axis=1)
    stage2_true = (labels[labels != 0] == 2).astype(int)
    stage2_base = np.argmax(p2[labels != 0], axis=1)
    stage2_aug = np.argmax(p2_aug[labels != 0], axis=1)
    rows = []
    for i, (path, true) in enumerate(zip(paths, labels)):
        row = {
            'filename': Path(path).name,
            'relative_path': str(Path(path).relative_to(root)),
            'true_label': CLASSES[true],
            'stage1_true': 'NORMAL' if true == 0 else 'PNEUMONIA',
            'stage1_pred': 'NORMAL' if stage1_pred[i] == 0 else 'PNEUMONIA',
            'stage1_correct': int(stage1_pred[i] == stage1_true[i]),
            'stage1_pneumonia_prob': float(p1[i, 1]),
            'stage2_baseline_viral_prob': float(p2[i, 1]),
            'stage2_augmentation_viral_prob': float(p2_aug[i, 1]),
        }
        for name in preds:
            row[name + '_pred'] = CLASSES[preds[name][i]]
            row[name + '_correct'] = int(preds[name][i] == true)
            row[name + '_confidence'] = float(probs[name][i].max())
            row[name + '_true_probability'] = float(probs[name][i, true])
        row['_path'] = str(path)
        rows.append(row)
    with (out / 'per_image_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        keys = [k for k in rows[0] if not k.startswith('_')]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in keys} for r in rows])
    transitions = {}
    for a, b in [('baseline_050', 'baseline_055'), ('baseline_050', 'augmentation_050')]:
        changed = [r for r in rows if r[a+'_pred'] != r[b+'_pred']]
        gained = [r for r in rows if r[a+'_correct'] == 0 and r[b+'_correct'] == 1]
        lost = [r for r in rows if r[a+'_correct'] == 1 and r[b+'_correct'] == 0]
        transitions[a + '_to_' + b] = {
            'changed_predictions': len(changed), 'became_correct': len(gained),
            'became_incorrect': len(lost), 'net_correct_change': len(gained)-len(lost),
            'gains_by_true_class': dict(Counter(r['true_label'] for r in gained)),
            'losses_by_true_class': dict(Counter(r['true_label'] for r in lost)),
            'gains_filenames': [r['relative_path'] for r in gained],
            'losses_filenames': [r['relative_path'] for r in lost],
        }
    stage1_errors = [r for r in rows if not r['stage1_correct']]
    baseline_errors = [r for r in rows if not r['baseline_050_correct']]
    changed055 = [r for r in rows if r['baseline_050_pred'] != r['baseline_055_pred']]
    viral_lost = [r for r in rows if r['true_label'] == 'VIRAL' and r['baseline_050_correct'] and not r['baseline_055_correct']]
    low_confidence = sorted(rows, key=lambda r: r['baseline_050_confidence'])[:args.gallery_limit]
    gallery(stage1_errors, out/'stage1_errors.png', 'Stage 1 errors', args.gallery_limit)
    gallery(changed055, out/'threshold_changed.png', 'Threshold 0.50 -> 0.55: changed predictions', args.gallery_limit)
    gallery(viral_lost, out/'viral_lost_at_055.png', 'VIRAL correct at 0.50, wrong at 0.55', args.gallery_limit)
    gallery(low_confidence, out/'low_confidence.png', 'Lowest confidence (baseline 0.50)', args.gallery_limit)
    summary = {
        'validation_images': int(len(labels)),
        'class_order': list(CLASSES),
        'models': {name: summarize(labels, pred) for name, pred in preds.items()},
        'stage1': {
            'accuracy': float(accuracy_score(stage1_true, stage1_pred)),
            'errors': len(stage1_errors),
            'confusion_matrix_normal_pneumonia': confusion_matrix(stage1_true, stage1_pred, labels=[0,1]).tolist(),
        },
        'stage2_on_true_pneumonia_only': {
            'n': int(len(stage2_true)),
            'baseline_accuracy': float(accuracy_score(stage2_true, stage2_base)),
            'augmentation_accuracy': float(accuracy_score(stage2_true, stage2_aug)),
            'baseline_confusion_matrix_bacterial_viral': confusion_matrix(stage2_true, stage2_base, labels=[0,1]).tolist(),
            'augmentation_confusion_matrix_bacterial_viral': confusion_matrix(stage2_true, stage2_aug, labels=[0,1]).tolist(),
        },
        'transitions': transitions,
        'baseline_050_errors': len(baseline_errors),
        'baseline_050_high_confidence_wrong_ge_0_80': sum(r['baseline_050_confidence'] >= .8 for r in baseline_errors),
        'warning': 'Exploratory validation, not independent evidence. Dataset has known split integrity issues. Confidence is not calibrated. Gallery images are for research error review, not diagnosis.',
    }
    (out/'error_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['stage1','stage2_on_true_pneumonia_only']}, indent=2))
    for name, value in transitions.items():
        print(name, {k: value[k] for k in ('changed_predictions','became_correct','became_incorrect','net_correct_change')})
    print('Saved:', out/'error_summary.json')
    print('Saved:', out/'per_image_predictions.csv')
    print('Galleries:', out/'stage1_errors.png', out/'threshold_changed.png', out/'viral_lost_at_055.png', out/'low_confidence.png')

if __name__ == '__main__':
    main()
