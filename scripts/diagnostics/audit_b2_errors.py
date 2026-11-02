"""Compare segmentation masks for correctly vs incorrectly classified B2 validation X-rays.
Read-only for existing datasets, models and predictions. Produces contact sheets and CSV.
"""
from pathlib import Path
import argparse
import csv
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw
from inspect_pediatric_masks import process, fit, CLASSES, EXTENSIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--per-group', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error('--per-group must be positive')
    root = args.root.resolve()
    predictions = root / 'results/classification/b2_mask_sensitivity/predictions.csv'
    model_path = root / 'best_segmentation_model_v3.keras'
    val = root / 'dataset/validation'
    for p in (predictions, model_path, val):
        if not p.exists():
            raise FileNotFoundError(p)
    df = pd.read_csv(predictions)
    df = df.loc[df['iterations'] == 3].copy()
    if len(df) != 875 or df.duplicated(['true_class', 'filename']).any():
        raise ValueError('Expected 875 unique iteration-3 predictions')
    out = root / 'results/segmentation/b2_error_audit'
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    model = tf.keras.models.load_model(model_path, compile=False)
    report = []
    for cls in CLASSES:
        group = df.loc[df['true_class'] == cls]
        for status, subset in [('correct', group.loc[group['predicted_class'] == cls]),
                               ('wrong', group.loc[group['predicted_class'] != cls])]:
            if len(subset) < args.per_group:
                raise ValueError(f'{cls}/{status}: only {len(subset)} samples')
            selected = rng.sample(list(subset.to_dict('records')), args.per_group)
            page = Image.new('RGB', (940, 350 * len(selected) + 55), (248, 249, 252))
            draw = ImageDraw.Draw(page)
            draw.text((15, 12), f'{cls} / {status.upper()} | original / overlay / B2 masked', fill=(0, 0, 0))
            originals = {p.stem: p for p in (val / cls).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}
            for i, row in enumerate(selected):
                stem = Path(row['filename']).stem
                if stem not in originals:
                    raise FileNotFoundError(f'Original not found: {cls}/{stem}')
                path = originals[stem]
                with Image.open(path) as f:
                    image = f.convert('RGB')
                overlay, masked, raw_pct, exp_pct, components = process(image, model)
                y = 50 + i * 350
                for j, im in enumerate((image, overlay, masked)):
                    page.paste(fit(im), (15 + j * 310, y))
                draw.text((15, y + 305), f'{i+1}. {path.name[:42]} | pred={row["predicted_class"]}', fill=(0, 0, 0))
                draw.text((15, y + 322), f'mask {raw_pct:.1f}% -> B2 {exp_pct:.1f}% | regions={components}', fill=(35, 40, 50))
                report.append({'true_class': cls, 'status': status, 'predicted_class': row['predicted_class'],
                               'filename': path.name, 'raw_mask_pct': round(raw_pct, 3),
                               'expanded_mask_pct': round(exp_pct, 3), 'mask_components': components,
                               'prob_normal': row['prob_normal'], 'prob_bacterial': row['prob_bacterial'],
                               'prob_viral': row['prob_viral']})
                print(f'{cls} {status}: {i+1}/{args.per_group} {path.name}', flush=True)
            target = out / f'{cls.lower()}_{status}.png'
            page.save(target)
            print('Saved:', target, flush=True)
    csv_path = out / 'selected_samples.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print('\nSample summary (not representative of full validation set):')
    result = pd.DataFrame(report)
    print(result.groupby(['true_class', 'status'])[['raw_mask_pct', 'expanded_mask_pct', 'mask_components']].mean().round(2).to_string())
    print('Saved:', csv_path)
    print('Visual audit only: no pediatric ground-truth segmentation masks.')

if __name__ == '__main__':
    main()
