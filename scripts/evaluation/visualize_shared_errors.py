"""Visualize validation X-rays misclassified by all three existing models.
Place in scripts/evaluation/visualize_shared_errors.py and run from project root.
No model loading, training, or test-set access. Requires pandas and Pillow.
"""
from pathlib import Path
import textwrap
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'experiments/classification_error_analysis/validation/shared_error_analysis/all_three_wrong.csv'
OUT = INPUT.parent / 'visual_review'
LABELS = ('NORMAL', 'BACTERIAL', 'VIRAL')
COLS = ('pred_baseline', 'pred_mobilenet', 'pred_finetuned')
THUMB = (290, 235)
CARD_W, CARD_H = 340, 365
GRID_COLS, PER_PAGE = 3, 12


def font(size):
    for path in ('C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f'Run analyze_shared_errors.py first: {INPUT}')
    df = pd.read_csv(INPUT, encoding='utf-8-sig')
    needed = {'file', 'actual', *COLS}
    if not needed.issubset(df.columns):
        raise ValueError(f'Missing CSV columns: {sorted(needed - set(df.columns))}')
    if df['file'].duplicated().any():
        raise ValueError('Duplicate file paths found; review manually.')
    df = df.sort_values(['actual', 'file']).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    title_font, body_font, small_font = font(20), font(16), font(13)
    missing = []
    for label in LABELS:
        sub = df[df['actual'] == label].reset_index(drop=True)
        for offset in range(0, len(sub), PER_PAGE):
            batch = sub.iloc[offset:offset + PER_PAGE]
            rows = (len(batch) + GRID_COLS - 1) // GRID_COLS
            sheet = Image.new('RGB', (GRID_COLS * CARD_W, rows * CARD_H + 48), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((16, 12), f'{label} | three-model errors | {offset + 1}-{offset + len(batch)} of {len(sub)}',
                      fill='black', font=title_font)
            for i, (_, record) in enumerate(batch.iterrows()):
                x = (i % GRID_COLS) * CARD_W
                y = (i // GRID_COLS) * CARD_H + 48
                path_str = str(record['file']).replace('\\', '/')
                path = ROOT / Path(path_str)
                if not path.is_file():
                    missing.append(path_str)
                    draw.rectangle((x + 12, y + 10, x + 326, y + 240), outline='red', width=2)
                    draw.text((x + 22, y + 105), 'IMAGE NOT FOUND', fill='red', font=body_font)
                else:
                    try:
                        with Image.open(path) as im:
                            im = ImageOps.exif_transpose(im).convert('RGB')
                            im.thumbnail(THUMB, Image.Resampling.LANCZOS)
                            ix = x + (CARD_W - im.width) // 2
                            iy = y + 10 + (THUMB[1] - im.height) // 2
                            sheet.paste(im, (ix, iy))
                    except (OSError, ValueError) as exc:
                        missing.append(f'{path_str}: {exc}')
                draw.text((x + 12, y + 253), f'Actual: {label}', fill='black', font=body_font)
                draw.text((x + 12, y + 276), f'CNN: {record["pred_baseline"]}', fill='black', font=small_font)
                draw.text((x + 12, y + 296), f'MobileNet: {record["pred_mobilenet"]}', fill='black', font=small_font)
                draw.text((x + 12, y + 316), f'Fine-Tuned: {record["pred_finetuned"]}', fill='black', font=small_font)
                name = Path(path_str).name
                draw.text((x + 12, y + 338), textwrap.shorten(name, width=41, placeholder='...'),
                          fill='#555555', font=small_font)
            output = OUT / f'{label}_page_{offset // PER_PAGE + 1:02d}.jpg'
            sheet.save(output, quality=90)
            print(f'Saved: {output.relative_to(ROOT)}')
    (OUT / 'missing_images.txt').write_text('\n'.join(missing), encoding='utf-8')
    print(f'\nTotal shared errors: {len(df)}')
    print('By class: ' + ', '.join(f'{lab}={int((df.actual == lab).sum())}' for lab in LABELS))
    print(f'Missing/unreadable images: {len(missing)}')
    print(f'Review folder: {OUT}')
    print('Reminder: images alone cannot verify whether dataset labels are clinically correct.')


if __name__ == '__main__':
    main()
