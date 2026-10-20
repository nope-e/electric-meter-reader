"""Build two-class readout-window YOLO data without modifying source exports."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont
import yaml

ROOT = Path(__file__).resolve().parent
NAMES = ['lcd', 'wheel']  # lcd includes electronic LED and other digital displays.
SOURCES = {
    'Digital Meter.v1i.yolov8': 'display',
    'DigitalGauge.v1i.yolov8': 'display',
    'Electricity meter.v5i.yolov8': 'typed_window',
    'meter-reader-detection.v5i.yolov8': 'reading_parent',
}
TYPE_MAP = {'Digital': 'lcd', 'Analog': 'wheel',
            'Digital meter': 'lcd', 'analog meter': 'wheel'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xyxy(box):
    x, y, w, h = box
    return [x-w/2, y-h/2, x+w/2, y+h/2]


def parse_labels(path, names):
    annotations = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5 and (len(values) < 7 or len(values) % 2 != 1):
            raise ValueError(f'{path}:{line_no}: invalid YOLO row')
        index = int(values[0])
        if not 0 <= index < len(names):
            raise ValueError(f'{path}:{line_no}: invalid class')
        coords = list(map(float, values[1:]))
        if not all(math.isfinite(v) for v in coords):
            raise ValueError(f'{path}:{line_no}: nonfinite coordinate')
        if len(coords) == 4:
            bounds = xyxy(coords)
        else:
            bounds = [min(coords[::2]), min(coords[1::2]),
                      max(coords[::2]), max(coords[1::2])]
        if min(bounds) < -0.002 or max(bounds) > 1.002:
            raise ValueError(f'{path}:{line_no}: outside image: {bounds}')
        clipped = [min(1., max(0., v)) for v in bounds]
        a, b, c, d = clipped
        if c <= a or d <= b:
            raise ValueError(f'{path}:{line_no}: empty box')
        annotations.append({'class': names[index], 'line': line_no,
                            'box': [(a+c)/2, (b+d)/2, c-a, d-b],
                            'polygon': len(coords) != 4, 'clipped': clipped != bounds})
    return annotations


def inherit_type(reading, parents):
    """Require an unambiguous typed parent covering most of this reading."""
    a, b, c, d = xyxy(reading['box'])
    area = (c-a)*(d-b)
    candidates = []
    for parent in parents:
        e, f, g, h = xyxy(parent['box'])
        overlap = max(0., min(c,g)-max(a,e))*max(0., min(d,h)-max(b,f))/area
        if overlap >= 0.5:
            candidates.append((overlap, TYPE_MAP[parent['class']]))
    types = {name for _, name in candidates}
    if len(types) != 1:
        raise ValueError(f'Reading has missing/ambiguous typed parent: {candidates}')
    return types.pop()


def select_windows(annotations, mode, rule):
    if rule.get('exclude'):
        return [], rule['reason']
    if mode == 'display':
        selected = [(rule.get('type', 'lcd'), a) for a in annotations if a['class'] == 'display']
    elif mode == 'typed_window':
        selected = [(TYPE_MAP[a['class']], a) for a in annotations if a['class'] in TYPE_MAP]
    else:
        readings = [a for a in annotations if a['class'] == 'Reading']
        if rule.get('drop_largest_reading'):
            if len(readings) != 2:
                raise ValueError('Reviewed duplicate Reading case changed')
            readings.remove(max(readings, key=lambda a: a['box'][2]*a['box'][3]))
        parents = [a for a in annotations if a['class'] in TYPE_MAP]
        selected = [(rule['type'] if 'type' in rule else inherit_type(a, parents), a)
                    for a in readings]
    if not selected:
        return [], 'Missing positive readout-window annotation; not a negative image'
    return [{'class': name, 'box': a['box'], 'source_line': a['line'],
             'from_polygon': a['polygon']} for name, a in selected], None


def preview(rows, source_root, destination):
    # Include both classes from every source and each manual correction family.
    selected = {}
    for source in SOURCES:
        for name in NAMES:
            candidates = [r for r in rows if r['source'] == source
                          and any(b['class'] == name for b in r['windows'])]
            for r in candidates[:4]:
                selected[r['image']] = r
    for r in rows:
        if r['review_rule']:
            selected.setdefault(r['source'] + r['origin'], r)
    values = list({r['image']: r for r in selected.values()}.values())
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    for start in range(0, len(values), 24):
        canvas = Image.new('RGB', (1440, 4*280), '#202020')
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(values[start:start+24]):
            with Image.open(source_root / row['image']) as im:
                im = im.convert('RGB').resize((234,234))
            pen = ImageDraw.Draw(im)
            for window in row['windows']:
                bounds = [v*234 for v in xyxy(window['box'])]
                color = '#40e0ff' if window['class'] == 'lcd' else '#50ff70'
                pen.rectangle(bounds, outline=color, width=2)
                pen.text((bounds[0], max(0, bounds[1]-16)), window['class'],
                         font=font, fill=color, stroke_width=1, stroke_fill='black')
            x, y = i%6*240, i//6*280
            canvas.paste(im, (x,y))
            draw.text((x,y+236), row['source'][:28], font=font, fill='white')
            draw.text((x,y+254), row['origin'][:28], font=font, fill='white')
        canvas.save(destination / f'preview_{start//24:02}.jpg')


def verify_dataset(output, source_root):
    config = yaml.safe_load((output/'data.yaml').read_text(encoding='utf-8'))
    names = config['names']
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if names not in [NAMES, ['area']]:
        raise ValueError(f'Unexpected dataset classes: {names}')
    rows = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    excluded = json.loads((output/'excluded.json').read_text(encoding='utf-8'))
    families, pixels = defaultdict(set), defaultdict(set)
    expected_files = set()
    for row in rows:
        source = source_root/row['image']
        if sha(source) != row['image_sha256'] or sha(output/row['output_image']) != row['image_sha256']:
            raise ValueError(f'Image checksum mismatch: {source}')
        source_label = source.parent.parent/'labels'/f'{source.stem}.txt'
        if sha(source_label) != row['label_sha256']:
            raise ValueError(f'Source labels changed: {source_label}')
        lines = (output/row['output_label']).read_text().splitlines()
        if len(lines) != len(row['windows']) or not lines:
            raise ValueError('Unexpected annotation count or unlabeled positive')
        for line, window in zip(lines, row['windows']):
            values = line.split()
            if len(values) != 5 or int(values[0]) != names.index(window['class']):
                raise ValueError('Incorrect exported class/format')
            coords = list(map(float, values[1:]))
            if any(not math.isfinite(v) for v in coords) or min(coords[2:]) <= 0:
                raise ValueError('Invalid exported coordinates')
            if min(xyxy(coords)) < -1e-7 or max(xyxy(coords)) > 1+1e-7:
                raise ValueError('Exported box outside image')
            if max(abs(a-b) for a,b in zip(coords, window['box'])) > 5.1e-9:
                raise ValueError('Exported window coordinates changed')
        families[row['source'],row['origin']].add(row['split'])
        pixels[row['pixel_sha256']].add(row['split'])
        expected_files.update([row['output_image'], row['output_label']])
    if any(len(v) != 1 for v in list(families.values()) + list(pixels.values())):
        raise ValueError('Source family / exact pixel leakage across splits')
    actual = {p.relative_to(output).as_posix() for split in ['train','valid','test']
              for sub in ['images','labels'] for p in (output/split/sub).iterdir()}
    if actual != expected_files:
        raise ValueError('Unexpected/missing output files')
    if {r['image'] for r in rows} & {r['image'] for r in excluded}:
        raise ValueError('Excluded image included in training')
    verification = {'verified_images':len(rows), 'verified_boxes':sum(len(r['windows']) for r in rows),
                    'source_and_copied_image_hashes_match':True, 'source_label_hashes_match':True,
                    'class_names':names, 'all_labels_bbox_format':True, 'coordinates_match_manifest':True,
                    'no_known_origin_or_exact_pixel_cross_split':True,
                    'output_inventory_matches_manifest':True}
    (output/'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent/'datas')
    parser.add_argument('--output', type=Path, default=ROOT/'datasets/meter_windows')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    source_root, output = args.source.resolve(), args.output.resolve()
    if args.verify_only:
        print(json.dumps(verify_dataset(output, source_root), indent=2))
        return
    # Refuse overwrite: reruns cannot leave stale images or destroy earlier data.
    if output.exists():
        raise FileExistsError(f'Choose a new --output; destination exists: {output}')
    reviews_path = ROOT/'window_label_reviews.json'
    reviews = json.loads(reviews_path.read_text(encoding='utf-8'))['rules']
    rows, excluded, metadata = [], [], {}
    applied = Counter()
    for source, mode in SOURCES.items():
        folder = source_root/source
        config = yaml.safe_load((folder/'data.yaml').read_text(encoding='utf-8'))
        names = config['names']
        if isinstance(names, dict):
            names = [names[i] for i in range(len(names))]
        metadata[source] = config
        for split in ['train','valid','test']:
            for image in sorted((folder/split/'images').iterdir()):
                if image.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp'}:
                    continue
                label = folder/split/'labels'/f'{image.stem}.txt'
                origin = image.stem.split('.rf.')[0]
                key = f'{source}/{origin}'
                rule = reviews.get(key, {})
                if rule:
                    if rule['files'].get(image.name) != [sha(image), sha(label)]:
                        raise ValueError(f'Reviewed source changed: {image}')
                    applied[key] += 1
                annotations = parse_labels(label, names)
                windows, reason = select_windows(annotations, mode, rule)
                row = {'source': source, 'image': image.relative_to(source_root).as_posix(),
                       'origin': origin, 'original_split': split, 'split': split,
                       'image_sha256': sha(image), 'label_sha256': sha(label),
                       'windows': windows, 'review_rule': key if rule else None}
                if reason:
                    excluded.append({**row, 'reason': reason})
                    continue
                with Image.open(image) as im:
                    rgb = im.convert('RGB')
                    row['pixel_sha256'] = hashlib.sha256(
                        str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    row['size'] = list(rgb.size)
                rows.append(row)
    if set(applied) != set(reviews) or any(applied[k] != len(v['files']) for k,v in reviews.items()):
        raise ValueError('Manual review entries do not match the source inventory')
    # Connected groups: all original/augmented siblings and exact decoded duplicates.
    groups = list(range(len(rows)))
    def find(i):
        while groups[i] != i:
            groups[i] = groups[groups[i]]
            i = groups[i]
        return i
    keys, pixels = {}, {}
    for i, row in enumerate(rows):
        for key in [(row['source'],row['origin']), ('pixels',row['pixel_sha256'])]:
            if key in keys:
                groups[find(i)] = find(keys[key])
            keys[key] = i
        digest = row['pixel_sha256']
        if digest in pixels:
            raise ValueError('Exact duplicate images need annotation reconciliation before merge')
        pixels[digest] = i
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[find(i)].append(row)
    priority = {'train': 0, 'valid': 1, 'test': 2}
    for group in members.values():
        split = max((r['original_split'] for r in group), key=priority.get)
        for row in group:
            row['split'] = split
    counts = {}
    for split in priority:
        rr = [r for r in rows if r['split'] == split]
        counts[split] = {'images': len(rr), 'boxes': dict(Counter(
            b['class'] for r in rr for b in r['windows']))}
    report = {'classes': NAMES, 'lcd_scope': 'All electronic digital readouts, including LCD and LED',
              'counts': counts, 'images': len(rows), 'boxes': sum(len(r['windows']) for r in rows),
              'excluded': len(excluded), 'split_moves': sum(r['split'] != r['original_split'] for r in rows),
              'source_metadata': metadata, 'review_config_sha256': sha(reviews_path),
              'independence_limit': 'Grouping covers source filenames and exact pixels; physical-meter independence is not established.'}
    # Validate everything above before creating the destination.
    output.mkdir(parents=True)
    for split in priority:
        for sub in ['images','labels']:
            (output/split/sub).mkdir(parents=True)
    for row in rows:
        image = source_root/row['image']
        stem = list(SOURCES).index(row['source'])
        filename = f's{stem}_{image.stem}'
        row['output_image'] = f"{row['split']}/images/{filename}{image.suffix.lower()}"
        row['output_label'] = f"{row['split']}/labels/{filename}.txt"
        shutil.copy2(image, output/row['output_image'])
        text = '\n'.join(f"{NAMES.index(b['class'])} " + ' '.join(f'{v:.8f}' for v in b['box'])
                         for b in row['windows']) + '\n'
        (output/row['output_label']).write_text(text, encoding='utf-8')
        check = parse_labels(output/row['output_label'], NAMES)
        if len(check) != len(row['windows']):
            raise ValueError('Written label verification failed')
    (output/'data.yaml').write_text(yaml.safe_dump({
        'path': output.as_posix(), 'train':'train/images', 'val':'valid/images',
        'test':'test/images', 'nc':2, 'names':NAMES}, sort_keys=False), encoding='utf-8')
    for name, value in [('summary',report), ('manifest',rows), ('excluded',excluded)]:
        (output/f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(reviews_path, output/reviews_path.name)
    preview(rows, source_root, output)
    verify_dataset(output, source_root)
    print(json.dumps({k:v for k,v in report.items() if k != 'source_metadata'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
