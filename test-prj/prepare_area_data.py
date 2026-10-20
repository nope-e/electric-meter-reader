"""Collapse reviewed window types into one area class without moving boxes or splits."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from merge_window_data import ROOT, verify_dataset


def build(source, output):
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    verify_dataset(source, ROOT.parent/'datas')
    original = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    rows = copy.deepcopy(original)
    for row in rows:
        for window in row['windows']:
            window['source_meter_type'] = window['class']
            window['class'] = 'area'
    output.mkdir(parents=True)
    for split in ['train','valid','test']:
        for sub in ['images','labels']:
            (output/split/sub).mkdir(parents=True)
    for before, row in zip(original, rows):
        shutil.copy2(source/row['output_image'], output/row['output_image'])
        old_lines = (source/row['output_label']).read_text().splitlines()
        # Preserve the exact coordinate strings, changing only the class column.
        new_lines = ['0 '+line.split(maxsplit=1)[1] for line in old_lines]
        (output/row['output_label']).write_text('\n'.join(new_lines)+'\n', encoding='utf-8')
        assert before['split'] == row['split']
        assert [b['box'] for b in before['windows']] == [b['box'] for b in row['windows']]
    config = {'path':output.as_posix(), 'train':'train/images', 'val':'valid/images',
              'test':'test/images', 'nc':1, 'names':['area']}
    (output/'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    summary = {'classes':['area'], 'images':len(rows), 'boxes':sum(len(r['windows']) for r in rows),
               'counts':{split:{'images':sum(r['split']==split for r in rows),
                                'boxes':sum(len(r['windows']) for r in rows if r['split']==split)}
                         for split in ['train','valid','test']},
               'source_types':dict(Counter(b['source_meter_type'] for r in rows for b in r['windows'])),
               'parent_dataset':str(source),
               'parent_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
               'change':'Only detection class IDs changed; images, coordinates, exclusions and splits preserved.'}
    for name, value in [('manifest',rows), ('summary',summary)]:
        (output/f'{name}.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    for name in ['excluded.json','window_label_reviews.json']:
        shutil.copy2(source/name, output/name)
    verify_dataset(output, ROOT.parent/'datas')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'datasets/meter_windows')
    parser.add_argument('--output', type=Path, default=ROOT/'datasets/meter_area')
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
