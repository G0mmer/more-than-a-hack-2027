"""Step 1. Download the five public DFL matches used for training and check their SHA-256.

Run from the repository root: python training/download_dfl.py --out data/dfl
Files that already exist with the expected digest are kept.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

SOURCES = Path(__file__).with_name('dfl_sources.json')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def main(out):
    for item in json.loads(SOURCES.read_text())['files']:
        target = out/item['match']/item['name']
        if target.exists() and digest(target) == item['sha256']:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix('.partial')
        with urlopen(item['url']) as response, open(partial, 'wb') as f:
            for block in iter(lambda: response.read(1 << 24), b''):
                f.write(block)
        if digest(partial) != item['sha256']:
            raise ValueError('Unexpected content: '+item['name'])
        partial.replace(target)
        print('Downloaded', item['name'], flush=True)
    print('All DFL files verified in', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('data/dfl'))
    main(p.parse_args().out)
