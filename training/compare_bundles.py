"""Step 9. Compare a retrained model archive with the shipped one, file by file.

Run from the repository root:
    python -B training/compare_bundles.py runs/weights/model.zip submission/model.zip
Reports exact equality, or the largest absolute difference of tensors, arrays and JSON numbers.
"""
import argparse
import io
import json
from zipfile import ZipFile

import numpy as np
import torch


def numbers(o):
    if isinstance(o, dict):
        return [x for k in sorted(o) for x in numbers(o[k])]
    if isinstance(o, list):
        return [x for v in o for x in numbers(v)]
    return [float(o)] if isinstance(o, (int, float)) and not isinstance(o, bool) else []


def tensors(o, prefix=''):
    if isinstance(o, torch.Tensor):
        yield prefix, o.double()
    elif isinstance(o, dict):
        for k in sorted(o):
            yield from tensors(o[k], f'{prefix}/{k}')
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield prefix, torch.tensor(float(o), dtype=torch.float64)


def difference(name, a, b):
    if a == b:
        return 'identical'
    if name.endswith('.pt'):
        x, y = (dict(tensors(torch.load(io.BytesIO(v), map_location='cpu', weights_only=False))) for v in (a, b))
        return f'max abs diff {max((x[k]-y[k]).abs().max().item() for k in y):.3g}' if x.keys() == y.keys() else 'different structure'
    if name.endswith('.npz'):
        with np.load(io.BytesIO(a)) as x, np.load(io.BytesIO(b)) as y:
            return f'max abs diff {max(float(np.abs(x[k].astype(float)-y[k].astype(float)).max()) for k in y.files):.3g}'
    if name.endswith('.json'):
        x, y = numbers(json.loads(a)), numbers(json.loads(b))
        if len(x) == len(y):
            return f'max abs diff {max((abs(p-q) for p, q in zip(x, y)), default=0.):.3g} (or metadata)'
    return 'different'


def main(a):
    new, old = ZipFile(a.retrained), ZipFile(a.shipped)
    for name in sorted(set(new.namelist()) | set(old.namelist())):
        if name.endswith('/'):
            continue
        if name not in new.namelist() or name not in old.namelist():
            print(f'{name:40s} only in one archive')
            continue
        print(f'{name:40s} {difference(name, new.read(name), old.read(name))}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('retrained')
    p.add_argument('shipped')
    main(p.parse_args())
