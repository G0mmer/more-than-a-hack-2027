"""Step 2. Convert the raw DFL XML into per-half tracking arrays with H/A/D labels.

Run from the repository root: python -B training/prepare_dfl.py --dfl data/dfl --out data/dfl_prepared
The output folder holds dfl/<match>_<half>.npz and a manifest with their SHA-256. Each half is
also compared with the reference digests of the arrays used for the shipped weights
(training/dfl_prepared_reference.json); file bytes can differ across platforms through
compression, so the arrays are compared, not the files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession_lab.data import sha256  # noqa: E402
from possession_lab.multimatch_run import MATCHES  # noqa: E402
from possession_lab.robust_run import read_match  # noqa: E402

REFERENCE = Path(__file__).with_name('dfl_prepared_reference.json')


def array_digest(path):
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for key in sorted(z.files):
            a = np.ascontiguousarray(z[key])
            h.update(f'{key}|{a.dtype.str}|{a.shape}|'.encode())
            h.update(a.tobytes())
    return h.hexdigest()


def main(a):
    if a.out.exists():
        raise ValueError('Use a new output folder')
    reference = json.loads(REFERENCE.read_text())
    files, mismatched = {}, []
    for match in MATCHES:
        halves, _, _ = read_match(a.dfl, match)
        for d, y in halves:
            name = f'dfl/{match}_{d.period}.npz'
            (a.out/'dfl').mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.out/name, period=d.period, frame_ids=d.frame_ids, ball=d.ball,
                                players=d.players, teams=d.teams, pitch=d.pitch, fps=d.fps, truth=y)
            files[name] = sha256(a.out/name)
            if array_digest(a.out/name) != reference[name]:
                mismatched.append(name)
        print('Prepared', match, flush=True)
    manifest = dict(schema='external-final-round-input-v1',
                    data_sources=['Sportec/DFL CC BY 4.0'], sha256=files)
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if mismatched:
        print('Arrays differ from the reference:', ', '.join(mismatched), flush=True)
    else:
        print('All 10 halves identical to the arrays used for the shipped weights', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dfl', type=Path, default=Path('data/dfl'))
    p.add_argument('--out', type=Path, default=Path('data/dfl_prepared'))
    main(p.parse_args())
