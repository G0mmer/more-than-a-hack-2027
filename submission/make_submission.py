"""Generate the possession CSV for the provided match from the frozen model in this folder.

Run from the repository root:
    python -B submission/make_submission.py --data-dir data --out submission/eStart_Lodz.csv
Reads only data/tactical/BAR-ATH_tactical_atd.xml and BAR-ATH_tactical_atd.txt. Checks the model
archive digest, runs on CPU with deterministic PyTorch, validates the CSV and prints its SHA-256.
"""
import argparse
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession_lab.data import sha256  # noqa: E402
from possession_lab.submission import write_validated  # noqa: E402
from possession_lab.submission_candidate import CandidateBundle, configure_cpu, predict_match  # noqa: E402

MODEL = Path(__file__).with_name('model.zip')
MODEL_SHA256 = '9b67f97fee21467944354010f1aac3ba00eb0b5bd2884209d0319eb1a15dca81'
VARIANT = ('dfl_restart', 'hybrid')


def main(a):
    if a.out.exists() or a.out.suffix.lower() != '.csv':
        raise ValueError('Use a new CSV output path')
    if a.model == MODEL and sha256(MODEL) != MODEL_SHA256:
        raise ValueError('submission/model.zip differs from the frozen model')
    if a.model != MODEL:
        print('Using a model archive other than the frozen one, SHA-256', sha256(a.model), flush=True)
    configure_cpu()
    with tempfile.TemporaryDirectory() as tmp:
        ZipFile(a.model).extractall(tmp)
        labels = predict_match(CandidateBundle(tmp), a.data_dir, [VARIANT])[VARIANT]
    write_validated(a.out, labels)
    print('Validated', len(labels), 'rows, SHA-256', sha256(a.out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--out', type=Path, default=Path('submission/eStart_Lodz.csv'))
    p.add_argument('--model', type=Path, default=MODEL,
                   help='Model archive; defaults to the frozen submission model')
    main(p.parse_args())
