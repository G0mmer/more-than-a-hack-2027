"""Step 8. Assemble trained parts into the model archive read by submission/make_submission.py.

Run from the repository root:
    python -B training/assemble_bundle.py --anchor-tcn runs/weights/anchor_tcn \
        --anchor-experts runs/weights/anchor_experts --p1 runs/weights/p1 \
        --team-motion runs/weights/team_motion.pt --restart runs/weights/restart.json \
        --out runs/weights/model.zip
The archive is deterministic: sorted entries with a fixed timestamp.
"""
import argparse
import json
import shutil
from pathlib import Path
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession_lab.data import sha256  # noqa: E402
from possession_lab.ensemble_predict import EnsembleBundle  # noqa: E402
from possession_lab.multimatch_run import MATCHES  # noqa: E402

TCN_FILES = ('bundle.json', 'complete.json', 'model.pt', 'scaler.npz')
EXPERT_FILES = {'motion_A': ('complete.json', 'config.json', 'model.crfsuite', 'scaler.npz'),
                'dead_specialist': ('complete.json', 'config.json', 'model.crfsuite', 'scaler.npz'),
                'gate_dead': ('complete.json', 'config.json', 'gate.npz')}


def copy(source, target, names):
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(source/name, target/name)


def digests(root):
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != 'manifest.json'}


def write_zip(root, out):
    with ZipFile(out, 'w', ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                info = ZipInfo(p.relative_to(root).as_posix(), (1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                z.writestr(info, p.read_bytes())


def assemble(a, root):
    copy(a.anchor_tcn, root/'anchor/tcn', TCN_FILES)
    for part, names in EXPERT_FILES.items():
        copy(a.anchor_experts/part, root/'anchor'/part, names)
    copy(a.p1, root/'p1', TCN_FILES)
    shutil.copyfile(a.team_motion, root/'team_motion.pt')
    restart = json.loads(a.restart.read_text())
    (root/'restart.json').write_text(json.dumps({'dfl_restart': restart['dfl_restart']}))
    anchor = dict(schema='all-five-expert-package-v1', training_matches=list(MATCHES),
                  artifact_sha256=digests(root/'anchor'))
    (root/'anchor/complete.json').write_text(json.dumps(anchor, indent=2))
    dfl = dict(dataset='DFL', matches=list(MATCHES))
    manifest = dict(schema='submission-candidates-v1',
                    anchor_signature=EnsembleBundle(root/'anchor', 'moe_dead').model_signature,
                    inputs=dict(anchor=dfl, p1=dict(dfl, seed=7), team_motion=dfl, restart=dfl),
                    sha256=digests(root))
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))


def main(a):
    if a.out.exists():
        raise ValueError('Use a new output path')
    with tempfile.TemporaryDirectory() as tmp:
        assemble(a, Path(tmp))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        write_zip(Path(tmp), a.out)
    print('Model archive', a.out, 'SHA-256', sha256(a.out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('anchor-tcn', 'anchor-experts', 'p1', 'team-motion', 'restart', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
