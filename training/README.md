# Rebuild the model weights

Reader: anyone who wants to check that `submission/model.zip` comes from public data only.
Every step uses the five public DFL matches. None reads data about the provided match.

Use the environment from the root `requirements.txt`. Run every command from the repository
root. Outputs go to the ignored folders `data/` and `runs/`. Each step refuses to overwrite an
existing output.

| Step | Command | Time (CPU) |
|---|---|---|
| 1 Download DFL (about 1.9 GB) and check SHA-256 | `python training/download_dfl.py --out data/dfl` | network |
| 2 Convert XML to per-half arrays with H/A/D labels | `python -B training/prepare_dfl.py --dfl data/dfl --out data/dfl_prepared` | 5 min |
| 3 Anchor TCN | `python -B training/train_anchor_tcn.py --inputs data/dfl_prepared --out runs/weights/anchor_tcn` | 3 min |
| 4 Motion CRF, dead-ball CRF and gate | `python -B -m possession_lab.motion_refit --dfl-root data/dfl --tcn <anchor TCN folder> --out runs/weights/anchor_experts` | 10 min |
| 5 P1 team model | `python -B training/train_p1.py --inputs data/dfl_prepared --out runs/weights/p1` | 4 min |
| 6 Player-movement model | `python -B training/train_team_motion.py --inputs data/dfl_prepared --out runs/weights/team_motion.pt` | 2 min |
| 7 Stoppage and restart heads | `python -B training/train_restart.py --inputs data/dfl_prepared --out runs/weights/restart.json` | 2 min |
| 8 Assemble the archive | `python -B training/assemble_bundle.py --anchor-tcn runs/weights/anchor_tcn --anchor-experts runs/weights/anchor_experts --p1 runs/weights/p1 --team-motion runs/weights/team_motion.pt --restart runs/weights/restart.json --out runs/weights/model.zip` | seconds |
| 9 Compare with the shipped archive | `python -B training/compare_bundles.py runs/weights/model.zip submission/model.zip` | seconds |

In step 4, `--tcn` selects the anchor TCN whose outputs the gate is fitted on:
- the shipped one (unzip `submission/model.zip` and pass `anchor/tcn`) reproduces the shipped
  gate exactly;
- `runs/weights/anchor_tcn` gives a fully retrained chain.

To see the effect on the output, write a CSV with the retrained archive and compare it with the
submitted one:

```text
python -B submission/make_submission.py --model runs/weights/model.zip --out runs/retrained.csv
```

## Expected result on Windows x64 CPU

| Part | Against the shipped weights |
|---|---|
| Prepared DFL arrays (step 2) | identical bytes |
| Motion CRF, dead-ball CRF, gate (step 4 with the shipped anchor) | identical bytes |
| Player-movement network (step 6) | identical network weights; auxiliary linear term and its temperature within 1e-12 |
| Stoppage and restart heads (step 7) | within 1e-13 |
| Anchor TCN (step 3) | different weights: shipped ones were trained on macOS arm64 |
| P1 (step 5) | different weights: shipped ones were trained on a Colab GPU |
| Whole archive (step 8), CSV on a DFL-based test input | 99.77% of frames agree with the shipped model |

All seeds are fixed in the scripts. Steps 3 to 7 run with deterministic PyTorch on a fixed number
of CPU threads. The shipped anchor and P1 weights come from other hardware, so their retrained
versions agree closely with them but not bit for bit. Details and the effect on the CSV:
[docs/report.md](../docs/report.md), section 7.
