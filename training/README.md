# Rebuild the model weights

Reader: anyone who wants to check that `submission/model.zip` comes from public data only.
Every step uses the five public DFL matches. None reads data about the provided match.

Use the environment from the root `requirements.txt` (on macOS see the note in the root README).
Run every command from the repository root. Outputs go to the ignored folders `data/` and `runs/`.
Each step refuses to overwrite an existing output.

First unzip the shipped model once, for step 4: `python -m zipfile -e submission/model.zip runs/shipped`

| Step | Command | Time (CPU) |
|---|---|---|
| 1 Download DFL (1.8 GB) and check SHA-256; files already present with the right digest are kept | `python training/download_dfl.py --out data/dfl` | network |
| 2 Convert XML to per-half arrays with H/A/D labels and compare them with the reference | `python -B training/prepare_dfl.py --dfl data/dfl --out data/dfl_prepared` | 5 min |
| 3 Anchor TCN | `python -B training/train_anchor_tcn.py --inputs data/dfl_prepared --out runs/weights/anchor_tcn` | 3 min |
| 4 Motion CRF, dead-ball CRF and gate | `python -B -m possession_lab.motion_refit --dfl-root data/dfl --tcn runs/shipped/anchor/tcn --out data/external/runs/anchor_experts` | 10 min |
| 5 P1 team model | `python -B training/train_p1.py --inputs data/dfl_prepared --out runs/weights/p1` | 4 min |
| 6 Player-movement model | `python -B training/train_team_motion.py --inputs data/dfl_prepared --out runs/weights/team_motion.pt` | 2 min |
| 7 Stoppage and restart heads | `python -B training/train_restart.py --inputs data/dfl_prepared --out runs/weights/restart.json` | 2 min |
| 8 Assemble the archive | `python -B training/assemble_bundle.py --anchor-tcn runs/weights/anchor_tcn --anchor-experts data/external/runs/anchor_experts --p1 runs/weights/p1 --team-motion runs/weights/team_motion.pt --restart runs/weights/restart.json --out runs/weights/model.zip` | seconds |
| 9 Compare with the shipped archive | `python -B training/compare_bundles.py runs/weights/model.zip submission/model.zip` | seconds |

Step 4 writes under `data/external/runs/`: the CRF code accepts only that folder or a folder
outside the repository. Its `--tcn` selects the anchor TCN whose outputs the gate is fitted on.
The shipped one (`runs/shipped/anchor/tcn`) isolates the gate comparison;
`runs/weights/anchor_tcn` gives a fully retrained chain.

To see the effect on the output, write a CSV with the retrained archive and compare it with the
submitted one:

```text
python -B submission/make_submission.py --model runs/weights/model.zip --out runs/retrained.csv
```

## Expected result

Each part was trained once on one platform. Retrained on that platform, it comes out bit for bit
or within floating-point rounding; on another platform it differs.

| Part | Trained on | Windows x64 retrain | macOS ARM64 retrain |
|---|---|---|---|
| Prepared DFL arrays (step 2) | Windows x64 | identical | not checked |
| Anchor TCN (step 3) | macOS ARM64 | max abs diff 0.477 | **identical** |
| Motion CRF, dead-ball CRF, gate (step 4) | Windows x64 | **identical** | gate max abs diff 0.028 |
| P1 (step 5) | Colab GPU | max abs diff 0.316 | max abs diff 0.326 |
| Player-movement model (step 6) | Windows x64 | network **identical**, linear term within 1e-12 | max abs diff 0.160 |
| Stoppage and restart heads (step 7) | Windows x64 | within 1e-13 | within 2e-14 |

On a DFL-based test input, a CSV written with the fully retrained Windows archive agrees with the
shipped model on 99.77% of frames. All seeds are fixed in the scripts, and steps 3 to 7 run with
deterministic PyTorch on a fixed number of CPU threads.
