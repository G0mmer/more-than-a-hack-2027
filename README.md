# eStart Lodz — More than a Hack 2027, ball possession

This repository is our submission to the
[More than a Hack 2027 initial challenge](https://github.com/metrica-sports/more-than-a-hack-2027).
It is a fork of the challenge repository. `CHALLENGE.md`, `RULES.md`, `SUBMISSION.md`, `DATA.md`,
`EVALUATION.md`, `FAQ.md`, `games.py`, `evaluation/` and `starter/` are the organisers' files,
unchanged. This README replaces the challenge README; the original is in the upstream repository.

The model labels every frame of the tactical video as home possession (H), away possession (A),
dead ball (D) or outside the halves (X), using only the automatic tracking of the match. All
weights were trained on five public Bundesliga matches. No data about the provided match was
used for training, calibration or model choice.

| I want to | Go to |
|---|---|
| regenerate the submitted CSV | [below](#regenerate-the-submission) |
| rebuild the model weights from public data | [training/README.md](training/README.md) |
| understand what the model does | [docs/model_overview.md](docs/model_overview.md) |
| find the code behind each step | [docs/code_overview.md](docs/code_overview.md) |
| read the technical report | [docs/report.md](docs/report.md), [PDF](docs/eStart_Lodz.pdf) |
| see the data sources and our annotations | [docs/data_and_annotations.md](docs/data_and_annotations.md) |
| see how models were evaluated | [docs/evaluation_protocol.md](docs/evaluation_protocol.md) |

## Regenerate the submission

Tested with 64-bit Python 3.13.11 on Windows x64, CPU only.

1. Unpack the challenge data bundle into `data/` as described in [DATA.md](DATA.md). Only
   `data/tactical/BAR-ATH_tactical_atd.xml` and `data/tactical/BAR-ATH_tactical_atd.txt` are read.
2. Install and run from the repository root:

```text
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -B submission/make_submission.py --data-dir data --out submission/eStart_Lodz.csv
.venv/Scripts/python evaluation/validate_submission.py submission/eStart_Lodz.csv
```

On Linux or macOS use `.venv/bin/python`. On Windows, keep the repository in a short path such
as `C:\estart`: PyTorch fails to install in deeply nested folders unless long paths are enabled.

`make_submission.py` checks the SHA-256 of `submission/model.zip`, runs on CPU with deterministic
PyTorch operations, validates the file and prints its SHA-256.

| File | SHA-256 |
|---|---|
| `submission/model.zip` | `9b67f97fee21467944354010f1aac3ba00eb0b5bd2884209d0319eb1a15dca81` |
| submitted `eStart_Lodz.csv` | `a6fc0521798d37001ea76f33a722185172de45153c31e876a4954f324cf3f6af` |

Byte-identical output is verified on Windows x64 CPU. Other platforms may differ in a small
number of frames because of floating-point differences.

## Repository layout

```text
submission/       frozen model and the script that writes the CSV
training/         scripts that rebuild submission/model.zip from public DFL data
possession_lab/   library used by both
docs/             report, overviews, data sources, annotations, evaluation protocol
evaluation/, starter/, games.py, root *.md files   organiser files, unchanged
```

## Licence and data

Code: see [LICENSE](LICENSE). The match data is not included; `data/` is ignored by git.
External datasets are cited in [docs/data_and_annotations.md](docs/data_and_annotations.md).
