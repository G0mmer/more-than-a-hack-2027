# Data sources, annotations and choice of inputs

Reader: organisers checking which data the model was built on and who labelled what.

## Rule boundary

No data about the provided match (BAR–ATH) was used to train, calibrate, select or tune any
model. No one labelled or corrected anything on that match. The public five-minute ground truth
was not used to compare versions. At inference the model reads only the tactical ATD tracking (XML and TXT).

## External datasets

| Dataset | Matches | Use in the submission | Source and licence |
|---|---|---|---|
| IDSSE / Sportec DFL (Bassek et al., 2025) | J03WMX, J03WN1, J03WPY, J03WOH, J03WOY | **All deployed weights.** Training, calibration and five whole-match validation folds | [figshare 28196177](https://doi.org/10.6084/m9.figshare.28196177), CC BY 4.0. Exact files and SHA-256: [training/dfl_sources.json](../training/dfl_sources.json) |
| SoccerTrack v2 | 118577, 128057 | External test blocks labelled by the team (below). Not used for fitting or calibration | [atomscott/soccertrack-v2](https://huggingface.co/datasets/atomscott/soccertrack-v2), CC BY 4.0, revision 325bd4c57ee8e135ab652f3921d274526ada496b |
| SoccerTrack v2 | 117092, 117093, 118575, 118576 | Team labels for development of dropped video and adaptation variants | same |
| SoccerTrack v2 | 118578 | Event labels (BAS) for a dropped weak-supervision restart head | same |
| Metrica Sports sample data | public sample games | Early tracking-physics experiments, not in the final model | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) |

DFL labels come from the dataset itself: `BallStatus` gives D, `BallPossession` gives H or A.
`training/prepare_dfl.py` converts them to the tracking frame axis.

## Our annotations

Team members labelled blocks of SoccerTrack video in a browser form, with model predictions
hidden. For each block they confirmed which team is home, then marked contiguous intervals
(`start_frame`, `end_frame_exclusive`, 25 Hz) as H, A, D or X.

Conventions:
- a controlled pass or a short loose ball keeps the last controlling team until a clear takeover;
- D starts at the stoppage and ends at the restart;
- an unresolved challenge or boundary is X.

Boundaries are browser-time estimates, so they are precise to a few frames at best.

| Match | Unique minutes | Reviewer-minutes | Role |
|---|---:|---:|---|
| 117092 | 3 | 3 | development |
| 117093 | 3 | 3 | development |
| 118575 | 2 | 4 | development, two independent reviews |
| 118576 | 15 | 15 | development diagnostics |
| 118577 | 15 | 15 | external test |
| 128057 | 10 | 10 | external test |
| **Total** | **48** | **50** | 16 block reviews |

Files: [annotation_manifest.json](annotation_manifest.json) (reviewer identifier, protocol,
source video hash, duration per block) and [annotations/](annotations/) (intervals per block).
Most blocks have one reviewer, so agreement between reviewers is measured only on 118575.

## Which inputs the model uses, and why

| Provided input | Used | Reason |
|---|---|---|
| Tactical ATD, players | yes | Available in every external training set in the same form (positions per slot). |
| Tactical ATD, ball | yes | Visible in 56.9% of tactical frames (organiser DATA.md). The model is trained with synthetic ball gaps and has a separate player-only path for long gaps. |
| Homography | no | ATD is already in pitch coordinates; homography is only needed to read the video. |
| Smart Tagging | no | No external training set has comparable automatic tags. DFL events are made by human analysts, so using them as a stand-in would train on better information than Smart Tagging provides. |
| Broadcast tracking | no | We built an automatic tactical-to-broadcast time alignment (65,868 frames matched) but did not measure any possession gain from it. |
| Tactical video | no | See below. |

### Why no video model

Rules forbid labelling the provided video or checking predictions against it, so a video model
has to learn entirely from other matches recorded with other cameras. Our external video with
possession labels was small (48 labelled minutes, most of it reserved for testing), and every
image signal we tried failed on it:

- image plus tracking fusion, trained on 3 minutes and tested on 3 minutes of another match
  (4,500 frames), lowered Spell F1 by 0.27 to 0.34 against tracking alone;
- frozen MobileNetV3 and DINOv2 features with a learned dead-ball head labelled every frame D,
  as did the time-shifted control;
- a pose model was matched or beaten by its own shuffled-pose control;
- an adapted ball detector (WASB) produced 22 usable ball positions out of 3,500 frames.

We did not estimate how much labelled video would be needed. With the data available, image
input was not reliable enough to add to a submission scored on an unseen camera.
