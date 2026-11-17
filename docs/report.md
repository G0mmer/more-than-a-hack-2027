# eStart Lodz: possession from tracking alone

Technical report, More than a Hack 2027 initial challenge, 17 November 2026.

## 1. Approach

We label each frame of the tactical video as H, A, D or X from the automatic tracking only:
the ball and player positions in the tactical ATD files. We do not use the video, homography,
Smart Tagging or the broadcast recording.

The reason is the training data. Rules forbid labels on the provided match, so every model
must learn from other matches. Five public Bundesliga matches (DFL) come with frame-level ball
status and possession, in the same form as our input: player and ball positions per frame. We
found no comparable public source for video, and image models trained on our own small set of
labelled external video did not transfer (section 2).

The ball is visible in 56.9% of tactical frames (organiser `DATA.md`), so the model must keep
working through long ball gaps. The pipeline therefore combines several sources of evidence:

1. **Anchor.** A temporal convolutional network (TCN, 8 s of context, team geometry) and two
   conditional random fields (CRF) on player and ball motion produce H/A/D probabilities at
   12.5 Hz. A gate learned on DFL weights them by ball visibility and player coverage. A
   temporal decoder (chain strength 0.25) turns them into spells.
2. **Extra stoppages.** Two L2 logistic regressions on player motion at 5 Hz score each moment
    for "dead ball" and "restart". An interval is added as D only if:

    - player coverage is at least 0.80;
    - mean state and nearby restart evidence are at least 0.61, with a state contrast of at
      least 0.055, and the state changes around the interval;
    - it is 3 to 23 seconds long, at most one per six-minute cell, with a 50-frame margin to
      existing D;
    - additions per half stay within `max(250, floor(0.015 × half frames))` frames.

    D frames of the anchor are never removed.

3. **Home or away on live frames.** Where the ball and at least five players per team are
   visible, a team-symmetric TCN (P1) is blended in with weight 0.5. After the ball has been
   missing for one second or more, a player-movement network is blended in with weight 0.5.
   Neither step may change a D label; the code raises an error if one does.
4. **Output.** Labels are repeated to 25 Hz. Frames outside the two halves, as defined in the
   organiser's `games.py`, get X.

Diagrams: [model_overview.md](model_overview.md) (what the model does) and
[code_overview.md](code_overview.md) (where it is in the code).

## 2. Assets used and dropped

| Asset | Status | Reason |
|---|---|---|
| Tactical ATD, players and ball | used | Same form as the DFL training data |
| Homography | not used | ATD is already in pitch coordinates |
| Smart Tagging | not used | No training set with comparable automatic tags. DFL events come from human analysts, which would train on better information than Smart Tagging gives |
| Broadcast tracking | not used | We built an automatic time alignment to the tactical video (65,868 frames matched) but measured no possession gain |
| Tactical video, image models | dropped | See below |
| Restart head refitted on our manual labels | dropped | Lowered Spell F1 on an external test match from 0.5296 to 0.4901 |
| Restart heads with SoccerTrack event labels (BAS) | dropped | No consistent gain on external matches |

Image models were tested only on external SoccerTrack video, because the provided video may not
be labelled. Our labelled external video was small: 48 unique minutes, most of it reserved for
testing.

- Fusing image features with tracking, trained on 3 minutes and tested on 3 minutes of another
  match (4,500 frames), lowered Spell F1 by 0.27 to 0.34.
- Frozen MobileNetV3 and DINOv2 features with a learned dead-ball head labelled every frame D,
  as did the time-shifted control.
- A pose model did no better than its shuffled-pose control.
- An adapted ball detector (WASB) recovered 22 usable ball positions out of 3,500 frames.

We did not estimate how much labelled video would be needed for a transferable image model.

## 3. Training data

| Data | Amount | Labels | Used for |
|---|---|---|---|
| DFL / Sportec (Bassek et al., 2025), J03WMX, J03WN1, J03WPY, J03WOH, J03WOY | 5 matches, 10 halves | `BallStatus` and `BallPossession` from the dataset | all deployed weights |
| SoccerTrack v2, 118577 and 128057 | 4 blocks, 30,000 frames | labelled by our team, one reviewer per block | external test only |
| SoccerTrack v2, 117092, 117093, 118575, 118576 | 23 unique minutes | labelled by our team | development of dropped variants |

The team labelled 48 unique minutes in total (50 reviewer-minutes, 16 block reviews) in a browser
form, with predictions hidden. The protocol, reviewer identifiers and intervals are in
[data_and_annotations.md](data_and_annotations.md) and `docs/annotations/`. None of these labels
are on the provided match, and none were used to fit the deployed weights.

## 4. Possession rules

The deployed model learns possession from DFL labels: a team keeps possession from control of
the ball until the opponent gains control, and D covers the time the ball is out of play.
Concretely:

- **Loose balls and challenges:** there is no separate detector. The decoder keeps the previous
  team until the evidence favours the opponent, so a short loose ball or a lost challenge
  stays with the previous team until the takeover is clear.
- **Restarts:** D ends when play resumes. Stoppages are recognised from tracking patterns: the
  ball leaving the pitch, and players slowing down and then accelerating together.
- **Goals:** treated as stoppages. A quick restart in which players keep moving can be missed.
- **Halftime and frames outside the halves:** X. Stoppages inside a half are D.

Our manual labels for external testing follow the same rules, with X for moments the reviewer
could not resolve.

## 5. Evaluation

Full rules: [evaluation_protocol.md](evaluation_protocol.md).

- Every decision used DFL whole-match folds (one match held out from fitting and calibration)
  or frozen models on the external SoccerTrack blocks.
- The provided match, including its public five minutes, was not used for any comparison.
- The primary metric is the official Spell F1, with macro F1, D F1, missed and false stoppages,
  and restart timing as diagnostics.
- Input conditions add synthetic ball gaps and position noise with fixed seeds.

| Evaluation | Spell F1 |
|---|---:|
| DFL J03WMX held out, long ball gaps (40%) | 0.5069 |
| SoccerTrack 118577, player-only tracking | 0.4747 |
| SoccerTrack 128057, player-only tracking | 0.5296 |

Within the DFL folds, adding P1 with the D path fixed raised clean Spell F1 from 0.5799 to
0.6906, and with long ball gaps from 0.4387 to 0.4970.

## 6. Limitations and further work

- **Training data.** Five matches, one league, one tracking provider. Tracking conventions of
  the provided match differ.
- **Ball gaps.** Synthetic gaps only approximate real tracking failures.
- **External test.** Two matches, one reviewer per block, scored more than once during
  development. No confidence intervals were computed.
- **Stoppages.** Slow live play can be read as a stoppage, and quick restarts can be missed. H/A
  refinement cannot repair a wrong D boundary.
- **Missing ball.** Without the ball, H/A relies on player movement only, and track-slot identity
  switches corrupt those features.
- **Team-swap symmetry.** The D path is not team-swap symmetric. Swapping the teams changed 254 D
  labels in 145,967 frames.

With more time we would label more external matches with two reviewers each, measure how many
labelled video minutes an image model needs, and test the model on new matches frozen in advance.

## 7. Reproduction

**Submission.** Run `submission/make_submission.py` (README). It verifies the model archive
digest, runs on CPU with deterministic PyTorch and validates the CSV. In fresh environments
installed from `requirements.txt` on Windows x64, repeated runs wrote identical bytes. The SHA-256
of the submitted CSV is listed in the README.

**Weights.** `training/` rebuilds the model archive from the public DFL files in nine numbered
steps. On Windows x64 CPU, retraining from the raw files gave:

| Part | Result against the shipped weights |
|---|---|
| Prepared DFL arrays | identical bytes |
| Motion CRF, dead-ball CRF, gate | identical bytes |
| Player-movement network | identical network weights; auxiliary linear term and its temperature differ by at most 1e-12 |
| Stoppage and restart heads | weights differ by at most 1e-13 |
| Anchor TCN | different weights. Shipped weights were trained on macOS arm64; our input scaler differs from theirs by one unit in the last place, and the difference grows during training. Three of our environments (Python 3.12 and 3.13, two NumPy versions) gave the same maximum difference |
| P1 TCN | different weights. Shipped weights were trained on a Colab GPU |

A CSV from the retrained parts agrees with the CSV from the shipped weights on 99.77% of frames
of a test input built from DFL tracking. Using the retrained restart and movement parts alone
gives the identical CSV. Retraining therefore confirms where the weights come from. The
submission is defined by the shipped weights.

## Sources

Bassek, M. et al. (2025). An integrated dataset of spatiotemporal and event data in elite
soccer. *Scientific Data*. DOI 10.1038/s41597-025-04505-y. Data: DOI 10.6084/m9.figshare.28196177.

SoccerTrack v2, https://huggingface.co/datasets/atomscott/soccertrack-v2, revision
325bd4c57ee8e135ab652f3921d274526ada496b.
