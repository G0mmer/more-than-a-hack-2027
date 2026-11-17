# How models were evaluated and chosen

Reader: organisers and reviewers who want to know what the reported scores mean and which data
each decision was based on.

## Data that decisions were allowed to use

| Data | Role |
|---|---|
| DFL, five matches | Training, calibration and whole-match validation |
| SoccerTrack 118577 and 128057, four labelled blocks (30,000 frames) | External test, scored only with frozen models |
| BAR–ATH (provided match) | Never used for any score, comparison or choice. Its public five minutes were not run through the scorer for model comparison |

## Validation on DFL

- **Folds.** Five folds, each holding out one whole match (both halves). The held-out match
  is excluded from fitting and from calibration.
- **Inside a source half.** The first 80% is used for fitting and the last 20% for calibration
  (temperatures, gates), with 125 frames removed on each side of the split.
- **Final weights.** The same recipe is refitted once on all five matches. These weights are
  not scored on DFL, because no match is held out.
- **Equal weight per match.** Scores are averaged within a match first, then across matches.

### Input conditions

The same corrupted inputs are given to every model. Masks depend only on the seed, the match and
the half.

| Condition | What is changed |
|---|---|
| clean | nothing |
| missing20 / missing40 | ball removed in short runs of 13–50 frames until 20% / 40% of frames are missing |
| long40 | ball removed in runs of 50–200 frames until 40% of frames are missing |
| noise | 0.3 m Gaussian noise on ball and player positions |
| no_ball | ball removed for the whole half |

Mask seeds are 1000, 1001 and 1002, offset by `10 × match index + half index`. Training seeds
are fixed in the code: 7 for the networks and 711578 for the restart heads.

### Metrics

| Metric | Use |
|---|---|
| Spell F1 (official, `evaluation/possession_metrics.py`) | primary |
| Macro F1 over H, A, D (official) | secondary |
| D F1 | dead-ball quality |
| D episode recall (one-to-one match, IoU ≥ 0.3) | missed stoppages |
| False D seconds per minute of live play | false stoppages |
| Restart recall within ±1 s | restart timing |

## Selection rules

- **Anchor network recipe.** Eight recipes were trained with seed 7. A recipe could be promoted
  only if it passed every guard in each of three training seeds (7, 17, 27):
  - positive mean Spell F1 gain under missing-ball conditions, winning on at least 4 of 5
    matches;
  - clean Spell F1 loss of at most 0.005 on average and 0.03 per match;
  - D F1 loss of at most 0.005;
  - restart recall loss of at most 0.02 on average and 0.05 per case.

  Among the candidates that passed, the highest mean missing40/long40 Spell F1 wins. The
  deployed P1 recipe (`masked_aug`) passed in all three seeds. Seed 7 is deployed; seeds are
  not averaged.
- **Components.** Each component was tested in the DFL folds with the rest of the pipeline
  fixed. The H/A components are not allowed to change D labels, and the code checks this. Adding
  P1 with the D path fixed raised clean Spell F1 from 0.5799 to 0.6906 and long40 Spell F1 from
  0.4387 to 0.4970.
- **External test.** Each configuration was frozen before its external blocks were scored. The
  blocks have no ball coordinates, so they test the player-only path.

## Scores of the submitted pipeline

| Evaluation | Condition | Spell F1 |
|---|---|---:|
| DFL J03WMX held out, both halves (mask seed 1000 first half, 1001 second half) | long40 | 0.5069 |
| SoccerTrack 118577, two blocks | no ball | 0.4747 |
| SoccerTrack 128057, two blocks | no ball | 0.5296 |

These are development numbers on other providers' tracking, not a prediction of the competition
score.

## Limits of this evidence

- Five DFL matches from one league and one tracking provider.
- Synthetic ball gaps only approximate real tracking failures.
- External blocks: one reviewer each; they were scored more than once during development, so they
  are not a fully prospective test.
- No confidence intervals were computed across matches.
