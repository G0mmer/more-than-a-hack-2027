"""Generate the eStart TCN baseline from tactical ATD. No labels are read."""
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from evaluation import possession_metrics as metrics
from games import GAME, GAMES, half_video_blocks, require
from starter.baseline import team_of_each_track
from .data import Tracking, sha256


DEFAULT_MODEL = Path(__file__).resolve().parent / "baseline_model"


def read_atd(meta, raw, frame_count, pitch):
    """Read original EPTS slots in metadata order. Missing rows retain their video index."""
    teams = team_of_each_track(meta)
    if set(teams.tolist()) != {0, 1}:
        raise ValueError("Tactical metadata must contain both teams")
    players = np.full((frame_count, len(teams), 2), np.nan, dtype=np.float32)
    ball = np.full((frame_count, 2), np.nan, dtype=np.float32)
    previous = -1
    with open(raw, encoding="utf-8-sig") as stream:
        for row, line in enumerate(stream, 1):
            try:
                frame_field, player_field, ball_field = line.rstrip().split(":")
                frame = int(frame_field) - 1
                if (row == 1 and frame != 0) or not previous < frame < frame_count:
                    raise ValueError("frame order or bounds")
                # Each slot is x,y,speed. Preserve outside-pitch observations and NaNs.
                # Parse every token: fromstring may silently ignore a malformed suffix.
                values = np.array([float(v) for v in player_field.rstrip(";").replace(";", ",").split(",")])
                ball_values = np.array([float(v) for v in ball_field.rstrip(";").split(",")])
                if values.size != len(teams) * 3 or ball_values.size != 3:
                    raise ValueError("unexpected EPTS columns")
                xy = values.reshape(-1, 3)[:, :2]
                if np.isinf(xy).any() or np.isinf(ball_values[:2]).any():
                    raise ValueError("infinite coordinates")
                players[frame] = xy * pitch
                ball[frame] = ball_values[:2] * pitch
                previous = frame
            except ValueError as error:
                raise ValueError(f"Invalid tactical ATD at row {row}") from error
    if previous != frame_count - 1:
        raise ValueError("Incomplete tactical tracking timeline")
    return ball, players, teams


def predict_halves(model, ball, players, teams, pitch, blocks):
    labels = np.full(len(ball), "X")
    previous = -1
    for half, first, last in blocks:
        if not previous < first <= last < len(ball):
            raise ValueError("Invalid half bounds on the tactical video timeline")
        rows = slice(first, last + 1)
        if not np.isfinite(players[rows]).all(axis=2).any():
            raise ValueError(f"No player observations in half {half}")
        tracking = Tracking({1: "firstHalf", 2: "secondHalf"}[half],
                            np.arange(first, last + 1), ball[rows], players[rows], teams, pitch)
        _, states = model.predict(tracking)
        if states.shape != (last - first + 1,) or not np.isin(states, [0, 1, 2]).all():
            raise ValueError("TCN returned an invalid decoded sequence")
        labels[rows] = np.array(["H", "A", "D"])[states]
        previous = last
    return labels


def write_validated(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = NamedTemporaryFile(dir=path.parent, mode="w", encoding="ascii", newline="",
                                delete=False, suffix=".tmp")
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write("frame,label\n")
            handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
        error = metrics.validate_submission(metrics.read_label_csv(temporary), len(labels))
        if error:
            raise ValueError(error)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Bundle root containing tactical/")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=Path("evaluation/results/eStart_Lodz.csv"))
    args = parser.parse_args()
    if args.out.suffix.lower() != ".csv":
        parser.error("--out must be a CSV file")
    import torch
    from .team_temporal import TeamTemporalBundle
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = TeamTemporalBundle.load(args.model)
    game = GAMES[GAME]
    meta, raw = require(["atd_meta", "atd_raw"], data_dir=args.data_dir)
    ball, players, teams = read_atd(meta, raw, game["n_video_frames"], game["pitch"])
    labels = predict_halves(model, ball, players, teams, game["pitch"], half_video_blocks(GAME))
    write_validated(args.out, labels)
    print(f"Validated {len(labels)} rows. SHA-256 {sha256(args.out)}")


if __name__ == "__main__":
    main()
