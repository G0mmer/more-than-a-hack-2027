"""Streaming Sportec/DFL adapter; references never enter the model contract."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np


@dataclass(frozen=True)
class Tracking:
    period: str
    frame_ids: np.ndarray
    ball: np.ndarray
    players: np.ndarray
    teams: np.ndarray
    pitch: tuple
    fps: float = 25.0

    def __post_init__(self):
        n = len(self.frame_ids)
        if self.ball.shape != (n, 2) or self.players.shape != (n, len(self.teams), 2):
            raise ValueError('Tracking shape mismatch')
        if not n or np.any(np.diff(self.frame_ids) != 1):
            raise ValueError('Frames must form a nonempty contiguous timeline')
        if self.fps != 25 or not np.isin(self.teams, [0, 1]).all():
            raise ValueError('This first adapter requires 25 Hz and H/A team codes')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def truth_code(status, possession):
    if status == '0':
        return 2
    if status == '1' and possession in ('1', '2'):
        return int(possession) - 1
    return 3


def load_dfl(meta_path, positions_path):
    meta = ET.parse(meta_path).getroot()
    general = meta.find('.//General')
    env = meta.find('.//Environment')
    match_id = general.attrib['MatchId']
    team_map = {general.attrib['HomeTeamId']: 0, general.attrib['GuestTeamId']: 1}
    pitch = (float(env.attrib['PitchX']), float(env.attrib['PitchY']))
    records = {}
    current = None
    samples = []
    counts = {}
    for event, elem in ET.iterparse(positions_path, events=('start', 'end')):
        if event == 'start' and elem.tag == 'FrameSet':
            current = dict(elem.attrib)
            samples = []
            if current.get('MatchId') != match_id:
                raise ValueError('Metadata/positions match mismatch')
        elif event == 'end' and elem.tag == 'Frame':
            a = elem.attrib
            if current is None:
                raise ValueError('Frame outside FrameSet')
            # Ball identification uses semantic attributes, never player-list position.
            is_ball = 'BallStatus' in a or current.get('TeamId', '').lower() == 'ball' or current.get('PersonId', '').lower() == 'ball'
            keep = current.get('TeamId') in team_map or is_ball
            if keep:
                timestamp = datetime.fromisoformat(a['T']).timestamp()
                samples.append((int(a['N']), float(a.get('X', 'nan')), float(a.get('Y', 'nan')),
                                timestamp, truth_code(a.get('BallStatus'), a.get('BallPossession')) if is_ball else 3))
                if is_ball:
                    key = str((a.get('BallStatus'), a.get('BallPossession')))
                    counts[key] = counts.get(key, 0) + 1
                    current['_ball'] = True
            elem.clear()
        elif event == 'end' and elem.tag == 'FrameSet':
            if samples:
                arr = np.asarray(samples, dtype=np.float64)
                if np.any(np.diff(arr[:, 0]) <= 0):
                    raise ValueError('Duplicate or non-monotonic frames in track')
                periods = records.setdefault(current['GameSection'], [])
                periods.append((current, arr))
            current, samples = None, []
            elem.clear()

    output, audit = [], {'match_id': match_id, 'pitch_m': pitch, 'flag_counts': counts, 'periods': []}
    for period in ('firstHalf', 'secondHalf'):
        tracks = records.get(period)
        if not tracks:
            raise ValueError('Missing half: ' + period)
        balls = [arr for attr, arr in tracks if attr.get('_ball')]
        if len(balls) != 1:
            raise ValueError('Expected exactly one ball track per half')
        first = min(int(arr[0, 0]) for _, arr in tracks)
        last = max(int(arr[-1, 0]) for _, arr in tracks)
        ids = np.arange(first, last + 1)
        n = len(ids)
        ball_track = balls[0]
        origin = ball_track[0, 3] - (ball_track[0, 0] - first) / 25
        player_tracks = [(attr, arr) for attr, arr in tracks if not attr.get('_ball')]
        ball = np.full((n, 2), np.nan, dtype=np.float32)
        players = np.full((n, len(player_tracks), 2), np.nan, dtype=np.float32)
        truth = np.full(n, 3, dtype=np.int8)
        teams = []
        clock_error = 0.0
        for attr, arr in tracks:
            error = np.max(np.abs(arr[:, 3] - (origin + (arr[:, 0] - first) / 25)))
            clock_error = max(clock_error, float(error))
        if clock_error > 0.002:
            raise ValueError(f'Non-25Hz or inconsistent timestamps: {clock_error:.6f}s')
        idx = ball_track[:, 0].astype(int) - first
        ball[idx] = ball_track[:, 1:3]
        truth[idx] = ball_track[:, 4].astype(np.int8)
        for slot, (attr, arr) in enumerate(player_tracks):
            players[arr[:, 0].astype(int) - first, slot] = arr[:, 1:3]
            teams.append(team_map[attr['TeamId']])
        # Sportec uses the pitch centre as origin. Convert to [0, length] x [0, width].
        offset = np.asarray(pitch, dtype=np.float32) / 2
        ball += offset
        players += offset
        tracking = Tracking(period, ids, ball, players, np.asarray(teams), pitch)
        output.append((tracking, truth))
        audit['periods'].append({'period': period, 'frames': n, 'first_frame': first, 'last_frame': last,
            'clock_max_error_s': clock_error, 'gt_coverage': float(np.mean(truth != 3)),
            'ball_coverage': float(np.isfinite(ball).all(axis=1).mean()),
            'mean_players_visible': float(np.isfinite(players).all(axis=2).sum(axis=1).mean()),
            'gt_counts': {label: int(np.sum(truth == k)) for k, label in enumerate('HADX')},
            'ball_outside_fraction': float(np.mean(np.any((ball < 0) | (ball > np.asarray(pitch)), axis=1)))})
    unexpected = set(records) - {'firstHalf', 'secondHalf'}
    if unexpected:
        raise ValueError('Unsupported periods: ' + str(unexpected))
    return output, audit
