from dataclasses import dataclass

from tracking_postprocess import stabilize_player_roles


@dataclass
class Player:
    role: str
    track_id: int | None
    bbox: tuple[int, int, int, int]
    conf: float
    center: tuple[int, int]


def _top(track_id, center):
    x, y = center
    return Player("top_player", track_id, (x - 20, y - 50, x + 20, y + 50), 0.9, center)


def test_stabilize_player_roles_rejects_a_distant_one_frame_switch():
    tracks = [[_top(1, (500, 250))], [_top(99, (900, 250))], [_top(1, (520, 250))]]

    stabilized = stabilize_player_roles(tracks, (720, 1280, 3))

    assert [frame[0].center for frame in stabilized] == [(500, 250), (500, 250), (520, 250)]


def test_stabilize_player_roles_drops_a_stale_box_after_bounded_gap():
    tracks = [[_top(1, (500, 250))], [], [], []]

    stabilized = stabilize_player_roles(tracks, (720, 1280, 3), max_missing_frames=2)

    assert [len(frame) for frame in stabilized] == [1, 1, 1, 0]
