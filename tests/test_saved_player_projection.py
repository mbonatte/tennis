import numpy as np

from analysis import project_player_tracks


def test_project_player_tracks_accepts_saved_json_player_dictionaries():
    projected = project_player_tracks(
        [[{"role": "bottom_player", "bbox": [10, 20, 30, 40]}]],
        [np.eye(3, dtype=np.float32)],
    )

    assert projected == [{"bottom_player": (20.0, 40.0)}]
