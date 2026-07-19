from analysis import (
    ShotEvent,
    _estimate_pre_bounce_serve_hit,
    refine_bounce_frames,
    resolve_contact_like_bounces,
)


def test_contact_like_bounce_becomes_shot_when_reversal_reaches_player_box():
    ball = [(0.0, 0.0)] * 10 + [(95.0, 125.0), (100.0, 120.0), (105.0, 115.0), (110.0, 110.0)]
    players = [[] for _ in ball]
    players[12] = [{"role": "bottom_player", "bbox": [100, 100, 140, 150]}]

    bounces, shots = resolve_contact_like_bounces(
        {10},
        [ShotEvent(frame=11, player_role=None, ball_speed_kmh=None, reason="trajectory")],
        ball,
        players,
        fps=30,
    )

    assert bounces == set()
    assert [(shot.frame, shot.player_role, shot.reason) for shot in shots] == [
        (12, "bottom_player", "player_contact_reclassified_from_bounce")
    ]


def test_serve_contact_estimate_prefers_a_local_direction_reversal():
    ball = [(0.0, 0.0), (0.0, 10.0), (0.0, 20.0), (0.0, 30.0), (0.0, 25.0), (0.0, 20.0)]

    assert _estimate_pre_bounce_serve_hit(ball, bounce_frame=12, fps=30) == 3


def test_bounce_before_a_later_player_hit_is_not_suppressed():
    """A real bounce must survive even when the next hit is only 11 frames later."""
    shots = [
        ShotEvent(165, "bottom_player", None, "serve"),
        ShotEvent(296, "top_player", None, "player_contact_trajectory_recovery"),
    ]

    bounces = refine_bounce_frames(
        {238, 283},
        [(0.0, 0.0)] * 411,
        shots,
        fps=30,
    )

    assert bounces == {238, 283}
