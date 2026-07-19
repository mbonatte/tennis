import numpy as np
import pytest

player = pytest.importorskip("player")
torch = player.torch


class FakeBoxes:
    def __len__(self):
        return 0


class FakeResult:
    boxes = FakeBoxes()


class FakeInnerModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True


class FakeYolo:
    instances = []

    def __init__(self, path):
        self.path = path
        self.overrides = {}
        self.model = FakeInnerModel()
        self.device = None
        self.grad_states = []
        self.track_calls = 0
        self.instances.append(self)

    def to(self, device):
        self.device = device
        return self

    def track(self, frame, **kwargs):
        self.grad_states.append(torch.is_grad_enabled())
        self.track_calls += 1
        assert kwargs["persist"] is True
        assert kwargs["device"] == self.device
        return [FakeResult()]


def test_box_player_tracker_reuses_one_model_and_preserves_state(monkeypatch):
    FakeYolo.instances.clear()
    monkeypatch.setattr(player, "YOLO", FakeYolo)
    tracker = player.BoxPlayerTracker("players.pt", device="cpu")
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]

    first, _ = tracker.track_frames(frames[:2])
    second, _ = tracker.track_frames(frames[2:])

    assert len(FakeYolo.instances) == 1
    assert FakeYolo.instances[0].track_calls == 3
    assert FakeYolo.instances[0].grad_states == [False, False, False]
    assert FakeYolo.instances[0].model.eval_called
    assert len(first) == 2 and len(second) == 1
    tracker.close()
    assert tracker.model is None


def test_pose_models_are_not_loaded_when_pose_stage_is_disabled(monkeypatch):
    FakeYolo.instances.clear()
    monkeypatch.setattr(player, "YOLO", FakeYolo)

    player.BoxPlayerTracker("players.pt", device="cpu")

    assert [instance.path for instance in FakeYolo.instances] == ["players.pt"]


def test_hybrid_tracker_reads_each_selected_checkpoint_once(monkeypatch):
    FakeYolo.instances.clear()
    monkeypatch.setattr(player, "YOLO", FakeYolo)

    tracker = player.HybridPlayerTracker("players.pt", "pose.pt", device="cpu")

    assert [instance.path for instance in FakeYolo.instances] == ["players.pt", "pose.pt"]
    assert tracker.recovery_model is not tracker.model


def _detection(track_id, center, *, confidence=0.8):
    x, y = center
    return {
        "track_id": track_id,
        "bbox": (x - 20, y - 50, x + 20, y + 50),
        "conf": confidence,
        "center": center,
        "area": 4000,
    }


def _uninitialized_box_tracker():
    tracker = object.__new__(player.BoxPlayerTracker)
    tracker.last_top_player = None
    tracker.last_bottom_player = None
    tracker._top_missing_frames = 0
    tracker._bottom_missing_frames = 0
    tracker.max_missing_frames = 2
    return tracker


def test_player_assignment_prefers_previous_location_over_distant_distractor():
    tracker = _uninitialized_box_tracker()
    shape = (720, 1280, 3)
    tracker._assign_top_bottom_players([_detection(10, (500, 250))], shape)

    players = tracker._assign_top_bottom_players(
        [_detection(10, (520, 250), confidence=0.55), _detection(99, (900, 250), confidence=0.99)], shape
    )

    assert [(item.role, item.track_id, item.center) for item in players] == [("top_player", 10, (520, 250))]


def test_player_assignment_holds_briefly_then_drops_missing_player():
    tracker = _uninitialized_box_tracker()
    shape = (720, 1280, 3)
    tracker._assign_top_bottom_players([_detection(10, (500, 250))], shape)

    assert tracker._assign_top_bottom_players([], shape)[0].center == (500, 250)
    assert tracker._assign_top_bottom_players([], shape)[0].center == (500, 250)
    assert tracker._assign_top_bottom_players([], shape) == []
