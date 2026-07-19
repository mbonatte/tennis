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
