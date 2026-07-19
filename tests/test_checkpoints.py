import pytest

from tennis_analyzer.checkpoints import normalize_state_dict


@pytest.mark.parametrize("wrapper", [None, "state_dict", "model_state_dict"])
def test_checkpoint_state_dict_wrappers_are_supported(wrapper):
    state = {"layer.weight": object()}
    checkpoint = state if wrapper is None else {wrapper: state, "epoch": 2}

    assert normalize_state_dict(checkpoint) == state


def test_dataparallel_prefix_is_removed_only_when_consistent():
    first, second = object(), object()

    assert normalize_state_dict({"module.one": first, "module.two": second}) == {"one": first, "two": second}
    assert normalize_state_dict({"module.one": first, "two": second}) == {"module.one": first, "two": second}


@pytest.mark.parametrize("checkpoint", [None, [], {}, {1: object()}])
def test_invalid_checkpoint_shapes_are_rejected(checkpoint):
    with pytest.raises(TypeError, match="checkpoint"):
        normalize_state_dict(checkpoint)
