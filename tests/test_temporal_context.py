import pytest

from tennis_analyzer.pipeline.temporal import TemporalContextBuffer


@pytest.mark.parametrize(
    ("chunks", "expected_inputs", "expected_trimmed"),
    [
        ([[0]], [[0]], [[0]]),
        ([[0], [1]], [[0], [0, 1]], [[0], [1]]),
        ([[0], [1], [2]], [[0], [0, 1], [0, 1, 2]], [[0], [1], [2]]),
        ([[0, 1], [2]], [[0, 1], [0, 1, 2]], [[0, 1], [2]]),
        ([[0, 1, 2], [3]], [[0, 1, 2], [1, 2, 3]], [[0, 1, 2], [3]]),
        ([[0, 1, 2], [3, 4], [5]], [[0, 1, 2], [1, 2, 3, 4], [3, 4, 5]], [[0, 1, 2], [3, 4], [5]]),
    ],
)
def test_temporal_context_preserves_boundaries_without_duplicate_outputs(chunks, expected_inputs, expected_trimmed):
    context = TemporalContextBuffer[int](context_size=2)
    actual_inputs = []
    actual_trimmed = []

    for chunk in chunks:
        combined, overlap = context.prepend(chunk)
        actual_inputs.append(combined)
        actual_trimmed.append(combined[overlap:])

    assert actual_inputs == expected_inputs
    assert actual_trimmed == expected_trimmed
    assert [item for chunk in actual_trimmed for item in chunk] == [item for chunk in chunks for item in chunk]


def test_empty_final_chunk_does_not_duplicate_context():
    context = TemporalContextBuffer[int](context_size=2)
    context.prepend([0, 1, 2])

    combined, overlap = context.prepend([])

    assert combined == [1, 2]
    assert combined[overlap:] == []


def test_context_can_be_released():
    context = TemporalContextBuffer[int](context_size=2)
    context.prepend([0, 1, 2])
    context.clear()

    assert context.prepend([3]) == ([3], 0)


def test_negative_context_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        TemporalContextBuffer(context_size=-1)
