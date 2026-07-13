from __future__ import annotations

import numpy as np

from court_reference import CourtReference


def test_court_reference_builds_in_memory_without_runtime_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reference = CourtReference()

    assert reference.court.shape == (reference.court_total_height, reference.court_total_width)
    assert np.count_nonzero(reference.court) > 0
    assert not (tmp_path / "court_configurations").exists()
