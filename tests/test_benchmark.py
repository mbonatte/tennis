from scripts.benchmark_pipeline import benchmark_scenario


def test_fake_pipeline_benchmark_reports_model_lifecycle_counts(sample_video):
    baseline = benchmark_scenario(sample_video, "reload_per_chunk_baseline", 4, fake_model_mb=1)
    persistent = benchmark_scenario(sample_video, "persistent_single_pass", 4, fake_model_mb=1)
    low_memory = benchmark_scenario(sample_video, "low_memory_multi_pass", 4, fake_model_mb=1)

    assert baseline.model_load_count == 9
    assert persistent.model_load_count == 3
    assert low_memory.model_load_count == 3
    assert {baseline.frame_count, persistent.frame_count, low_memory.frame_count} == {10}
    assert all(result.runtime_seconds > 0 for result in (baseline, persistent, low_memory))
    assert all(result.peak_rss_mb > 0 for result in (baseline, persistent, low_memory))
