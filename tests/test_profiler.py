import time


def test_profiler_reports_seconds_calls_and_non_overlapping_wall_fraction(tmp_path):
    from splendor_rl.profiling import WallProfiler

    profiler = WallProfiler(True)
    with profiler.measure("environment_step"):
        time.sleep(0.001)
    profiler.add("actor_inference", 0.001, 2)
    report = profiler.report()
    assert report["components"]["environment_step"]["calls"] == 1
    assert report["components"]["actor_inference"]["calls"] == 2
    assert report["components"]["actor_inference"]["average_seconds"] == 0.0005
    assert sum(
        item["wall_fraction"] for item in report["components"].values()
    ) <= 1.000001
    profiler.write(tmp_path / "profile.json")
    assert (tmp_path / "profile.json").exists()
