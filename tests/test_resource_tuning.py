def _row(workers, speed, cpu, ram):
    return {
        "workers": workers,
        "envs_per_worker": 8,
        "learning_transitions_per_second": speed,
        "cpu_percent_mean": cpu,
        "cpu_percent_peak": cpu,
        "ram_percent_peak": ram,
        "free_ram_gb_minimum": 64 * (1 - ram / 100),
        "vram_gb_peak": 3,
        "gpu_temperature_c_peak": 55,
        "illegal_actions": 0,
        "invariant_violations": 0,
    }


def test_balanced_prefers_lower_resources_within_ten_percent():
    from splendor_rl.resource_tuning import select_recommendation

    rows = [_row(6, 400, 59, 58), _row(8, 425, 76, 66)]
    assert select_recommendation(rows, "balanced")["workers"] == 6
    assert select_recommendation(rows, "maximum")["workers"] == 8


def test_balanced_rejects_insufficient_foreground_ram_headroom():
    from splendor_rl.resource_tuning import select_recommendation

    rows = [_row(6, 390, 55, 60), _row(8, 420, 60, 75)]
    assert select_recommendation(rows, "balanced")["workers"] == 6
