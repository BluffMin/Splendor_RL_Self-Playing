from splendor_rl.progress import ProgressConfig, ProgressMode, make_training_progress


def test_resume_progress_uses_absolute_initial_and_stop_target():
    progress = make_training_progress(
        311_296,
        114_688,
        1_000_000,
        ProgressConfig(ProgressMode.NEVER),
    )
    assert progress.n == 114_688
    assert progress.total == 311_296
