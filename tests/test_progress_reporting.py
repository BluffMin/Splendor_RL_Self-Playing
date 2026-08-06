from splendor_rl.progress import NullProgress


def test_disabled_progress_tracks_count_and_closes():
    progress = NullProgress()
    progress.total = 20
    progress.update_training(7, transitions=7, update_index=1, episodes=0, metrics={})
    progress.status("updating")
    progress.close()
    assert progress.n == 7
