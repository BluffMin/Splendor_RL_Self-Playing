from splendor_rl.orchestration import load_best_state, update_best_checkpoints


def summary(step, rank, win, score):
    stats = {
        "average_rank": rank,
        "fractional_first_place_rate": win,
        "average_score": score,
    }
    return {
        "transition_count": step,
        "aggregate": stats,
        "matchups": {"policy_vs_random": stats, "policy_vs_greedy": stats},
    }


def test_best_files_only_improve(tmp_path):
    source = tmp_path / "step.pt"
    source.write_bytes(b"one")
    state = {}
    assert (
        len(update_best_checkpoints(source, summary(1, 3, 0.1, 4), tmp_path, state))
        == 3
    )
    source.write_bytes(b"two")
    assert update_best_checkpoints(source, summary(2, 4, 0, 1), tmp_path, state) == []
    assert (
        tmp_path / "best_average_rank.pt"
    ).read_bytes() == b"one" and load_best_state(tmp_path)
