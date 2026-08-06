from splendor_rl.league.bootstrap import AnchorGroups


def test_default_anchor_groups_are_separated():
    groups = AnchorGroups()
    assert groups.hard == ("greedy", "noble", "blocking")
    assert groups.saturated == ("random", "shortest")
    assert not set(groups.hard) & set(groups.saturated)
