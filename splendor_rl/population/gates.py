def exploiter_acceptance(mean_score, lower_bound, *, min_score=0.55, min_lower=0.50):
    return mean_score >= min_score and lower_bound > min_lower


def population_promotion_decision(
    *,
    head_passed,
    anchors_passed,
    meta_delta,
    min_meta_delta=0.0,
    exploiter_regression=0.0,
    max_exploiter_regression=0.03,
):
    reasons = []
    if not head_passed:
        reasons.append("head_to_head_gate")
    if not anchors_passed:
        reasons.append("anchor_gate")
    if meta_delta < min_meta_delta:
        reasons.append("meta_strategy_regression")
    if exploiter_regression > max_exploiter_regression:
        reasons.append("exploiter_robustness_regression")
    return {"passed": not reasons, "reasons": reasons}
