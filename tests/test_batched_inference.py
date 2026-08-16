import numpy as np
import torch


def test_batched_logits_and_legal_argmax_match_individual_inference():
    from league_helpers import models

    from splendor_rl.distributions import MaskedCategorical

    actor, _ = models()
    rng = np.random.default_rng(42)
    observations = torch.tensor(rng.normal(size=(7, 475)), dtype=torch.float32)
    masks = torch.tensor(rng.random((7, 373)) > 0.8)
    masks[:, 0] = True
    with torch.inference_mode():
        batched = MaskedCategorical(actor(observations), masks)
        batched_actions = batched.mode()
        individual_actions = torch.stack(
            [
                MaskedCategorical(
                    actor(observations[index : index + 1]),
                    masks[index : index + 1],
                ).mode()[0]
                for index in range(len(observations))
            ]
        )
    assert torch.equal(batched_actions, individual_actions)
    assert masks.gather(1, batched_actions[:, None]).all()

