from splendor_rl.models import PrivilegedCritic, SharedActor
from splendor_rl.population.config import ROLES, PopulationConfig
from splendor_rl.population.learner import make_learner


def test_trainable_roles_have_independent_parameters_and_optimizers():
    config=PopulationConfig(hidden_sizes=[8]); actor=SharedActor(475,373,[8]); critic=PrivilegedCritic(475,[8]); learners=[make_learner(role,actor.state_dict(),critic.state_dict(),config,{"actor":475,"critic":475,"action":373},"cpu") for role in ROLES]
    assert len({id(x.optimizer) for x in learners})==5 and len({id(x.actor) for x in learners})==5
