import copy

import jax.numpy as jnp
import pytest

from skiopt import courses, policies
from skiopt.config import load_config
from skiopt.objectives import make_trajectory
from skiopt.rollout import n_steps_for
from skiopt.types import SimParams

ALL_TERMS = ("gravity", "friction", "drag", "turn", "tight", "skid")


def make_params(on=ALL_TERMS, **sim_overrides) -> SimParams:
    """SimParams from configs/base.yaml with only the physics terms in `on` switched on."""
    cfg = copy.deepcopy(load_config())
    cfg["physics"]["terms"] = {t: t in on for t in ALL_TERMS}
    cfg["sim"].update(sim_overrides)
    return SimParams.from_config(cfg)


def run_line(kappa, p: SimParams, length=300.0, pitch_deg=20.0):
    """Roll out an open-loop curvature sequence (or a constant) on a constant-pitch slope."""
    n = n_steps_for(length, p.dy)
    kappa = jnp.broadcast_to(jnp.asarray(kappa, dtype=jnp.result_type(float)), (n,))
    course = courses.constant_pitch(length, pitch_deg)
    return make_trajectory(policies.OpenLoopKappa(n), p, n)(kappa, course)


@pytest.fixture
def base_params():
    return make_params()
