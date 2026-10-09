"""Evolutionary methods (PLAN.md §9).

- cmaes: full-covariance CMA-ES (evosax) on the hard score.
- cmaes_smooth: the same, ranked by the smooth loss, to separate the optimizer from the objective.
- ga: our own simple GA: tournament selection, arithmetic crossover, Gaussian mutation with a
  decaying step, elitism.

All start from the fall line plus noise and rank by minimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from evosax.algorithms import CMA_ES

from skiopt.budget import Budget
from skiopt.optim.common import Problem, Tracker


def _fitness(ev, objective: str):
    return ev.smooth_loss if objective == "smooth" else ev.hard_score


def cmaes(prob: Problem, method: dict, budget: Budget, key, log, init=None):
    objective = method.get("objective", "hard")
    pop = int(method.get("population", 32))
    sigma = float(method.get("sigma", 1.0))
    stop_at = float(method.get("_stop_at", budget.total))

    mean = prob.policy.zero() if init is None else jnp.asarray(init)
    es = CMA_ES(population_size=pop, solution=mean)
    es_params = es.default_params.replace(std_init=sigma)
    key, k_init = jax.random.split(key)
    state = es.init(k_init, mean, es_params)

    tracker = Tracker(log, budget)
    cost = pop * prob.n_courses
    while budget.used + cost <= stop_at:
        key, k_ask, k_tell = jax.random.split(key, 3)
        population, state = es.ask(k_ask, state, es_params)
        ev = prob.evaluate(population, prob.courses)
        budget.charge_rollouts(cost)
        state, _ = es.tell(k_tell, population, _fitness(ev, objective), state, es_params)
        tracker.update(population, ev, sigma=float(state.std))
    return tracker.result()


def cmaes_smooth(prob: Problem, method: dict, budget: Budget, key, log, init=None):
    return cmaes(prob, {**method, "objective": "smooth"}, budget, key, log, init)


def ga(prob: Problem, method: dict, budget: Budget, key, log, init=None):
    pop = int(method.get("population", 64))
    elite = int(method.get("elite", 4))
    tournament = int(method.get("tournament", 3))
    sigma0, sigma1 = float(method.get("sigma", 1.0)), float(method.get("sigma_final", 0.05))
    p_mut = float(method.get("p_mutate", 0.2))
    stop_at = float(method.get("_stop_at", budget.total))
    cost = pop * prob.n_courses
    n_params = prob.policy.n_params

    key, k0 = jax.random.split(key)
    if init is None:
        population = prob.init_population(k0, pop, sigma0)
    else:
        population = jnp.asarray(init) + sigma0 * jax.random.normal(k0, (pop, n_params))

    @jax.jit
    def breed(key, population, fitness, sigma):
        k_t, k_a, k_m, k_n = jax.random.split(key, 4)
        n_child = pop - elite
        order = jnp.argsort(fitness)
        elites = population[order[:elite]]
        # Tournament selection: each parent is the best of `tournament` random individuals.
        contenders = jax.random.randint(k_t, (2, n_child, tournament), 0, pop)
        winners = jnp.take_along_axis(contenders, jnp.argmin(fitness[contenders], axis=-1)[..., None], -1)[..., 0]
        a, b = population[winners[0]], population[winners[1]]
        alpha = jax.random.uniform(k_a, (n_child, 1))
        children = alpha * a + (1 - alpha) * b                       # arithmetic crossover
        mask = jax.random.uniform(k_m, children.shape) < p_mut
        children = children + mask * sigma * jax.random.normal(k_n, children.shape)
        return jnp.concatenate([elites, children])

    tracker = Tracker(log, budget)
    start_used = budget.used
    span = max(stop_at - start_used, 1.0)
    while budget.used + cost <= stop_at:
        ev = prob.evaluate(population, prob.courses)
        budget.charge_rollouts(cost)
        frac = (budget.used - start_used) / span
        sigma = sigma0 * (sigma1 / sigma0) ** frac
        tracker.update(population, ev, sigma=float(sigma))
        key, k = jax.random.split(key)
        population = breed(k, population, ev.hard_score, sigma)
    return tracker.result()
