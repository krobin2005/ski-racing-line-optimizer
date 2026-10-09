"""Adam on the smooth loss (PLAN.md §9), with the gate penalty annealed upward.

`restarts` > 1 runs that many independent starts in parallel and shares the budget among them
(random-restart Adam). The champion is the best hard score seen at any evaluation."""

from __future__ import annotations

import jax
import numpy as np
import optax

from skiopt.budget import Budget
from skiopt.optim.common import Problem, Tracker


def adam(prob: Problem, method: dict, budget: Budget, key, log, init=None):
    restarts = int(method.get("restarts", 1))
    lr, lr_final = float(method.get("lr", 0.05)), float(method.get("lr_final", 0.005))
    lam0, lam1 = float(method.get("lam_gate_start", 0.1)), float(method.get("lam_gate_end", 100.0))
    clip = float(method.get("grad_clip", 10.0))
    eval_every = int(method.get("eval_every", 20))
    stop_at = float(method.get("_stop_at", budget.total))

    tracker = Tracker(log, budget)
    if init is None:
        params = prob.init_population(key, restarts, float(method.get("init_sigma", 0.5)))
    else:
        params = init[None].repeat(restarts, axis=0)

    C = prob.n_courses
    per_step = restarts * C * budget.grad_step_cost
    per_eval = restarts * C
    start_used = budget.used
    span = max(stop_at - start_used, 1.0)
    n_steps = max(int(span / (per_step + per_eval / eval_every)), 1)
    schedule = optax.exponential_decay(lr, n_steps, lr_final / lr)
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(schedule))
    state = opt.init(params)

    @jax.jit
    def update(params, grads, state):
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state

    step = 0
    while budget.used + per_step + per_eval <= stop_at:  # always leave room for a final evaluation
        frac = (budget.used - start_used) / span
        obj_t = prob.obj._replace(lam_gate=lam0 * (lam1 / lam0) ** frac)
        loss, grads = prob.value_and_grad(params, prob.courses, obj_t)
        budget.charge_rollouts(per_step)
        params, state = update(params, grads, state)
        step += 1
        if step % eval_every == 0 or budget.used + per_step + per_eval > stop_at:
            ev = prob.evaluate(params, prob.courses, obj_t)
            budget.charge_rollouts(per_eval)
            tracker.update(params, ev, step=step, lam_gate=float(obj_t.lam_gate),
                           smooth=float(np.min(np.asarray(loss))),
                           grad_norm=float(np.linalg.norm(np.asarray(grads), axis=-1).mean()))
    if tracker.best_params is None:  # budget too small for one evaluation cycle
        ev = prob.evaluate(params, prob.courses)
        budget.charge_rollouts(per_eval)
        tracker.update(params, ev, step=step)
    return tracker.result()
