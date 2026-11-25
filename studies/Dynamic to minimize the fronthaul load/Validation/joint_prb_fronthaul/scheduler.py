"""Joint per-PRB candidate selection under integer per-radio budgets.

values : per-PRB candidate gains, [P][C_p] (ragged allowed). A 1-D array is shared by all PRBs.
costs  : per-PRB integer costs [P][C_p, R]; a single [C, R] array is shared by all PRBs.
budget : integer [R] radio budgets.
Both solvers return dict(indices, value, status, gap, runtime_s).
"""
import itertools
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack


def _prepare(values, costs, budget):
    budget = np.asarray(budget, dtype=np.int64)
    if np.ndim(costs[0]) == 1:  # one [C, R] table shared by every PRB
        n_prb = len(values) if np.ndim(values[0]) else None
        costs = [costs] * (n_prb if n_prb else 1)
    if np.ndim(values[0]) == 0:  # one gain vector shared by every PRB
        values = [values] * len(costs)
    vals = [np.asarray(v, dtype=float).ravel() for v in values]
    cst = [np.asarray(c, dtype=np.int64).reshape(len(v), -1) for v, c in zip(vals, costs)]
    return vals, cst, budget


def _fallback_info(fallback, vals, cst, budget):
    """Return (fallback array or None, its value, feasible?)."""
    if fallback is None:
        return None, -np.inf, False
    fb = np.asarray(fallback, dtype=np.int64)
    if fb.shape != (len(vals),) or any(i < 0 or i >= len(v) for v,i in zip(vals,fb)):
        raise ValueError("fallback must have one valid candidate index per PRB")
    used = sum(c[i] for c, i in zip(cst, fb))
    return fb, float(sum(v[i] for v, i in zip(vals, fb))), bool((used <= budget).all())


def _default_can_finish(cst, fb):
    """Check the fallback tail, then search the actual remaining cost vectors."""
    opts = []
    for c in cst:
        _, inv = np.unique(c > 0, axis=0, return_inverse=True)
        order = np.argsort(inv.ravel(), kind="stable")
        breaks = np.flatnonzero(np.diff(inv.ravel()[order])) + 1
        rows = []
        for group in np.split(c[order], breaks):
            lowest = group.min(0)
            if np.any(np.all(group == lowest, axis=1)):
                rows.append(lowest[None,:])
            else:
                rows.append(np.unique(group, axis=0))
        opts.append(np.concatenate(rows))
    floor = [np.min(o, axis=0) for o in opts]
    radios = [int((o>0).sum(axis=1).min()) for o in opts]
    min_unit = min((int(o[o>0].min()) for o in opts if np.any(o>0)), default=1)
    memo = {}

    def feasible(rem, blocks):
        if not blocks:
            return True
        if np.minimum(rem//min_unit,len(blocks)).sum() < sum(radios[b] for b in blocks):
            return False
        if (np.sum([floor[b] for b in blocks], axis=0) > rem).any():
            return False
        key = (tuple(rem), blocks)
        if key not in memo:
            eligible = [o[np.all(o <= rem, axis=1)] for o in (opts[b] for b in blocks)]
            idx = min(range(len(blocks)), key=lambda j:len(eligible[j]))
            tail = blocks[:idx] + blocks[idx+1:]
            memo[key] = all(len(e) for e in eligible) and any(
                feasible(rem - m, tail) for m in eligible[idx])
        return memo[key]

    def can_finish(after, next_blocks):
        blocks = tuple(next_blocks)
        return (not blocks or
                (fb is not None and np.all(after >= sum(cst[b][fb[b]] for b in blocks))) or
                feasible(after, blocks))

    return can_finish


def _greedy_pass(order, vals, cst, budget, can_finish, vector_check, by_gain):
    rem, pick = budget.copy(), np.full(len(vals), -1, dtype=np.int64)
    for i, p in enumerate(order):
        after = rem - cst[p]
        ok = (after >= 0).all(axis=1)
        tail = order[i+1:]
        if vector_check:
            ok &= np.asarray(can_finish(after, tail), bool)
            if not ok.any():
                return None
            k = int(np.argmax(np.where(ok, vals[p], -np.inf)))
        else:
            k = next((j for j in by_gain[p] if ok[j] and can_finish(after[j],tail)),None)
            if k is None:
                return None
        pick[p], rem = k, after[k]
    return pick


def greedy_select(values, costs, budget, **options):
    t0 = time.perf_counter()
    vals, cst, budget = _prepare(values, costs, budget)
    fb, _, fb_ok = _fallback_info(options.get("fallback"), vals, cst, budget)
    check = options.get("can_finish")
    can_finish = check or _default_can_finish(cst, fb if fb_ok else None)
    by_gain = [np.argsort(-v,kind="stable") for v in vals]
    best, best_val, best_order = None, -np.inf, None
    for order in itertools.permutations(range(len(vals))):
        pick = _greedy_pass(order, vals, cst, budget, can_finish, check is not None, by_gain)
        if pick is not None:
            val = float(sum(v[i] for v, i in zip(vals, pick)))
            if val > best_val:
                best, best_val, best_order = pick, val, order
    if best is None and fb_ok:
        best, best_val = fb, _fallback_info(fb, vals, cst, budget)[1]
        status = "fallback"
    else:
        status = "infeasible" if best is None else "feasible"
    return dict(indices=best, value=best_val if best is not None else None,
                status=status, gap=None, runtime_s=time.perf_counter() - t0,
                order=best_order if best_order is not None else [-1]*len(vals))


def joint_select(values, costs, budget, **options):
    t0 = time.perf_counter()
    vals, cst, budget = _prepare(values, costs, budget)
    fb, fb_val, fb_ok = _fallback_info(options.get("fallback"), vals, cst, budget)
    n_prb, n_rad = len(vals), len(budget)
    sizes = np.array([len(v) for v in vals])
    offs = np.concatenate([[0], np.cumsum(sizes)])
    n_var = int(offs[-1])
    flat = np.concatenate(vals)
    scale = float(np.abs(flat).max()) or 1.

    rows, cols = np.repeat(np.arange(n_prb), sizes), np.arange(n_var)
    pick_rows = coo_matrix((np.ones(n_var), (rows, cols)), shape=(n_prb, n_var))
    cost_mat = np.vstack(cst).T.astype(float)  # [R, n_var]
    cost_rows = coo_matrix(cost_mat)
    A = vstack([pick_rows, cost_rows]).tocsr()
    lo = np.concatenate([np.ones(n_prb), np.full(n_rad, -np.inf)])
    hi = np.concatenate([np.ones(n_prb), budget.astype(float)])

    opts = {}
    if options.get("time_limit") is not None:
        opts["time_limit"] = float(options["time_limit"])
    if options.get("mip_rel_gap") is not None:
        opts["mip_rel_gap"] = float(options["mip_rel_gap"])
    res = milp(-flat / scale, constraints=LinearConstraint(A, lo, hi),
               integrality=np.ones(n_var), bounds=Bounds(0, 1), options=opts)

    best, best_val = (fb, fb_val) if fb_ok else (None, -np.inf)
    incumbent = False
    if res.x is not None:  # optimal or time-limited incumbent: round and re-check
        pick = np.array([int(np.argmax(res.x[offs[p]:offs[p + 1]])) for p in range(n_prb)])
        used = sum(c[i] for c, i in zip(cst, pick))
        val = float(sum(v[i] for v, i in zip(vals, pick)))
        if (used <= budget).all() and val > best_val:
            best, best_val = pick, val
            incumbent = True
    names = {0: "optimal", 1: "time_limit", 2: "infeasible", 3: "unbounded"}
    status = names.get(res.status, "error")
    if best is None and status == "optimal":
        status = "invalid_incumbent"
    if best is not None and not incumbent and status != "optimal":
        status += "_with_fallback"
    dual = getattr(res, "mip_dual_bound", None)
    upper = -dual*scale if dual is not None and np.isfinite(dual) else None
    if best is None:
        gap = None
    else:
        if upper is None or upper < best_val - 1e-7*scale:
            status += "_relaxed_bound"
            upper = sum(v.max() for v in vals)
        gap = max(0., upper-best_val) / max(abs(best_val), 1e-12*scale)
    return dict(indices=best, value=best_val if best is not None else None, status=status,
                gap=None if gap is None else float(gap), runtime_s=time.perf_counter() - t0)
