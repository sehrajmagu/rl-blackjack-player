"""Policy iteration that does NOT assume a stable policy exists.

Run with:  python -m src.policy_iter.policy_iteration_non_stationary [--state TOTAL SOFT UPCARD] [--random N] [--verbose] [--actions] [--fundamental] [--table]

What it does
  1. Starts from simple policies: HIT in every state (STAND where hitting is not
     allowed, i.e. 21) and STAND in every state. Optionally N random mixes.
  2. Repeats evaluate -> improve, recording the whole trajectory of policies.
     After every update it checks whether the policy
        - stopped changing      (fixed point), or
        - returned to a policy it already visited (cycle).
     Neither is assumed: the loop reports whichever actually happens.
  3. Compares where the different starts end up and prints the best end point,
     with sweeps and runtime per start.
  4. Optionally (--fundamental, --table) analyses the best policy as an absorbing
     chain: fundamental matrix, expected steps, and win/tie/lose probabilities.

Self-contained: only depends on the MDP definition (model.py, rules.py), not on
any other policy_iter file.

Note on cycles: with "change only if strictly cheaper" (as in improve_policy) the
expected return can never go down between iterations, so a cycle cannot occur
for this MDP. The detector is there so that claim is checked, not assumed.
"""

import argparse
import random
import time

import numpy as np

from ..model import GAMMA, HIT, STAND, STATES, TERMINAL, _describe, build_model
from ..rules import CARD_PROBS

ACTION_NAME = {STAND: "STAND", HIT: "HIT"}
OUTCOMES = ("win", "tie", "lose")
TOL = 1e-12
IDX = {s: i for i, s in enumerate(STATES)}


# ----------------------------------------------------------------------------
# Policy evaluation and improvement
# ----------------------------------------------------------------------------
def evaluate_policy(P, policy, gamma=GAMMA):
    """V_pi: solve (I - gamma*Q) V = r, where Q is the game-state block of P_pi."""
    n = len(STATES)
    Q = np.zeros((n, n))
    r = np.zeros(n)
    for s in STATES:
        i = IDX[s]
        for p, nxt, reward in P[s][policy[s]]:
            r[i] += p * reward
            if nxt != TERMINAL:
                Q[i, IDX[nxt]] += p
    v = np.linalg.solve(np.eye(n) - gamma * Q, r)
    return {s: v[IDX[s]] for s in STATES}


def q_value(P, V, state, action, gamma=GAMMA):
    return sum(
        p * (reward + (0.0 if nxt == TERMINAL else gamma * V[nxt]))
        for p, nxt, reward in P[state][action]
    )


def improve_policy(P, V, policy):
    """Greedy improvement, one state at a time. Returns (new_policy, changed)."""
    new_policy, changed = {}, False
    for s in STATES:
        costs = {a: -q_value(P, V, s, a) for a in P[s]}
        best = policy[s]
        for a, c in costs.items():  # keep current action unless strictly cheaper
            if c < costs[best] - TOL:
                best = a
        if best != policy[s]:
            changed = True
        new_policy[s] = best
    return new_policy, changed


# ----------------------------------------------------------------------------
# Starting policies
# ----------------------------------------------------------------------------
def start_policy(P, kind, rng=None):
    policy = {}
    for s in STATES:
        if kind == "stand":
            policy[s] = STAND
        elif kind == "hit":
            policy[s] = HIT if HIT in P[s] else STAND  # 21 can only stand
        else:  # random mix
            policy[s] = rng.choice(list(P[s]))
    return policy


def _key(policy):
    return tuple(policy[s] for s in STATES)


# ----------------------------------------------------------------------------
# Value of a freshly dealt hand
# ----------------------------------------------------------------------------
def start_distribution():
    """P(first decision state) from a dealt two-card hand vs a random upcard."""
    start = {}
    for c1, p1 in CARD_PROBS.items():
        for c2, p2 in CARD_PROBS.items():
            if c1 == 1 and c2 == 1:
                total, soft = 12, True
            elif c1 == 1 or c2 == 1:
                total, soft = 10 + c1 + c2, True  # ace counted as 11
            else:
                total, soft = c1 + c2, False
            for up, pu in CARD_PROBS.items():
                s = (total, soft, up)
                start[s] = start.get(s, 0.0) + p1 * p2 * pu
    return start


START = start_distribution()


def expected_return(V):
    """Expected value of a freshly dealt hand: sum over start states of P(start) * V."""
    return sum(START[s] * V[s] for s in START)


# ----------------------------------------------------------------------------
# One run: iterate until fixed point / cycle / max_iters
# ----------------------------------------------------------------------------
def run_from(P, policy, trace_state, max_iters=50):
    """Iterate evaluate -> improve until a fixed point, a cycle, or max_iters."""
    seen = {_key(policy): 0}
    rows, traj = [], [policy[trace_state]]
    status, fixed = f"no fixed point within {max_iters} iterations", False

    for it in range(1, max_iters + 1):
        V = evaluate_policy(P, policy)
        new, _ = improve_policy(P, V, policy)
        n_changed = sum(new[s] != policy[s] for s in STATES)
        rows.append({"it": it, "changed": n_changed,
                     "hits": sum(a == HIT for a in policy.values()),
                     "ev": expected_return(V)})

        if n_changed == 0:  # policy is already greedy w.r.t. its own V
            status, fixed = f"fixed point after {it - 1} policy update(s)", True
            break

        policy = new
        traj.append(policy[trace_state])
        k = _key(policy)
        if k in seen:
            status = f"CYCLE of length {it - seen[k]} (back to policy #{seen[k]})"
            break
        seen[k] = it

    if not fixed:  # V is stale: it belongs to the previous policy
        V = evaluate_policy(P, policy)
    return {"status": status, "policy": policy, "V": V, "rows": rows, "traj": traj}


# ----------------------------------------------------------------------------
# Absorbing-chain analysis of a fixed policy
# ----------------------------------------------------------------------------
def fundamental_analysis(P, policy):
    """Absorbing-chain analysis of a fixed policy.

    Returns (N, steps, outcomes), all indexed by IDX (row i = STATES[i]):
      N[i, j]  expected visits to game state j starting from i
      steps[i] expected number of actions until the hand ends
      outcomes[o]["prob"][i]   P(hand ends as o), o in OUTCOMES
      outcomes[o]["steps"][i]  E[steps | hand ends as o]  (nan if P(o) = 0)
    """
    n = len(STATES)
    Q = np.zeros((n, n))
    R = {o: np.zeros(n) for o in OUTCOMES}  # P(i -> TERMINAL with outcome o)
    for s in STATES:
        i = IDX[s]
        for p, nxt, reward in P[s][policy[s]]:
            if nxt == TERMINAL:
                R["win" if reward > 0 else "lose" if reward < 0 else "tie"][i] += p
            else:
                Q[i, IDX[nxt]] += p

    N = np.linalg.inv(np.eye(n) - Q)  # fundamental matrix (undiscounted)
    steps = N.sum(axis=1)             # t = N 1

    outcomes = {}
    for o, r in R.items():
        h = N @ r                     # P(absorbed with outcome o)
        g = N @ h                     # E[steps * 1{outcome o}]
        cond = np.divide(g, h, out=np.full(n, np.nan), where=h > 1e-15)
        outcomes[o] = {"prob": h, "steps": cond}
    return N, steps, outcomes


# ----------------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------------
def print_fundamental(policy, P, trace_state):
    N, steps, out = fundamental_analysis(P, policy)
    i = IDX[trace_state]
    print(f"\nFundamental matrix analysis (undiscounted), {_describe(trace_state)}:")
    print(f"  E[steps to finish] = {steps[i]:.4f}")
    for o in OUTCOMES:
        print(f"  {o:<4} P = {out[o]['prob'][i]:.4f}   E[steps | {o}] = {out[o]['steps'][i]:.4f}")

    # averaged over a freshly dealt hand
    w = np.array([START.get(s, 0.0) for s in STATES])
    probs = {o: float(w @ out[o]["prob"]) for o in OUTCOMES}
    print("  Fresh hand: " + ", ".join(f"P({o}) = {p:.4f}" for o, p in probs.items()))
    for o in OUTCOMES:
        if probs[o] > 0:
            cond = float(w @ np.nan_to_num(out[o]["prob"] * out[o]["steps"])) / probs[o]
            print(f"  Fresh hand E[steps | {o}] = {cond:.4f}")

    # uniform average over all game states
    print(f"  All {len(STATES)} states (uniform): E[steps] = {steps.mean():.4f}")
    for o in OUTCOMES:
        p = out[o]["prob"].mean()
        if p > 0:
            c = np.nan_to_num(out[o]["prob"] * out[o]["steps"]).mean() / p
            print(f"    {o:<4} P = {p:.4f}   E[steps | {o}] = {c:.4f}")


def print_hard_table(P, policy, totals=range(16, 22)):
    _, steps, out = fundamental_analysis(P, policy)
    print("\nHard totals 16-21: same statistics per state (undiscounted)")
    print(f"  {'state':<10} {'act':<5} {'E[steps]':>8} "
          + " ".join(f"{'P(' + o + ')':>8}" for o in OUTCOMES) + "  "
          + " ".join(f"{'E[s|' + o + ']':>9}" for o in OUTCOMES))
    for t in totals:
        for u in range(1, 11):
            s = (t, False, u)
            i = IDX[s]
            probs = " ".join(f"{out[o]['prob'][i]:>8.4f}" for o in OUTCOMES)
            conds = " ".join(f"{out[o]['steps'][i]:>9.4f}" for o in OUTCOMES)
            print(f"  hard {t:>2} v{'A' if u == 1 else u:>2} {ACTION_NAME[policy[s]]:<5} "
                  f"{steps[i]:>8.4f} {probs}  {conds}")
        print()

    # average over hard 16-21, weighted by how often each hand is dealt
    w = np.array([START.get(s, 0.0) if (s[0] >= 16 and not s[1]) else 0.0 for s in STATES])
    if w.sum() > 0:
        w = w / w.sum()
        print(f"  Average over hard 16-21 (weighted by deal frequency): "
              f"E[steps] = {w @ steps:.4f}, "
              + ", ".join(f"P({o}) = {w @ out[o]['prob']:.4f}" for o in OUTCOMES))


def print_run(name, res, trace_state):
    """Per-iteration detail (only shown with --verbose)."""
    print(f"\n--- Start: {name} ---")
    print(f"  {'iter':>4} {'changed':>8} {'#HIT':>5} {'E[return]':>11}")
    for r in res["rows"]:
        print(f"  {r['it']:>4} {r['changed']:>8} {r['hits']:>5} {r['ev']:>+11.6f}")
    path = " -> ".join(ACTION_NAME[a] for a in res["traj"])
    print(f"  {_describe(trace_state)}: {path}")


def print_summary(results):
    print(f"\n{'Start':<12} {'Sweeps':>6} {'#HIT':>5} {'E[return]':>11} {'Time(s)':>9}  Result")
    print("-" * 76)
    for name, res in results:
        hits = sum(a == HIT for a in res["policy"].values())
        print(f"{name:<12} {len(res['rows']):>6} {hits:>5} "
              f"{expected_return(res['V']):>+11.6f} {res['time']:>9.4f}  {res['status']}")
    print(f"\nTotal runtime: {sum(r['time'] for _, r in results):.4f}s")


def print_grid(policy):
    ups = range(1, 11)
    header = "         " + " ".join(f"{'A' if u == 1 else u:>2}" for u in ups)

    def block(title, soft, totals):
        print(f"\n{title}   (columns = dealer upcard)\n{header}")
        for t in totals:
            cells = " ".join(f"{'H' if policy[(t, soft, u)] == HIT else '.':>2}" for u in ups)
            print(f"  {'soft' if soft else 'hard'} {t:>2}  {cells}")

    block("HARD TOTALS", False, range(4, 22))
    block("SOFT TOTALS", True, range(12, 22))
    print("\n  H = hit, . = stand")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", nargs=3, metavar=("TOTAL", "SOFT(0/1)", "UPCARD"),
                    default=["16", "0", "10"],
                    help="state whose action is tracked (default: hard 16 vs 10)")
    ap.add_argument("--random", type=int, default=0, help="extra random starting policies")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="show every iteration of every run")
    ap.add_argument("--actions", action="store_true", help="print the full list of actions")
    ap.add_argument("--table", action="store_true", help="per-state table for hard 16-21")
    ap.add_argument("--fundamental", action="store_true",
                    help="print absorbing-chain analysis (win/tie/lose probabilities and steps)")
    args = ap.parse_args()

    trace = (int(args.state[0]), bool(int(args.state[1])), int(args.state[2]))
    assert trace in STATES, f"{trace} is not a valid state"

    P = build_model()
    rng = random.Random(args.seed)
    starts = [("all STAND", start_policy(P, "stand")), ("all HIT", start_policy(P, "hit"))]
    starts += [(f"random #{i + 1}", start_policy(P, "random", rng)) for i in range(args.random)]

    results = []
    for name, pol in starts:
        t0 = time.perf_counter()
        res = run_from(P, pol, trace)
        res["time"] = time.perf_counter() - t0
        if args.verbose:
            print_run(name, res, trace)
        results.append((name, res))

    print_summary(results)

    # Where did the different starts end up?
    ref_name, ref = results[0]
    print(f"\nEnd points compared with '{ref_name}':")
    for name, res in results[1:]:
        diff = [s for s in STATES if res["policy"][s] != ref["policy"][s]]
        if diff:
            shown = ", ".join(_describe(s) for s in diff[:5])
            more = f" (+{len(diff) - 5} more)" if len(diff) > 5 else ""
            print(f"  {name:<12} {len(diff)} state(s) differ: {shown}{more}")
        else:
            print(f"  {name:<12} identical policy")

    best_name, best = max(results, key=lambda kv: expected_return(kv[1]["V"]))
    print(f"\nBest end point: '{best_name}', E[return] = {expected_return(best['V']):+.6f}")
    print(f"Tracked state {_describe(trace)}: {ACTION_NAME[best['policy'][trace]]}")

    print_grid(best["policy"])

    if args.fundamental:
        print_fundamental(best["policy"], P, trace)

    if args.table:
        print_hard_table(P, best["policy"])

    if args.actions:
        print("\nList of actions ((total, soft, upcard) -> action):")
        print([(s, ACTION_NAME[best["policy"][s]]) for s in STATES])


if __name__ == "__main__":
    main()