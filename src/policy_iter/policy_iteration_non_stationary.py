"""Policy iteration that does NOT assume a stable policy exists.

Run with:  python -m src.policy_iter.policy_iteration_non_stationary [--state TOTAL SOFT UPCARD] [--random N] [--verbose] [--actions]

What it does
  1. Starts from simple policies: HIT in every state (STAND where hitting is not
     allowed, i.e. 21) and STAND in every state. Optionally N random mixes.
  2. Repeats evaluate -> improve, recording the whole trajectory of policies.
     After every update it checks whether the policy
        - stopped changing      (fixed point), or
        - returned to a policy it already visited (cycle).
     Neither is assumed: the loop reports whichever actually happens.
  3. Compares where the different starts end up and prints the best end point.

Self-contained: only depends on the MDP definition (model.py, rules.py), not on
any other policy_iter file.

Note on cycles: with "change only if strictly cheaper" (as in improve_policy) the
expected return can never go down between iterations, so a cycle cannot occur
for this MDP. The detector is there so that claim is checked, not assumed.
"""

import argparse
import random

import numpy as np

from ..model import GAMMA, HIT, STAND, STATES, TERMINAL, _describe, build_model
from ..rules import CARD_PROBS

ACTION_NAME = {STAND: "STAND", HIT: "HIT"}
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
# Printing
# ----------------------------------------------------------------------------
def print_run(name, res, trace_state):
    """Per-iteration detail (only shown with --verbose)."""
    print(f"\n--- Start: {name} ---")
    print(f"  {'iter':>4} {'changed':>8} {'#HIT':>5} {'E[return]':>11}")
    for r in res["rows"]:
        print(f"  {r['it']:>4} {r['changed']:>8} {r['hits']:>5} {r['ev']:>+11.6f}")
    path = " -> ".join(ACTION_NAME[a] for a in res["traj"])
    print(f"  {_describe(trace_state)}: {path}")


def print_summary(results):
    print(f"\n{'Start':<12} {'Iters':>5} {'#HIT':>5} {'E[return]':>11}  Result")
    print("-" * 64)
    for name, res in results:
        hits = sum(a == HIT for a in res["policy"].values())
        print(f"{name:<12} {len(res['rows']):>5} {hits:>5} "
              f"{expected_return(res['V']):>+11.6f}  {res['status']}")


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
    args = ap.parse_args()

    trace = (int(args.state[0]), bool(int(args.state[1])), int(args.state[2]))
    assert trace in STATES, f"{trace} is not a valid state"

    P = build_model()
    rng = random.Random(args.seed)
    starts = [("all STAND", start_policy(P, "stand")), ("all HIT", start_policy(P, "hit"))]
    starts += [(f"random #{i + 1}", start_policy(P, "random", rng)) for i in range(args.random)]

    results = []
    for name, pol in starts:
        res = run_from(P, pol, trace)
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

    if args.actions:
        print("\nList of actions ((total, soft, upcard) -> action):")
        print([(s, ACTION_NAME[best["policy"][s]]) for s in STATES])


if __name__ == "__main__":
    main()