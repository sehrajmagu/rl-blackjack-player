"""policy_iteration_2: policy iteration that does NOT assume a stable policy exists.

Run with:  python -m src.policy_iter.policy_iteration_2 [--state TOTAL SOFT UPCARD] [--random N]

What it does
  1. Starts from simple policies: HIT in every state (STAND where hitting is not
     allowed, i.e. 21) and STAND in every state. Optionally N random mixes.
  2. Repeats evaluate -> improve, recording the whole trajectory of policies.
     After every update it checks whether the policy
        - stopped changing      (fixed point), or
        - returned to a policy it already visited (cycle).
     Neither is assumed: the loop reports whichever actually happens.
  3. Compares where the different starts end up and prints the best end point.

Note on cycles: with "change only if strictly cheaper" (as in improve_policy) the
expected return can never go down between iterations, so a cycle cannot occur
for this MDP. The detector is there so that claim is checked, not assumed.
"""

import argparse
import random

from ..model import HIT, STAND, STATES, _describe, build_model
from ..rules import CARD_PROBS
from .policy_iteration import ACTION_NAME, evaluate_policy, improve_policy


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
    seen = {_key(policy): 0}
    rows = []  # one row per iteration
    traj = [policy[trace_state]]
    for it in range(1, max_iters + 1):
        V = evaluate_policy(P, policy)
        new, changed = improve_policy(P, V, policy)
        n_changed = sum(new[s] != policy[s] for s in STATES)
        rows.append({
            "it": it,
            "changed": n_changed,
            "hits": sum(a == HIT for a in policy.values()),
            "ev": expected_return(V),
        })
        if not changed:  # policy used for V is already greedy w.r.t. V
            return {"status": f"fixed point after {it - 1} policy update(s)", "policy": policy,
                    "V": V, "rows": rows, "traj": traj}
        policy = new
        traj.append(policy[trace_state])
        k = _key(policy)
        if k in seen:
            return {"status": f"CYCLE of length {it - seen[k]} (back to policy #{seen[k]})",
                    "policy": policy, "V": evaluate_policy(P, policy), "rows": rows, "traj": traj}
        seen[k] = it
    return {"status": f"no fixed point within {max_iters} iterations", "policy": policy,
            "V": evaluate_policy(P, policy), "rows": rows, "traj": traj}


# ----------------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------------
def print_run(name, res, trace_state):
    print(f"\n=== Start: {name} ===")
    print("  it  changed  #HIT  E[return of a dealt hand]")
    for r in res["rows"]:
        print(f"  {r['it']:2d}  {r['changed']:7d}  {r['hits']:4d}  {r['ev']:+.6f}")
    path = " -> ".join(ACTION_NAME[a] for a in res["traj"])
    print(f"  {_describe(trace_state)}: {path}")
    print(f"  result: {res['status']}")


def print_grid(policy):
    ups = list(range(1, 11))
    header = "        " + " ".join(f"{'A' if u == 1 else u:>2}" for u in ups)

    def cell(s):
        return "H" if policy[s] == HIT else "S"

    print("\nHard totals\n" + header)
    for t in range(4, 22):
        print(f"  hard {t:2d} " + " ".join(f"{cell((t, False, u)):>2}" for u in ups))
    print("\nSoft totals\n" + header)
    for t in range(12, 22):
        print(f"  soft {t:2d} " + " ".join(f"{cell((t, True, u)):>2}" for u in ups))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", nargs=3, metavar=("TOTAL", "SOFT(0/1)", "UPCARD"),
                    default=["16", "0", "10"], help="state whose action is tracked (default: hard 16 vs 10)")
    ap.add_argument("--random", type=int, default=0, help="extra random starting policies")
    ap.add_argument("--seed", type=int, default=0)
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
        print_run(name, res, trace)
        results.append((name, res))

    # Where did the different starts end up?
    ref_name, ref = results[0]
    print("\n=== Comparison of end points ===")
    for name, res in results[1:]:
        diff = [s for s in STATES if res["policy"][s] != ref["policy"][s]]
        print(f"  {name} vs {ref_name}: {len(diff)} state(s) differ", end="")
        if diff:
            print(" -> " + ", ".join(_describe(s) for s in diff[:10]))
        else:
            print(" (identical policy)")

    best_name, best = max(results, key=lambda kv: expected_return(kv[1]["V"]))
    policy = best["policy"]
    print(f"\nBest end point: start '{best_name}', E[return] = {expected_return(best['V']):+.6f}")

    print("\nFinal policy   H = hit, S = stand")
    print_grid(policy)

    print("\nList of actions ((total, soft, upcard) -> action):")
    print([(s, ACTION_NAME[policy[s]]) for s in STATES])


if __name__ == "__main__":
    main()
