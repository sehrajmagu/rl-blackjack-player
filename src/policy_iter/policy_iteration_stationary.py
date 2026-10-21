"""Policy iteration for the blackjack MDP defined in src/model.py.

Run with:  python -m src.policy_iter.policy_iteration [--detailed] [--state TOTAL SOFT UPCARD]

Pipeline (mirrors the plan):
  1. Build the transition matrix P_pi for a policy pi (280 states + 1 absorbing
     TERMINAL state).
  2. Decompose it for efficiency:
       a. The upcard never changes, so P_pi is block diagonal: 10 independent
          28-state blocks (+ the shared terminal state).
       b. Inside a block, order states (hard 11-21, soft 12-21, hard 4-10) so
          every transition goes to an *earlier* state. Then Q = transient part of P_pi is
          strictly lower triangular, and (I - Q) V = r is solved by plain
          forward substitution. No matrix inverse.
  3. Solve xP = x for the stationary distribution(s).
  4. Policy improvement, state by state: compare expected cost of STAND vs HIT,
     keep the cheaper one, and print the trace for a chosen state.
  5. Print the converged ("stationary") policy and the list of actions.
"""

import argparse
import time

import numpy as np

from ..model import GAMMA, HIT, STAND, STATES, TERMINAL, _describe, build_model

ACTION_NAME = {STAND: "STAND", HIT: "HIT"}
TOL = 1e-12


# ----------------------------------------------------------------------------
# 1. Transition matrix for a fixed policy
# ----------------------------------------------------------------------------
def build_policy_matrix(P, policy):
    """Full (n+1)x(n+1) stochastic matrix P_pi. Last row/col = TERMINAL (absorbing).

    Also returns the expected immediate reward vector r_pi and the index map.
    """
    idx = {s: i for i, s in enumerate(STATES)}
    n = len(STATES)
    T = np.zeros((n + 1, n + 1))
    r = np.zeros(n + 1)
    for s in STATES:
        for p, nxt, reward in P[s][policy[s]]:
            j = n if nxt == TERMINAL else idx[nxt]
            T[idx[s], j] += p
            r[idx[s]] += p * reward
    T[n, n] = 1.0  # terminal is absorbing
    return T, r, idx


# ----------------------------------------------------------------------------
# 2. Decomposition: per-upcard blocks, topologically ordered
# ----------------------------------------------------------------------------
def block_order(upcard):
    """States of one upcard block, ordered so transitions point to earlier states.

    Three groups:
      1. hard 11..21 (descending): can only reach higher hard totals.
      2. soft 12..21 (descending): reach higher soft totals, or hard 12..21
         (group 1) when the ace is forced down to 1.
      3. hard 4..10 (descending): reach higher hard totals, or soft states
         (group 2) by drawing an ace (e.g. hard 10 + A = soft 21).
    """
    def pick(soft, lo, hi):
        return sorted((s for s in STATES if s[2] == upcard and s[1] == soft and lo <= s[0] <= hi),
                      key=lambda s: -s[0])
    return pick(False, 11, 21) + pick(True, 12, 21) + pick(False, 4, 10)


def evaluate_policy(P, policy, gamma=GAMMA):
    """Exact V_pi by forward substitution, one upcard block at a time."""
    V = {}
    for upcard in range(1, 11):
        order = block_order(upcard)
        pos = {s: i for i, s in enumerate(order)}
        for s in order:  # earlier states are already solved
            v = 0.0
            for p, nxt, reward in P[s][policy[s]]:
                if nxt == TERMINAL:
                    v += p * reward
                else:
                    assert pos[nxt] < pos[s], f"ordering broken: {s} -> {nxt}"
                    v += p * (reward + gamma * V[nxt])
            V[s] = v
    return V


def evaluate_policy_dense(P, policy):
    """Slow reference: solve (I - Q) V = r with a dense solver (used to verify)."""
    T, r, _ = build_policy_matrix(P, policy)
    n = len(STATES)
    Q = T[:n, :n]
    return np.linalg.solve(np.eye(n) - Q, r[:n])


# ----------------------------------------------------------------------------
# 3. Stationary distributions: xP = x
# ----------------------------------------------------------------------------
def stationary_distribution(T):
    """Solve xP = x, sum(x) = 1. Also reports how many independent solutions exist.

    The number of eigenvalues equal to 1 is the number of recurrent classes.
    For blackjack that is exactly one (TERMINAL), so x = e_terminal is the ONLY
    stationary distribution for every policy. The policy is never visible in x,
    which is why V_pi (below) is what we compare, not x itself.
    """
    n = T.shape[0]
    A = np.vstack([T.T - np.eye(n), np.ones((1, n))])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    n_unit = int(np.sum(np.isclose(np.linalg.eigvals(T), 1.0, atol=1e-9)))
    return x, n_unit


def expected_visits(P, policy, state):
    """Expected number of times each state is visited starting from `state`
    (a row of the fundamental matrix N = (I - Q)^-1). This is the informative
    'occupancy' counterpart of the stationary distribution for an absorbing chain.
    """
    T, _, idx = build_policy_matrix(P, policy)
    n = len(STATES)
    e = np.zeros(n)
    e[idx[state]] = 1.0
    visits = np.linalg.solve((np.eye(n) - T[:n, :n]).T, e)  # e (I-Q)^-1
    return {s: visits[idx[s]] for s in STATES if visits[idx[s]] > 1e-12}


# ----------------------------------------------------------------------------
# 4. Policy improvement (cost = -value, so min cost == max value)
# ----------------------------------------------------------------------------
def q_value(P, V, state, action, gamma=GAMMA):
    return sum(
        p * (reward + (0.0 if nxt == TERMINAL else gamma * V[nxt]))
        for p, nxt, reward in P[state][action]
    )


def improve_policy(P, V, policy, trace_state=None):
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
        if s == trace_state:
            print(f"    state {_describe(s)}")
            for a, c in sorted(costs.items()):
                mark = "  <-- min cost" if a == best else ""
                print(f"      {ACTION_NAME[a]:5s} expected cost = {c:+.6f}{mark}")
            print(f"      current -> improved: {ACTION_NAME[policy[s]]} -> {ACTION_NAME[best]}")
    return new_policy, changed


def solve(P, theta=1e-9):
    """Quiet policy iteration with the same interface as value_iteration.solve:
    returns (V, policy, stats). theta is unused (policy iteration is exact) and is
    only kept so the two solvers are interchangeable.

    One iteration = one exact evaluation pass + one improvement pass, so
    updates counts 2 passes over the states per iteration.
    """
    start = time.perf_counter()
    policy = {s: STAND for s in STATES}
    deltas, V_old = [], None
    while True:
        V = evaluate_policy(P, policy)
        if V_old is not None:
            deltas.append(max(abs(V[s] - V_old[s]) for s in STATES))
        V_old = V
        policy, changed = improve_policy(P, V, policy)
        if not changed:
            break
    V[TERMINAL] = 0.0  # same convention as value_iteration
    iters = len(deltas) + 1
    stats = {
        "deltas": deltas,
        "sweeps": iters,
        "time": time.perf_counter() - start,
        "updates": iters * 2 * len(STATES),
    }
    return V, policy, stats


def policy_iteration(P, trace_state=None, max_iters=100):
    policy = {s: STAND for s in STATES}  # start: always stand
    V = None
    for it in range(1, max_iters + 1):
        V = evaluate_policy(P, policy)
        print(f"\nIteration {it}: policy evaluated "
              f"(mean V over states = {np.mean(list(V.values())):+.5f})")
        policy, changed = improve_policy(P, V, policy, trace_state)
        n_hit = sum(1 for a in policy.values() if a == HIT)
        print(f"  improved policy: {n_hit} HIT / {len(STATES) - n_hit} STAND states")
        if not changed:
            print("  policy stable -> converged")
            return policy, V, it
    raise RuntimeError("policy iteration did not converge")


# ----------------------------------------------------------------------------
# 5. Printing
# ----------------------------------------------------------------------------
def print_table(V, policy):  # same layout as value_iteration.print_table
    upcards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]  # ace last
    for soft in (False, True):
        totals = range(21, 11, -1) if soft else range(21, 3, -1)
        print("SOFT" if soft else "HARD", "".join(f"{'A' if up == 1 else up:>8}" for up in upcards))
        for total in totals:
            row = f"{total:>4} "
            for up in upcards:
                state = (total, soft, up)
                row += f"{V[state]:>+7.3f}" + ("h" if policy[state] == HIT else " ")
            print(row)
        print()


def print_policy_tables(policy):
    ups = list(range(1, 11))
    header = "        " + " ".join(f"{'A' if u == 1 else u:>2}" for u in ups)
    print("\nStationary (converged) policy   H = hit, S = stand")
    print("\nHard totals\n" + header)
    for t in range(4, 22):
        row = " ".join(f"{'H' if policy[(t, False, u)] == HIT else 'S':>2}" for u in ups)
        print(f"  hard {t:2d} {row}")
    print("\nSoft totals\n" + header)
    for t in range(12, 22):
        row = " ".join(f"{'H' if policy[(t, True, u)] == HIT else 'S':>2}" for u in ups)
        print(f"  soft {t:2d} {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", nargs=3, metavar=("TOTAL", "SOFT(0/1)", "UPCARD"),
                    default=["16", "0", "10"], help="state to trace (default: hard 16 vs 10)")
    ap.add_argument("--detailed", action="store_true", help="full trace, xP = x, action list")
    args = ap.parse_args()
    if not args.detailed:
        V, policy, stats = solve(build_model())
        print(stats["sweeps"], "sweeps")
        print(stats["updates"], "updates")
        print(round(stats["time"], 4), "seconds")
        print()
        print_table(V, policy)
        return
    trace = (int(args.state[0]), bool(int(args.state[1])), int(args.state[2]))
    assert trace in STATES, f"{trace} is not a valid state"

    P = build_model()
    print(f"Tracing state: {_describe(trace)}")

    policy, V, iters = policy_iteration(P, trace_state=trace)
    print(f"\nConverged in {iters} iterations.")

    # Verify the triangular solve against a dense solve and look at xP = x.
    dense = evaluate_policy_dense(P, policy)
    err = max(abs(dense[i] - V[s]) for i, s in enumerate(STATES))
    print(f"Max |forward-substitution V - dense V| = {err:.2e}")

    T, _, _ = build_policy_matrix(P, policy)
    x, n_unit = stationary_distribution(T)
    print(f"\nxP = x: {n_unit} eigenvalue(s) equal to 1 -> unique stationary distribution")
    print(f"  mass on TERMINAL = {x[-1]:.6f}, mass on all {len(STATES)} game states = "
          f"{x[:-1].sum():.2e}")
    print("  (every hand ends, so the policy cannot show up in x. "
          "See expected visits and V below.)")

    visits = expected_visits(P, policy, trace)
    print(f"\nExpected visits starting from {_describe(trace)} under the final policy:")
    for s, v in sorted(visits.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {_describe(s):18s} {v:.4f}")

    print(f"\nV*({_describe(trace)}) = {V[trace]:+.5f}  "
          f"(expected cost {-V[trace]:+.5f}) -> action {ACTION_NAME[policy[trace]]}")

    print_policy_tables(policy)

    print("\nList of stationary actions ((total, soft, upcard) -> action):")
    actions = [(s, ACTION_NAME[policy[s]]) for s in STATES]
    print(actions)


if __name__ == "__main__":
    main()
