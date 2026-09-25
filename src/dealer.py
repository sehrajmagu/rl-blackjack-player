"""The house: exact distribution of the dealer's final hand for each upcard.

The dealer follows a fixed policy (draw until 17+, optionally hitting soft 17),
so this is pure probability -- no learning involved. Results feed the reward
for the player's `stick` action.
"""

from functools import lru_cache

from .rules import CARD_PROBS, DEFAULT_RULES, Rules

BUST = "bust"
BLACKJACK = "blackjack"  # only reported when the dealer does not peek
FINAL_TOTALS = (17, 18, 19, 20, 21)


def add_card(total, soft, card):
    """Add a card to a hand given as (total, soft).

    `soft` means an ace is currently counted as 11. If a soft hand would bust,
    that ace drops back to 1 and the hand becomes hard.
    """
    if card == 1 and total + 11 <= 21:
        total, soft = total + 11, True
    else:
        total += card
    if total > 21 and soft:
        total, soft = total - 10, False
    return total, soft


def dealer_hits(total, soft, rules=DEFAULT_RULES):
    return total < 17 or (total == 17 and soft and rules.hit_soft_17)


@lru_cache(maxsize=None)
def _final_dist(total, soft, hit_soft_17):
    """Distribution over final outcomes starting from a dealer hand (total, soft)."""
    if total > 21:
        return {BUST: 1.0}
    if not dealer_hits(total, soft, Rules(hit_soft_17=hit_soft_17)):
        return {total: 1.0}

    dist = {}
    for card, p in CARD_PROBS.items():
        for outcome, q in _final_dist(*add_card(total, soft, card), hit_soft_17).items():
            dist[outcome] = dist.get(outcome, 0.0) + p * q
    return dist


def natural_prob(upcard):
    """Probability that the hole card gives the dealer blackjack."""
    if upcard == 1:
        return CARD_PROBS[10]
    if upcard == 10:
        return CARD_PROBS[1]
    return 0.0


def _makes_natural(upcard, hole):
    return {upcard, hole} == {1, 10}


def dealer_outcome_dist(upcard, rules=DEFAULT_RULES):
    """Distribution of the dealer's final result given the upcard (1 = ace).

    Keys are 17..21 and BUST. With `dealer_peeks`, the player only gets to act
    when the dealer has no blackjack, so the distribution is conditioned on
    that. Without peeking, dealer blackjacks are reported under BLACKJACK,
    since they beat a player's multi-card 21.
    """
    if upcard not in CARD_PROBS:
        raise ValueError(f"upcard must be 1..10, got {upcard}")

    start = add_card(0, False, upcard)
    dist = {outcome: 0.0 for outcome in FINAL_TOTALS + (BUST,)}
    if not rules.dealer_peeks:
        dist[BLACKJACK] = 0.0

    # The hole card is the first draw; handle it explicitly to separate naturals.
    for hole, p in CARD_PROBS.items():
        if _makes_natural(upcard, hole):
            if not rules.dealer_peeks:
                dist[BLACKJACK] += p
            continue
        hand = add_card(*start, hole)
        for outcome, q in _final_dist(*hand, rules.hit_soft_17).items():
            dist[outcome] += p * q

    if rules.dealer_peeks:
        no_natural = 1.0 - natural_prob(upcard)
        dist = {outcome: q / no_natural for outcome, q in dist.items()}
    return dist


if __name__ == "__main__":
    header = ["up"] + [str(t) for t in FINAL_TOTALS] + [BUST]
    print("  ".join(f"{h:>6}" for h in header))
    for up in range(1, 11):
        dist = dealer_outcome_dist(up)
        row = ["A" if up == 1 else str(up)]
        row += [f"{dist[o]:.3f}" for o in FINAL_TOTALS + (BUST,)]
        print("  ".join(f"{c:>6}" for c in row))
