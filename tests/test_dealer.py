import random

import pytest

from src.dealer import BLACKJACK, BUST, add_card, dealer_outcome_dist
from src.rules import CARD_PROBS, Rules

UPCARDS = range(1, 11)
ALL_RULES = [
    Rules(hit_soft_17=h17, dealer_peeks=peek)
    for h17 in (True, False)
    for peek in (True, False)
]


@pytest.mark.parametrize("rules", ALL_RULES)
@pytest.mark.parametrize("upcard", UPCARDS)
def test_distribution_sums_to_one(upcard, rules):
    assert sum(dealer_outcome_dist(upcard, rules).values()) == pytest.approx(1.0)


def test_add_card_aces():
    assert add_card(0, False, 1) == (11, True)       # lone ace is soft 11
    assert add_card(11, True, 1) == (12, True)       # A+A = soft 12
    assert add_card(16, True, 10) == (16, False)     # soft 16 + 10 -> hard 16
    assert add_card(20, False, 1) == (21, False)     # ace forced to count as 1


def test_blackjack_only_reported_without_peek():
    assert BLACKJACK not in dealer_outcome_dist(1, Rules(dealer_peeks=True))
    no_peek = dealer_outcome_dist(1, Rules(dealer_peeks=False))
    assert no_peek[BLACKJACK] == pytest.approx(CARD_PROBS[10])
    assert dealer_outcome_dist(5, Rules(dealer_peeks=False))[BLACKJACK] == 0.0


def test_h17_busts_more_than_s17_with_six_up():
    h17 = dealer_outcome_dist(6, Rules(hit_soft_17=True))[BUST]
    s17 = dealer_outcome_dist(6, Rules(hit_soft_17=False))[BUST]
    assert h17 > s17


# --- Independent Monte Carlo check (does not reuse add_card) ---------------

DECK = list(range(1, 10)) + [10] * 4


def _hand_value(cards):
    total = sum(cards)
    if 1 in cards and total + 10 <= 21:
        return total + 10, True
    return total, False


def _simulate(upcard, rules, rng):
    cards = [upcard, rng.choice(DECK)]
    if set(cards) == {1, 10}:
        return None if rules.dealer_peeks else BLACKJACK
    while True:
        total, soft = _hand_value(cards)
        if total > 21:
            return BUST
        if total > 17 or (total == 17 and not (soft and rules.hit_soft_17)):
            return total
        cards.append(rng.choice(DECK))


@pytest.mark.parametrize("rules", [Rules(hit_soft_17=True), Rules(hit_soft_17=False, dealer_peeks=False)])
@pytest.mark.parametrize("upcard", UPCARDS)
def test_matches_monte_carlo(upcard, rules):
    rng = random.Random(upcard)
    counts, n = {}, 0
    while n < 100_000:
        outcome = _simulate(upcard, rules, rng)
        if outcome is None:  # dealer natural under peek: player never acts
            continue
        counts[outcome] = counts.get(outcome, 0) + 1
        n += 1

    for outcome, p in dealer_outcome_dist(upcard, rules).items():
        assert counts.get(outcome, 0) / n == pytest.approx(p, abs=0.006)
