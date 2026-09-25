"""Game rules and card distribution shared by the whole project."""

from dataclasses import dataclass

# Infinite deck: every draw is independent. Aces are 1, face cards count as 10.
CARD_PROBS = {card: 1 / 13 for card in range(1, 10)}
CARD_PROBS[10] = 4 / 13


@dataclass(frozen=True)
class Rules:
    hit_soft_17: bool = True       # H17: dealer hits a soft 17
    dealer_peeks: bool = True      # dealer checks for blackjack when showing A or 10
    natural_payout: float = 1.5    # two-card 21; every other win pays 1


DEFAULT_RULES = Rules()
