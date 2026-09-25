# BLACKJACK PLAYER - REINFORCEMENT LEARNING PROJECT 
## ISHAAN VARIOR AND SEHRAJ MAGU

A Blackjack player built with Dynamic Programming: we model the game as an MDP and solve it with Policy Iteration and Value Iteration.

## Rules
- Infinite deck: each card is drawn independently. A-9 each have probability 1/13, and 10-valued cards (10, J, Q, K) have 4/13
- Player actions: **hit** or **stand**
- Payouts: win +1, loss -1, push 0
- A natural (two-card 21: an ace plus a 10-valued card) pays 1.5x. It is a push if the dealer also has a natural. Any other 21 pays 1x

All rules are configurable in `src/rules.py`.

## Modelling the House
The House operates on a hard and fast rule as it does in standard casinos. 
- Dealer hits on Soft 17 (an ace counted as 11) and below
- Stands on Hard 17 and above, and on Soft 18 and above
- Dealer peeks at face down card to check for blackjack if the face up card is an ace or a 10

Since the dealer's policy is fixed, its behaviour is pure probability. `dealer_outcome_dist(upcard)` in `src/dealer.py` gives the exact probability of the dealer finishing on 17, 18, 19, 20, 21 or busting. Because of the peek, if the player gets to act, the dealer does not have blackjack, so the distribution for an ace or 10 upcard excludes it.

Dealer bust probability by upcard:

| Upcard | A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Bust % | 20.1 | 35.7 | 37.7 | 39.7 | 41.8 | 43.9 | 26.2 | 24.5 | 22.8 | 23.0 |

## Setup
Requires Python 3.11. Create a virtual environment once, after cloning:
```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Run `source .venv/bin/activate` again in each new terminal. The `.venv/` folder is gitignored, so each of us makes our own.

## Running
With the venv active:
```bash
python -m src.dealer   # print the dealer outcome table
pytest                 # run tests
```
