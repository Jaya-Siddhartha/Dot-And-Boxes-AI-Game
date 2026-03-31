"""
Test draws: run multiple AI vs AI games and verify outcome variety.

This test runs 20 games of AlphaBeta vs AlphaBeta on a 4x4 grid
at depth 3, counting wins, losses, and draws to verify natural
variability in outcomes.
"""
import sys
sys.path.insert(0, ".")

from backend.engine.game import GameState
from backend.ai.strategies import AlphaBetaStrategy, MinimaxStrategy, AdaptiveLearningStrategy
from backend.learning.qlearning import QLearner

NUM_GAMES = 20
DEPTH = 3
GRID = 4

print(f"=== OUTCOME VARIETY TEST — {NUM_GAMES} games AB vs AB on {GRID}x{GRID} depth={DEPTH} ===\n")

results = {"p1_wins": 0, "p2_wins": 0, "draws": 0}
ql = QLearner(data_file="data/learning_data.json")

for game_num in range(NUM_GAMES):
    state = GameState(GRID, GRID)
    # Fresh strategies each game (clean TT, fresh noise)
    strats = {
        1: AlphaBetaStrategy(time_limit=3.0),
        2: AlphaBetaStrategy(time_limit=3.0),
    }

    move_count = 0
    while not state.is_game_over:
        cur = state.current_player
        ai = strats[cur]
        move, _, _ = ai.compute_move(state.clone(), DEPTH)
        if move is None:
            break
        state.apply_move(move)
        move_count += 1

    p1, p2 = state.scores[1], state.scores[2]
    if p1 > p2:
        results["p1_wins"] += 1
        outcome = "P1 wins"
    elif p2 > p1:
        results["p2_wins"] += 1
        outcome = "P2 wins"
    else:
        results["draws"] += 1
        outcome = "DRAW"

    print(f"  Game {game_num+1:2d}: {outcome:8s} | {p1}-{p2} | {move_count} moves")

print()
print("=== RESULTS ===")
print(f"  P1 Wins:  {results['p1_wins']} ({100*results['p1_wins']/NUM_GAMES:.0f}%)")
print(f"  P2 Wins:  {results['p2_wins']} ({100*results['p2_wins']/NUM_GAMES:.0f}%)")
print(f"  Draws:    {results['draws']} ({100*results['draws']/NUM_GAMES:.0f}%)")
print()

# Verify at least some variety
unique_outcomes = sum(1 for v in results.values() if v > 0)
if unique_outcomes >= 2:
    print(f"[PASS] At least {unique_outcomes} different outcome types observed")
else:
    print("[WARN] Only one outcome type — AI may be too deterministic")

# Now test Minimax vs AlphaBeta
print(f"\n=== MM vs AB — 10 games on {GRID}x{GRID} depth=2 ===\n")
results2 = {"p1_wins": 0, "p2_wins": 0, "draws": 0}

for game_num in range(10):
    state = GameState(GRID, GRID)
    strats = {
        1: MinimaxStrategy(),
        2: AlphaBetaStrategy(time_limit=3.0),
    }

    while not state.is_game_over:
        cur = state.current_player
        move, _, _ = strats[cur].compute_move(state.clone(), 2)
        if move is None:
            break
        state.apply_move(move)

    p1, p2 = state.scores[1], state.scores[2]
    if p1 > p2:
        results2["p1_wins"] += 1
        outcome = "MM wins"
    elif p2 > p1:
        results2["p2_wins"] += 1
        outcome = "AB wins"
    else:
        results2["draws"] += 1
        outcome = "DRAW"
    print(f"  Game {game_num+1:2d}: {outcome:8s} | {p1}-{p2}")

print()
print("=== MM vs AB RESULTS ===")
print(f"  Minimax Wins: {results2['p1_wins']}")
print(f"  AB Wins:      {results2['p2_wins']}")
print(f"  Draws:        {results2['draws']}")
print()
print("[DRAW TESTS COMPLETE]")
