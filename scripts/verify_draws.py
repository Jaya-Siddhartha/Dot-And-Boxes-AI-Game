"""Verify the new draw scoring is correct and draw > win."""
import sys
sys.path.insert(0, ".")
from backend.ai.heuristics import evaluate_state
from backend.engine.game import GameState
from backend.ai.strategies import MinimaxStrategy, AlphaBetaStrategy
import time

class FakeState:
    def __init__(self, p1, p2, rows=4, cols=4, over=True):
        self.scores = {1: p1, 2: p2}
        self.rows = rows; self.cols = cols
        self.is_game_over = over
        self.current_player = 1
        self.horizontal_lines = [[False]*cols for _ in range(rows+1)]
        self.vertical_lines   = [[False]*(cols+1) for _ in range(rows)]
        self.boxes = [[0]*cols for _ in range(rows)]

g_draw  = FakeState(8, 8)       # 4x4, 8-8 draw
g_win97 = FakeState(9, 7)       # 4x4, 9-7 win
g_win12 = FakeState(12, 4)      # 4x4, 12-4 landslide
g_draw3 = FakeState(4, 5, rows=3, cols=3)  # 3x3 odd, draw impossible

print("=== Draw scoring verification ===\n")

for bias in [0.0, 1.2, 2.0, 2.5]:
    d = evaluate_state(g_draw, bias)
    w = evaluate_state(g_win97, bias)
    b = evaluate_state(g_win12, bias)
    DRAW_WINS = "DRAW > WIN" if d > w else "WIN > DRAW (BAD)"
    print(f"draw_bias={bias}:  Draw(8-8)={d:.1f}  Win(9-7)={w:.1f}  Win(12-4)={b:.1f}  --> {DRAW_WINS}")

print()
print("On odd 3x3 grid (draw impossible, bias should be ignored):")
d3 = evaluate_state(g_draw3, 2.0)
print(f"  FakeState(4,5) on 3x3 is_game_over: eval={d3:.1f} (no draw reward since odd)")

print()
print("=== Simulated draws: Minimax vs Minimax on 4x4 with draw_bias=2.0 ===")
draws = 0; wins = 0; total = 10
for trial in range(total):
    state = GameState(4, 4)
    mm1 = MinimaxStrategy(noise_range=0.3)
    mm2 = MinimaxStrategy(noise_range=0.3)
    # Patch both to use draw_bias=2.0 in their evaluation
    import functools
    from backend.ai import heuristics as H
    orig = H.evaluate_state
    mm1._noisy_eval = functools.partial(lambda s, db, orig=orig, nr=0.3: orig(s, db), db=2.0)
    mm2._noisy_eval = functools.partial(lambda s, db, orig=orig, nr=0.3: orig(s, db), db=2.0)
    # Use alphabeta with drawn bias instead
    from backend.ai.strategies import AlphaBetaStrategy
    ab1 = AlphaBetaStrategy(draw_bias=2.0, noise_range=0.3)
    ab2 = AlphaBetaStrategy(draw_bias=2.0, noise_range=0.3)
    state2 = GameState(4, 4)
    move_count = 0
    while not state2.is_game_over and move_count < 100:
        cur = state2.current_player
        ai = ab1 if cur == 1 else ab2
        move, _, _ = ai.compute_move(state2, depth=3)
        if move is None: break
        state2.apply_move(move)
        move_count += 1
    p1, p2 = state2.scores[1], state2.scores[2]
    if p1 == p2:
        draws += 1
        print(f"  Trial {trial+1}: DRAW {p1}-{p2}")
    else:
        wins += 1
        print(f"  Trial {trial+1}: WIN/LOSS {p1}-{p2} (winner=P{'1' if p1>p2 else '2'})")

print(f"\nResults: {draws} draws / {total} total = {100*draws//total}% draw rate")
print(f"Target: at least 20-30% draws")
