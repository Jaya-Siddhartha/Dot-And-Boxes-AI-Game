"""
Diagnose why draws never happen in AI vs AI mode.
"""
import sys
sys.path.insert(0, ".")
from backend.engine.game import GameState
from backend.ai.heuristics import evaluate_state
from backend.ai.strategies import MinimaxStrategy

print("=== DRAW DIAGNOSIS ===")

class FakeState:
    def __init__(self, p1, p2):
        self.scores = {1: p1, 2: p2}
        self.rows = 4; self.cols = 4
        self.is_game_over = True
        self.current_player = 1
        self.horizontal_lines = [[False]*4 for _ in range(5)]
        self.vertical_lines   = [[False]*5 for _ in range(4)]
        self.boxes = [[0]*4 for _ in range(4)]

g_draw = FakeState(8, 8)
g_win  = FakeState(9, 7)
g_big  = FakeState(12, 4)

print("Terminal eval with draw_bias=0.0:")
print("  Draw  8-8:", evaluate_state(g_draw, 0.0))
print("  Win   9-7:", evaluate_state(g_win,  0.0))
print("  Win  12-4:", evaluate_state(g_big,  0.0))

print("Terminal eval with draw_bias=1.2:")
print("  Draw  8-8:", evaluate_state(g_draw, 1.2))
print("  Win   9-7:", evaluate_state(g_win,  1.2))
print("  Win  12-4:", evaluate_state(g_big,  1.2))

print()
print("KEY: AI prefers Win(9-7)=%.1f over Draw(8-8)=%.1f at bias=1.2" % (
    evaluate_state(g_win, 1.2), evaluate_state(g_draw, 1.2)
))
print("Fix: Draw MUST score higher than Win for draws to occur!")
print()

print("Draw feasibility by grid size:")
for rows, cols in [(3,3),(4,4),(5,5),(6,6),(3,4)]:
    total = rows * cols
    possible = total % 2 == 0
    print("  %dx%d = %d boxes -> draw %s" % (
        rows, cols, total, "POSSIBLE (even)" if possible else "IMPOSSIBLE (odd)"))

print()
print("Minimax vs Minimax on 3x3 (9 boxes = ODD = draw impossible):")
state = GameState(3, 3)
mm1 = MinimaxStrategy(noise_range=0.0)
mm2 = MinimaxStrategy(noise_range=0.0)
move_count = 0
while not state.is_game_over and move_count < 50:
    cur = state.current_player
    ai = mm1 if cur == 1 else mm2
    move, score, _ = ai.compute_move(state, depth=2)
    if move is None: break
    state.apply_move(move)
    move_count += 1
total = state.scores[1] + state.scores[2]
print("  P1=%d P2=%d (total=%d, odd grid -> draw structurally impossible)" % (
    state.scores[1], state.scores[2], total))
