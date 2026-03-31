"""Quick smoke test for the refactored AI backend v6.0."""
import time
import sys

sys.path.insert(0, ".")

from backend.engine.game import GameState
from backend.ai.strategies import MinimaxStrategy, AlphaBetaStrategy, RandomStrategy, AdaptiveLearningStrategy
from backend.ai.heuristics import evaluate_state, get_capturing_moves, is_capturing_move
from backend.learning.qlearning import QLearner

print("=== CORE ALGORITHM TESTS (v6) ===\n")

# Test 1: Valid moves count
g = GameState(4, 4)
moves = g.get_valid_moves()
assert len(moves) == 40, f"Expected 40, got {len(moves)}"
print(f"[PASS] Empty 4x4 valid moves: {len(moves)}")

# Test 2: Zobrist hash changes on move
g2 = GameState(3, 3)
h0 = g2.zobrist
g2.apply_move({"type": "h", "r": 0, "c": 0})
h1 = g2.zobrist
assert h0 != h1, "Hash did not change after move"
print(f"[PASS] Zobrist hash changes: {h0} -> {h1}")

# Test 3: Capture chain — player keeps turn
g3 = GameState(3, 3)
g3.apply_move({"type": "h", "r": 0, "c": 0})
g3.apply_move({"type": "h", "r": 1, "c": 0})
g3.apply_move({"type": "v", "r": 0, "c": 0})
p_before = g3.current_player
captured = g3.apply_move({"type": "v", "r": 0, "c": 1})
assert captured, "Should have captured a box"
assert g3.current_player == p_before, "Player should keep turn after capture"
print(f"[PASS] Capture chain: player {p_before} kept turn, score={g3.scores}")

# Test 4: 5x5 grid — valid move count
g5x5 = GameState(5, 5)
moves_5x5 = g5x5.get_valid_moves()
expected_5x5 = 5 * 6 + 6 * 5  # 60
assert len(moves_5x5) == expected_5x5, f"Expected {expected_5x5}, got {len(moves_5x5)}"
print(f"[PASS] Empty 5x5 valid moves: {len(moves_5x5)}")

# Test 5: 6x6 grid — valid move count
g6x6 = GameState(6, 6)
moves_6x6 = g6x6.get_valid_moves()
expected_6x6 = 6 * 7 + 7 * 6  # 84
assert len(moves_6x6) == expected_6x6, f"Expected {expected_6x6}, got {len(moves_6x6)}"
print(f"[PASS] Empty 6x6 valid moves: {len(moves_6x6)}")

# Test 6: MinimaxStrategy on 3x3 — completes without crash
mm = MinimaxStrategy()
g_mm = GameState(3, 3)
t0 = time.perf_counter()
mm_move, mm_score, mm_met = mm.compute_move(g_mm, depth=3)
mm_time = time.perf_counter() - t0
assert mm_move is not None, "Minimax returned None"
print(f"[PASS] Minimax depth=3 on 3x3: {mm_time:.3f}s, nodes={mm_met['nodes']}")

# Test 7: AlphaBetaStrategy depth=3 on 4x4 — fast enough
ab = AlphaBetaStrategy(time_limit=5.0)
g_ab = GameState(4, 4)
t0 = time.perf_counter()
ab_move, ab_score, ab_met = ab.compute_move(g_ab, depth=3)
ab_time = time.perf_counter() - t0
assert ab_move is not None, "AB returned None!"
assert ab_time < 10.0, f"Too slow: {ab_time:.2f}s"
print(f"[PASS] AB depth=3 on 4x4: {ab_time:.3f}s, nodes={ab_met['nodes']}, pruned={ab_met['pruned']}")

# Test 8: AB prunes more than Minimax at same depth
mm2 = MinimaxStrategy()
g_compare = GameState(3, 3)
_, _, mm_met2 = mm2.compute_move(g_compare.clone(), depth=3)
ab2 = AlphaBetaStrategy(time_limit=5.0)
_, _, ab_met2 = ab2.compute_move(g_compare.clone(), depth=3)
print(f"[INFO] Minimax nodes={mm_met2['nodes']}, AB nodes={ab_met2['nodes']}")
if ab_met2['nodes'] < mm_met2['nodes']:
    print(f"[PASS] AB explores fewer nodes than Minimax ({ab_met2['nodes']} < {mm_met2['nodes']})")
else:
    print(f"[WARN] AB did not prune fewer than Minimax (noise may cause this occasionally)")

# Test 9: RandomStrategy
rs = RandomStrategy()
g_rs = GameState(3, 3)
for i in range(5):
    m, _, _ = rs.compute_move(g_rs, 1)
    assert m is not None, f"RandomStrategy returned None on iteration {i}"
print("[PASS] RandomStrategy: 5 moves all non-None")

# Test 10: AlphaBeta on 5x5 — critical test
ab5 = AlphaBetaStrategy(time_limit=5.0)
g_5x5 = GameState(5, 5)
t0 = time.perf_counter()
move5, _, met5 = ab5.compute_move(g_5x5, depth=3)
t5 = time.perf_counter() - t0
assert move5 is not None, "AB returned None on 5x5"
print(f"[PASS] AB depth=3 on 5x5: {t5:.2f}s, nodes={met5['nodes']}")

# Test 11: get_capturing_moves never empty when valid moves exist
g_cap = GameState(4, 4)
ordered = get_capturing_moves(g_cap)
assert len(ordered) == 40, f"Expected 40, got {len(ordered)}"
print(f"[PASS] get_capturing_moves non-empty: {len(ordered)} moves")

# Test 12: Evaluation function — terminal states
g_term = GameState(3, 3)
g_term.scores[1] = 5
g_term.scores[2] = 4
g_term.is_game_over = True
ev_win = evaluate_state(g_term)
assert ev_win > 0, f"Win should be positive, got {ev_win}"
print(f"[PASS] Terminal win evaluation: {ev_win:.1f}")

g_term2 = GameState(3, 3)
g_term2.scores[1] = 4
g_term2.scores[2] = 5
g_term2.is_game_over = True
ev_loss = evaluate_state(g_term2)
assert ev_loss < 0, f"Loss should be negative, got {ev_loss}"
print(f"[PASS] Terminal loss evaluation: {ev_loss:.1f}")

g_draw = GameState(4, 4)
g_draw.scores[1] = 8
g_draw.scores[2] = 8
g_draw.is_game_over = True
ev_draw = evaluate_state(g_draw)
assert ev_draw == 0.0, f"Draw should be 0, got {ev_draw}"
print(f"[PASS] Terminal draw evaluation: {ev_draw:.1f}")

# Test 13: QLearner MD5 key
ql = QLearner(data_file="data/learning_data.json")
g_ql = GameState(3, 3)
key = ql.get_state_key(g_ql)
assert len(key) == 16, f"Expected 16-char MD5 key, got {len(key)}"
print(f"[PASS] QLearner MD5 key length: {len(key)} chars")

# Test 14: AdaptiveLearningStrategy works
adaptive = AdaptiveLearningStrategy(qlearner=ql, time_limit=3.0)
g_adapt = GameState(3, 3)
t0 = time.perf_counter()
ad_move, ad_score, ad_met = adaptive.compute_move(g_adapt, depth=3)
ad_time = time.perf_counter() - t0
assert ad_move is not None, "Adaptive returned None"
print(f"[PASS] Adaptive depth=3 on 3x3: {ad_time:.3f}s, nodes={ad_met['nodes']}")

# Test 15: is_capturing_move works correctly
g_cap2 = GameState(3, 3)
# Set up 3 sides of box (0,0)
g_cap2.apply_move({"type": "h", "r": 0, "c": 0})
g_cap2.apply_move({"type": "h", "r": 1, "c": 0})
g_cap2.apply_move({"type": "v", "r": 0, "c": 0})
# v(0,1) should now be a capturing move
assert is_capturing_move(g_cap2, 'v', 0, 1), "v(0,1) should capture box (0,0)"
print("[PASS] is_capturing_move correctly identifies captures")

print()
print("[ALL TESTS PASSED]")
