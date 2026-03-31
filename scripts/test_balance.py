"""Test the simplified FairnessController (stats tracker)."""
import sys
sys.path.insert(0, ".")

from backend.ai.balance import FairnessController

print("=== BALANCE SYSTEM TESTS (v6) ===\n")

# Test 1: Empty controller
fc = FairnessController()
stats = fc.get_stats()
assert stats["games_tracked"] == 0
print("[PASS] Empty controller: 0 games tracked")

# Test 2: Record results
fc.record_result(1)
fc.record_result(2)
fc.record_result(0)
fc.record_result(1)
stats = fc.get_stats()
assert stats["games_tracked"] == 4
assert stats["p1_wins"] == 2
assert stats["p2_wins"] == 1
assert stats["draws"] == 1
print(f"[PASS] 4 games recorded: P1={stats['p1_wins']} P2={stats['p2_wins']} D={stats['draws']}")

# Test 3: Percentages
assert stats["p1_win_pct"] == 50.0
assert stats["draw_pct"] == 25.0
print(f"[PASS] Percentages correct: P1={stats['p1_win_pct']}% Draw={stats['draw_pct']}%")

# Test 4: Handicap always 0 (no more handicap injection)
assert fc.get_handicap(1) == 0.0
assert fc.get_handicap(2) == 0.0
print("[PASS] Handicaps always 0.0 (no injection)")

# Test 5: Sliding window
fc2 = FairnessController()
for i in range(40):
    fc2.record_result(1)
stats2 = fc2.get_stats()
assert stats2["games_tracked"] == 30, f"Expected 30 (window), got {stats2['games_tracked']}"
print(f"[PASS] Sliding window caps at {stats2['games_tracked']} games")

# Test 6: Reset
fc2.reset()
assert fc2.get_stats()["games_tracked"] == 0
print("[PASS] Reset clears all outcomes")

print()
print("[ALL BALANCE TESTS PASSED]")
