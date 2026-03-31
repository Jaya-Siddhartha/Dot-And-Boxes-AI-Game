"""Live API integration tests v6.0 — tests all endpoints."""
import requests
import time
import sys

BASE = "http://localhost:8000/api"

def test(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        sys.exit(1)

print("=== LIVE API TESTS (v6) ===\n")

# Test 1: GET /state
r = requests.get(f"{BASE}/state")
test("GET /state", r.status_code == 200)
d = r.json()
print(f"  Grid: {d['rows']}x{d['cols']}")

# Test 2: Start 4x4 game
r = requests.post(f"{BASE}/start-game", json={"rows": 4, "cols": 4, "mode": "hvai", "strategy": "alphabeta"})
test("POST /start-game 4x4", r.status_code == 200 and r.json()["rows"] == 4)

# Test 3: Start 5x5 game — CRITICAL FIX TEST
r = requests.post(f"{BASE}/start-game", json={"rows": 5, "cols": 5, "mode": "hvai", "strategy": "minimax"})
d = r.json()
test("POST /start-game 5x5", r.status_code == 200 and d["rows"] == 5 and d["cols"] == 5)

# Test 4: Start 6x6 game
r = requests.post(f"{BASE}/start-game", json={"rows": 6, "cols": 6, "mode": "hvai", "strategy": "alphabeta"})
d = r.json()
test("POST /start-game 6x6", r.status_code == 200 and d["rows"] == 6)

# Test 5: Start 3x3 game
r = requests.post(f"{BASE}/start-game", json={"rows": 3, "cols": 3, "mode": "hvh", "strategy": "minimax"})
d = r.json()
test("POST /start-game 3x3", r.status_code == 200 and d["rows"] == 3)

# Test 6: Make a human move on 3x3
r = requests.post(f"{BASE}/move", json={"m_type": "h", "r": 0, "c": 0})
test("POST /move", r.status_code == 200)

# Test 7: Check state after move
r = requests.get(f"{BASE}/state")
state = r.json()
test("State after move — line drawn", state["horizontal_lines"][0][0] == True)

# Test 8: AI move with minimax
r = requests.post(f"{BASE}/start-game", json={"rows": 3, "cols": 3, "mode": "hvai", "strategy": "minimax"})
r = requests.post(f"{BASE}/move", json={"m_type": "h", "r": 0, "c": 0})  # human moves first
r = requests.post(f"{BASE}/ai-move", json={"strategy": "minimax", "depth": 2, "difficulty": "medium"})
test("POST /ai-move minimax", r.status_code == 200 and r.json()["move"] is not None)

# Test 9: AI move with alphabeta
r = requests.post(f"{BASE}/start-game", json={"rows": 3, "cols": 3, "mode": "hvai", "strategy": "alphabeta"})
r = requests.post(f"{BASE}/move", json={"m_type": "v", "r": 0, "c": 0})
r = requests.post(f"{BASE}/ai-move", json={"strategy": "alphabeta", "depth": 3, "difficulty": "hard"})
test("POST /ai-move alphabeta", r.status_code == 200)

# Test 10: AI move with adaptive
r = requests.post(f"{BASE}/start-game", json={"rows": 3, "cols": 3, "mode": "hvai", "strategy": "adaptive"})
r = requests.post(f"{BASE}/move", json={"m_type": "h", "r": 0, "c": 1})
r = requests.post(f"{BASE}/ai-move", json={"strategy": "adaptive", "depth": 3, "difficulty": "hard"})
test("POST /ai-move adaptive", r.status_code == 200)

# Test 11: Comparison
r = requests.post(f"{BASE}/start-game", json={"rows": 3, "cols": 3, "mode": "hvai", "strategy": "alphabeta"})
r = requests.post(f"{BASE}/comparison", json={"depth": 2})
test("POST /comparison depth=2", r.status_code == 200)
cr = r.json()
print(f"  MM nodes={cr['minimax']['nodes']} AB nodes={cr['alphabeta']['nodes']} savings={cr['pruning_savings_pct']}%")

# Test 12: History
r = requests.get(f"{BASE}/history")
test("GET /history", r.status_code == 200)
h = r.json()
print(f"  {len(h.get('games', []))} games stored")

# Test 13: Stats
r = requests.get(f"{BASE}/stats")
test("GET /stats", r.status_code == 200)

# Test 14: Learning stats (backend only)  
r = requests.get(f"{BASE}/learning-stats")
test("GET /learning-stats", r.status_code == 200)
ls = r.json()
print(f"  states_known={ls['stats'].get('states_known', 0)} games_played={ls['stats'].get('games_played', 0)}")

# Test 15: Suggest move
r = requests.post(f"{BASE}/start-game", json={"rows": 3, "cols": 3, "mode": "hvai", "strategy": "alphabeta"})
r = requests.get(f"{BASE}/suggest?depth=2")
test("GET /suggest", r.status_code == 200 and r.json()["move"] is not None)

# Test 16: AI vs AI with 5x5 grid — CRITICAL
r = requests.post(f"{BASE}/start-game", json={"rows": 5, "cols": 5, "mode": "aivai", "strategy": "minimax_vs_alphabeta"})
r = requests.post(f"{BASE}/ai-vs-ai", json={"strat1": "minimax", "strat2": "alphabeta", "depth": 2, "delay": 0.1, "rows": 5, "cols": 5})
test("POST /ai-vs-ai 5x5", r.status_code == 200)
# Wait for game to complete
time.sleep(12)
r = requests.get(f"{BASE}/state")
state = r.json()
test("AI vs AI 5x5 grid used", state["rows"] == 5 and state["cols"] == 5)
print(f"  5x5 game state: P1={state['scores']['1']} P2={state['scores']['2']} over={state['is_game_over']}")

# Test 17: Balance stats
r = requests.get(f"{BASE}/balance-stats")
test("GET /balance-stats", r.status_code == 200)

# Test 18: Reset
r = requests.post(f"{BASE}/reset")
test("POST /reset", r.status_code == 200)

print()
print("[ALL LIVE API TESTS PASSED]")
