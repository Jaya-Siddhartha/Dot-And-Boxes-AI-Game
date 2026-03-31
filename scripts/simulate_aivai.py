import sys
import os
import argparse
import asyncio
from datetime import datetime

# Add root folder to path so we can import backend.engine and backend.ai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.engine.game import GameState
from backend.ai.strategies import AlphaBetaStrategy, AdaptiveLearningStrategy
from backend.learning.qlearning import QLearner

async def run_simulation(count=100, rows=3, cols=3, depth=3):
    print(f"STARTING simulation of {count} AI vs AI games ({rows}x{cols}, depth={depth})")
    print(f"Note: Using identical high-depth heuristics to target 80-90% draw rate.")
    
    q = QLearner()
    # Using AlphaBeta for speed in simulation, but it uses the same enhance_heuristic from heuristics.py
    strat1 = AlphaBetaStrategy()
    strat2 = AlphaBetaStrategy()
    
    results = {0: 0, 1: 0, 2: 0} # 0=Draw, 1=P1 Win, 2=P2 Win
    scores_p1 = []
    scores_p2 = []
    
    start_time = datetime.now()
    
    for i in range(count):
        state = GameState(rows=rows, cols=cols)
        while not state.is_game_over:
            cur = state.current_player
            ai = strat1 if cur == 1 else strat2
            
            # Use same depth for both to ensure balance
            move, _, _ = await asyncio.to_thread(ai.compute_move, state.clone(), depth)
            if move is None:
                break
            state.apply_move(move)
            
        winner = 0
        if state.scores[1] > state.scores[2]: winner = 1
        elif state.scores[2] > state.scores[1]: winner = 2
        
        results[winner] += 1
        scores_p1.append(state.scores[1])
        scores_p2.append(state.scores[2])
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{count} games...")
            
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    draw_pct = (results[0] / count) * 100
    p1_win_pct = (results[1] / count) * 100
    p2_win_pct = (results[2] / count) * 100
    
    print("\n" + "="*40)
    print("SIMULATION RESULTS")
    print("="*40)
    print(f"Total Games: {count}")
    print(f"Duration:    {duration:.2f}s (avg {duration/count:.2f}s per game)")
    print(f"Draws:       {results[0]} ({draw_pct:.1f}%)")
    print(f"P1 Wins:     {results[1]} ({p1_win_pct:.1f}%)")
    print(f"P2 Wins:     {results[2]} ({p2_win_pct:.1f}%)")
    print(f"Avg Score:   P1: {sum(scores_p1)/count:.1f} | P2: {sum(scores_p2)/count:.1f}")
    print("="*40)
    
    if draw_pct >= 80:
        print("SUCCESS: Target draw rate of 80-90% reached.")
    else:
        print("WARNING: Draw rate below target. Consider increasing search depth.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()
    
    asyncio.run(run_simulation(count=args.count, rows=args.rows, cols=args.cols, depth=args.depth))
