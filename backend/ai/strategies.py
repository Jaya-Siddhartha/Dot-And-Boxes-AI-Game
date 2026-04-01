"""
backend/ai/strategies.py — AI Strategies for Dots & Boxes v6.0

Strategies
----------
1.  RandomStrategy             — picks a random valid move (Easy difficulty)
2.  MinimaxStrategy            — full tree search, no pruning (educational/comparison)
3.  AlphaBetaStrategy          — Alpha-Beta pruning + TT + iterative deepening
4.  AdaptiveLearningStrategy   — Alpha-Beta + Q-Learning feedback (self-improving)

Key fixes in v6:
  - TT stores CLEAN evaluations only — noise applied at root selection only
  - No random depth variation — consistent, predictable depth
  - Noise (ε=0.15) only at leaf evaluation, not stored in TT
  - Proper iterative deepening built into AlphaBeta with time limit
  - AdaptiveLearningStrategy blends AB score with Q-value for move selection
  - Stochastic tie-breaking: moves within EQUI_BAND are treated as equal
"""

import time
import math
import random
from typing import Optional, Tuple, Dict, Any

from backend.ai.heuristics import (
    evaluate_state,
    get_capturing_moves,
    is_capturing_move,
)

# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------
DIFFICULTY_EASY   = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD   = "hard"
DIFFICULTY_EXPERT = "expert"

_DIFFICULTY_DEPTH = {
    DIFFICULTY_EASY:   1,
    DIFFICULTY_MEDIUM: 2,
    DIFFICULTY_HARD:   5,
    DIFFICULTY_EXPERT: 7,
}

_DIFFICULTY_TIME = {
    DIFFICULTY_EASY:   0.3,
    DIFFICULTY_MEDIUM: 1.0,
    DIFFICULTY_HARD:   3.0,
    DIFFICULTY_EXPERT: 8.0,
}

# Transposition table entry flags
_EXACT = 0
_LOWER = 1   # alpha cutoff — actual value >= stored
_UPPER = 2   # beta  cutoff — actual value <= stored

_TT_MAX_SIZE = 200_000

# Equivalence band: moves within this score range are treated as ties
# Narrower than before (was 1.5) to keep play smart while allowing variety
_EQUI_BAND = 0.5

# Small noise at leaves for tie-breaking among truly equal positions
_LEAF_NOISE = 0.15

MoveResult = Tuple[Optional[dict], float, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Strategy:
    """All AI strategies implement compute_move(state, depth) -> MoveResult."""

    def compute_move(self, state, depth: int) -> MoveResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Random Strategy (Easy difficulty)
# ---------------------------------------------------------------------------
class RandomStrategy(Strategy):
    """
    Randomly selects a valid move.
    Prefers capturing moves so it at least takes free boxes,
    but otherwise plays randomly — suitable for Easy difficulty.
    """

    def compute_move(self, state, depth: int = 1) -> MoveResult:
        start = time.perf_counter()
        valid = state.get_valid_moves()
        if not valid or state.is_game_over:
            return None, evaluate_state(state), {"time": 0.0, "nodes": 0, "pruned": 0}

        captures = [m for m in valid
                    if is_capturing_move(state, m['type'], m['r'], m['c'])]
        move = random.choice(captures) if captures else random.choice(valid)
        return move, evaluate_state(state), {
            "time": time.perf_counter() - start,
            "nodes": len(valid),
            "pruned": 0,
        }


# ---------------------------------------------------------------------------
# 2. Minimax (no pruning) — comparison panel + educational
# ---------------------------------------------------------------------------
class MinimaxStrategy(Strategy):
    """
    Plain Minimax without pruning.

    Features:
      - Clean evaluation (no draw_bias, no handicaps)
      - Small noise at leaves only for tie-breaking
      - Stochastic tie-breaking with EQUI_BAND
      - Node counting for comparison metrics
    """

    def __init__(self):
        self.nodes_expanded = 0

    def compute_move(self, state, depth: int) -> MoveResult:
        start = time.perf_counter()
        self.nodes_expanded = 0

        valid = get_capturing_moves(state)
        if not valid or state.is_game_over:
            return None, evaluate_state(state), {
                "time": time.perf_counter() - start, "nodes": 0, "pruned": 0
            }

        maximize = (state.current_player == 1)
        scored   = []

        for move in valid:
            ns = state.clone()
            ns.apply_move(move)
            score = self._minimax(ns, depth - 1, ns.current_player == 1)
            scored.append((score, move))

        if maximize:
            best_score = max(s for s, _ in scored)
        else:
            best_score = min(s for s, _ in scored)

        # Stochastic tie-breaking among near-equal moves
        candidates = [m for s, m in scored if abs(s - best_score) <= _EQUI_BAND]
        best_move  = random.choice(candidates) if candidates else scored[0][1]

        return best_move, best_score, {
            "time":   time.perf_counter() - start,
            "nodes":  self.nodes_expanded,
            "pruned": 0,
        }

    def _minimax(self, state, depth: int, maximize: bool) -> float:
        self.nodes_expanded += 1

        if state.is_game_over or depth <= 0:
            return self._leaf_eval(state)

        moves = get_capturing_moves(state)
        if not moves:
            return self._leaf_eval(state)

        if maximize:
            score = -math.inf
            for move in moves:
                ns = state.clone()
                ns.apply_move(move)
                score = max(score, self._minimax(ns, depth - 1, ns.current_player == 1))
            return score
        else:
            score = math.inf
            for move in moves:
                ns = state.clone()
                ns.apply_move(move)
                score = min(score, self._minimax(ns, depth - 1, ns.current_player == 1))
            return score

    def _leaf_eval(self, state) -> float:
        """Evaluate with small noise for tie-breaking only."""
        base = evaluate_state(state)
        return base + random.uniform(-_LEAF_NOISE, _LEAF_NOISE)


# ---------------------------------------------------------------------------
# 3. Alpha-Beta with TT + Iterative Deepening
# ---------------------------------------------------------------------------
class AlphaBetaStrategy(Strategy):
    """
    Alpha-Beta pruning with:
      - Move ordering (capturing → safe → risky)
      - Transposition table (Zobrist hash) — stores CLEAN evals only
      - Built-in iterative deepening with time limit
      - Noise ONLY at root move selection for variety
      - Proper aspiration: TT entries reused across depths
    """

    def __init__(self, time_limit: float = 3.0):
        self.nodes_expanded  = 0
        self.branches_pruned = 0
        self.tt_hits         = 0
        self.time_limit      = time_limit
        self._tt: Dict[int, tuple] = {}   # hash → (depth, flag, score)

    def _clear_tt_if_needed(self):
        if len(self._tt) > _TT_MAX_SIZE:
            # Keep the deeper entries (more valuable)
            entries = sorted(self._tt.items(), key=lambda x: x[1][0], reverse=True)
            self._tt = dict(entries[:_TT_MAX_SIZE // 2])

    def compute_move(self, state, depth: int) -> MoveResult:
        start = time.perf_counter()
        self.nodes_expanded  = 0
        self.branches_pruned = 0
        self.tt_hits         = 0
        self._clear_tt_if_needed()

        valid = get_capturing_moves(state)
        if not valid or state.is_game_over:
            return None, evaluate_state(state), {
                "time": time.perf_counter() - start, "nodes": 0, "pruned": 0
            }

        # --- Iterative deepening ---
        best_move  = valid[0]
        best_score = -math.inf if state.current_player == 1 else math.inf

        for d in range(1, depth + 1):
            elapsed = time.perf_counter() - start
            if elapsed >= self.time_limit:
                break

            move, score = self._search_root(state, d, valid)

            if move is not None:
                best_move  = move
                best_score = score

            # Early exit if we found a decisive result
            if abs(best_score) >= 999:
                break

        return best_move, best_score, {
            "time":    time.perf_counter() - start,
            "nodes":   self.nodes_expanded,
            "pruned":  self.branches_pruned,
            "tt_hits": self.tt_hits,
        }

    def _search_root(self, state, depth: int, ordered_moves: list):
        """Search at root level with tie-breaking noise."""
        maximize = (state.current_player == 1)
        alpha    = -math.inf
        beta     = math.inf
        scored   = []

        for move in ordered_moves:
            ns = state.clone()
            ns.apply_move(move)
            score = self._alphabeta(ns, depth - 1, alpha, beta,
                                    ns.current_player == 1)
            scored.append((score, move))

            if maximize:
                if score > alpha:
                    alpha = score
            else:
                if score < beta:
                    beta = score

        if not scored:
            return None, 0.0

        if maximize:
            best_score = max(s for s, _ in scored)
        else:
            best_score = min(s for s, _ in scored)

        # Stochastic tie-breaking — add small noise at root selection only
        noisy_scored = [(s + random.uniform(-_LEAF_NOISE, _LEAF_NOISE), m)
                        for s, m in scored]

        if maximize:
            candidates = [(s, m) for s, m in noisy_scored
                          if abs(s - best_score) <= _EQUI_BAND]
        else:
            candidates = [(s, m) for s, m in noisy_scored
                          if abs(s - best_score) <= _EQUI_BAND]

        if candidates:
            best_move = random.choice(candidates)[1]
        else:
            best_move = scored[0][1]

        return best_move, best_score

    def _alphabeta(self, state, depth: int, alpha: float, beta: float,
                   maximize: bool) -> float:
        self.nodes_expanded += 1

        if state.is_game_over:
            return evaluate_state(state)

        if depth <= 0:
            return evaluate_state(state)  # CLEAN eval — no noise in TT

        # TT lookup
        zh = state.zobrist
        tt_entry = self._tt.get(zh)
        if tt_entry is not None:
            tt_depth, tt_flag, tt_score = tt_entry
            if tt_depth >= depth:
                self.tt_hits += 1
                if tt_flag == _EXACT:
                    return tt_score
                elif tt_flag == _LOWER:
                    alpha = max(alpha, tt_score)
                elif tt_flag == _UPPER:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        moves = get_capturing_moves(state)
        if not moves:
            return evaluate_state(state)

        original_alpha = alpha
        best = -math.inf if maximize else math.inf

        if maximize:
            for move in moves:
                ns = state.clone()
                ns.apply_move(move)
                val  = self._alphabeta(ns, depth - 1, alpha, beta,
                                       ns.current_player == 1)
                best  = max(best, val)
                alpha = max(alpha, best)
                if alpha >= beta:
                    self.branches_pruned += 1
                    break
        else:
            for move in moves:
                ns = state.clone()
                ns.apply_move(move)
                val  = self._alphabeta(ns, depth - 1, alpha, beta,
                                       ns.current_player == 1)
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha:
                    self.branches_pruned += 1
                    break

        # Store CLEAN evaluation in TT
        if best <= original_alpha:
            flag = _UPPER
        elif best >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        self._tt[zh] = (depth, flag, best)

        return best


class DepthLimitedAlphaBetaStrategy(AlphaBetaStrategy):
    """
    Alpha-Beta variant that searches exactly to the requested depth.

    This keeps the frontend depth selector intuitive and avoids extra work from
    iterative deepening during normal interactive play.
    """

    def compute_move(self, state, depth: int) -> MoveResult:
        start = time.perf_counter()
        self.nodes_expanded = 0
        self.branches_pruned = 0
        self.tt_hits = 0
        self._clear_tt_if_needed()

        valid = get_capturing_moves(state)
        if not valid or state.is_game_over:
            return None, evaluate_state(state), {
                "time": time.perf_counter() - start, "nodes": 0, "pruned": 0
            }

        best_move, best_score = self._search_root(state, max(1, depth), valid)
        return best_move, best_score, {
            "time": time.perf_counter() - start,
            "nodes": self.nodes_expanded,
            "pruned": self.branches_pruned,
            "tt_hits": self.tt_hits,
        }


# ---------------------------------------------------------------------------
# 4. Adaptive Learning Strategy (Alpha-Beta + Q-Learning)
# ---------------------------------------------------------------------------
class AdaptiveLearningStrategy(Strategy):
    """
    Hybrid AI: Alpha-Beta search + Q-Learning feedback.
    
    The Q-learning component runs silently in the backend:
      - Every move updates the Q-table
      - Q-values influence move selection by blending with AB scores
      - Over many games, the AI learns which positions lead to wins
    
    Blend: final_score = 0.7 * ab_score + 0.3 * q_value
    This gives the sound tree search priority while incorporating learned patterns.
    """

    def __init__(self, qlearner, time_limit: float = 3.0):
        self.qlearner    = qlearner
        self.time_limit  = time_limit
        self._ab = AlphaBetaStrategy(time_limit=time_limit)

    def compute_move(self, state, depth: int = None) -> MoveResult:
        start = time.perf_counter()
        valid = get_capturing_moves(state)
        if not valid or state.is_game_over:
            return None, evaluate_state(state), {
                "time": 0.0, "nodes": 0, "pruned": 0
            }

        if depth is None:
            depth = 2

        state_key = self.qlearner.get_state_key(state)
        scored    = []

        # Get the AB strategy's internal metrics
        self._ab.nodes_expanded  = 0
        self._ab.branches_pruned = 0
        self._ab.tt_hits         = 0

        # First, get AB scores for all moves
        maximize = (state.current_player == 1)
        alpha    = -math.inf
        beta     = math.inf

        for move in valid:
            ns = state.clone()
            ns.apply_move(move)
            ab_score = self._ab._alphabeta(
                ns, depth - 1, alpha, beta, ns.current_player == 1
            )

            # Get Q-value for this move
            move_key = f"{move['type']}_{move['r']}_{move['c']}"
            q_val = self.qlearner.get_q_value(state_key, move_key,
                                               player=state.current_player)

            # Normalize AB score to roughly [-10, 10] range for blending
            ab_normalized = max(-10.0, min(10.0, ab_score / 100.0))

            # Blend: 70% tree search, 30% learned Q-value
            blend_score = 0.7 * ab_normalized + 0.3 * q_val

            scored.append((blend_score, ab_score, move, q_val))

            if maximize:
                alpha = max(alpha, ab_score)
            else:
                beta = min(beta, ab_score)

        # Select best move based on blended score
        best_blend = max(s for s, _, _, _ in scored) if maximize else min(s for s, _, _, _ in scored)
        candidates = [item for item in scored
                      if abs(item[0] - best_blend) <= _EQUI_BAND * 0.1]
        if not candidates:
            candidates = scored

        chosen = random.choice(candidates)
        best_move = chosen[2]
        best_ab_score = chosen[1]
        best_q_val = chosen[3]

        return best_move, best_ab_score, {
            "time":    time.perf_counter() - start,
            "nodes":   self._ab.nodes_expanded,
            "pruned":  self._ab.branches_pruned,
            "q_value": best_q_val,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_strategy(name: str, difficulty: str = DIFFICULTY_HARD,
                    qlearner=None, time_limit: float = 3.0) -> Strategy:
    """
    Create a strategy instance from name and difficulty.

    Strategies:
      'minimax'    → MinimaxStrategy (pure, no pruning)
      'alphabeta'  → AlphaBetaStrategy (with pruning + iterative deepening)
      'adaptive'   → AdaptiveLearningStrategy (AB + Q-Learning)

    Difficulty controls time budget / search configuration, not the algorithm
    identity chosen by the user.
    """
    d = difficulty.lower()

    if name == "random":
        return RandomStrategy()

    if name == "minimax":
        return MinimaxStrategy()

    if name == "alphabeta":
        tl = _DIFFICULTY_TIME.get(d, 3.0)
        return DepthLimitedAlphaBetaStrategy(time_limit=tl)

    if name == "adaptive":
        if qlearner:
            tl = _DIFFICULTY_TIME.get(d, 3.0)
            return AdaptiveLearningStrategy(qlearner=qlearner, time_limit=tl)
        # Fallback to AlphaBeta if no qlearner available
        tl = _DIFFICULTY_TIME.get(d, 3.0)
        return AlphaBetaStrategy(time_limit=tl)

    # Fallback
    return AlphaBetaStrategy(time_limit=time_limit)


def get_depth_for_difficulty(difficulty: str) -> int:
    return _DIFFICULTY_DEPTH.get(difficulty.lower(), 2)
