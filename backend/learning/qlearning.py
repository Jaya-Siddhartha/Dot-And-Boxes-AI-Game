"""
backend/learning/qlearning.py — Q-Learning System v4.0

Improvements over v3:
  1. MD5-compressed state keys — shorter JSON keys, smaller file, faster lookups
  2. Q-table size limit (50,000 states) — prunes lowest-value states when exceeded
  3. Epsilon decay — starts at 0.2, decays to 0.05 as games accumulate
  4. Per-player Q-tables — separate tables for P1 and P2 improve learning quality
  5. Thread-safe save (uses temp file + atomic rename to avoid corruption)
  6. Robust load: handles both legacy (plain dict) and new (versioned) formats
"""

import json
import os
import hashlib
import random
import tempfile
from typing import Dict, Optional


# Maximum number of state entries before pruning
_Q_TABLE_MAX_STATES = 50_000

# Epsilon schedule: start → end, games to reach end
_EPSILON_START  = 0.20
_EPSILON_END    = 0.05
_EPSILON_GAMES  = 200   # after 200 games, epsilon stabilises at end value


class QLearner:
    """
    Q-Learning agent for Dots & Boxes.

    Q(s, a) is stored per-player (separate tables for P1 and P2).
    State keys are MD5 hashes of the board configuration —
    much shorter than the raw binary strings used previously.
    """

    def __init__(
        self,
        data_file:  str   = "data/learning_data.json",
        alpha:      float = 0.15,   # learning rate
        gamma:      float = 0.90,   # discount factor
    ):
        self.data_file    = data_file
        self.alpha        = alpha
        self.gamma        = gamma
        self.games_played = 0

        # Separate Q-tables per player for higher-quality learning
        # q_table[player][state_key][move_key] = float
        self.q_table: Dict[int, Dict[str, Dict[str, float]]] = {1: {}, 2: {}}

        self.load_data()

    # ------------------------------------------------------------------
    # Epsilon-greedy — decays over time
    # ------------------------------------------------------------------
    @property
    def epsilon(self) -> float:
        """Current exploration rate — decays from 0.2 to 0.05 over 200 games."""
        t = min(self.games_played / max(_EPSILON_GAMES, 1), 1.0)
        return _EPSILON_START + t * (_EPSILON_END - _EPSILON_START)

    # ------------------------------------------------------------------
    # State / move key helpers
    # ------------------------------------------------------------------
    def get_state_key(self, state) -> str:
        """
        Return a compact MD5 hash of the board as a state key.
        Much shorter than the raw binary string (32 chars vs 40+).
        """
        h_bits = "".join("1" if x else "0" for row in state.horizontal_lines for x in row)
        v_bits = "".join("1" if x else "0" for row in state.vertical_lines   for x in row)
        raw = f"H{h_bits}V{v_bits}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]   # 16 hex chars = 64-bit

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load_data(self) -> None:
        if not os.path.exists(self.data_file):
            return
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        if not isinstance(data, dict):
            return

        meta = data.get("_meta", {})
        self.games_played = meta.get("games_played", 0)

        # New format: versioned per-player tables
        if "q_table_p1" in data:
            self.q_table[1] = data.get("q_table_p1", {})
            self.q_table[2] = data.get("q_table_p2", {})
        elif "q_table" in data:
            # Legacy v3 — single shared table, migrate to P1 only
            legacy = data["q_table"]
            if isinstance(legacy, dict):
                self.q_table[1] = legacy
        # else: unexpected format — start clean

    def save_data(self) -> None:
        """Atomically save Q-table JSON to disk (temp file + rename)."""
        self._maybe_prune()
        os.makedirs(os.path.dirname(self.data_file) if os.path.dirname(self.data_file) else ".", exist_ok=True)

        total_moves = sum(
            len(moves)
            for table in self.q_table.values()
            for moves in table.values()
        )
        payload = {
            "_meta": {
                "version":       4,
                "games_played":  self.games_played,
                "states_p1":     len(self.q_table[1]),
                "states_p2":     len(self.q_table[2]),
                "total_moves":   total_moves,
            },
            "q_table_p1": self.q_table[1],
            "q_table_p2": self.q_table[2],
        }

        # Atomic write: write to a temp file next to the target, then rename
        dir_name   = os.path.dirname(self.data_file) or "."
        base_name  = os.path.basename(self.data_file)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=f".{base_name}.tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"))   # compact format
            os.replace(tmp_path, self.data_file)
        except Exception:
            # Fallback: direct write
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"))

    def _maybe_prune(self) -> None:
        """If Q-table exceeds size limit, remove the least-used states."""
        for player in (1, 2):
            table = self.q_table[player]
            if len(table) <= _Q_TABLE_MAX_STATES:
                continue
            # Score each state by max absolute Q-value; keep highest
            scored = sorted(
                table.items(),
                key=lambda kv: max(abs(v) for v in kv[1].values()) if kv[1] else 0.0,
                reverse=True,
            )
            self.q_table[player] = dict(scored[:_Q_TABLE_MAX_STATES])

    # ------------------------------------------------------------------
    # Q-value access
    # ------------------------------------------------------------------
    def get_q_value(self, state_key: str, move_key: str, player: int = 1) -> float:
        table = self.q_table.get(player, self.q_table[1])
        return table.get(state_key, {}).get(move_key, 0.0)

    def set_q_value(self, state_key: str, move_key: str, value: float, player: int = 1) -> None:
        table = self.q_table.setdefault(player, {})
        if state_key not in table:
            table[state_key] = {}
        table[state_key][move_key] = value

    def max_q_value(self, state_key: str, valid_moves: list, player: int = 1) -> float:
        if not valid_moves:
            return 0.0
        table = self.q_table.get(player, {})
        state_moves = table.get(state_key, {})
        if not state_moves:
            return 0.0
        max_v = -float("inf")
        for m in valid_moves:
            if isinstance(m, dict):
                mk = f"{m['type']}_{m['r']}_{m['c']}"
            else:
                mk = f"{m[0]}_{m[1]}_{m[2]}"
            v = state_moves.get(mk, 0.0)
            if v > max_v:
                max_v = v
        return max_v if max_v != -float("inf") else 0.0

    def update_q_value(
        self,
        state_key:       str,
        move_key:        str,
        reward:          float,
        next_state_key:  str,
        next_valid_moves: list,
        player:          int = 1,
    ) -> None:
        """Bellman backup: Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s') − Q(s,a)]"""
        current_q  = self.get_q_value(state_key, move_key, player)
        max_next_q = self.max_q_value(next_state_key, next_valid_moves, player)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.set_q_value(state_key, move_key, new_q, player)

    # ------------------------------------------------------------------
    # Epsilon-greedy move selection (for training sessions)
    # ------------------------------------------------------------------
    def choose_move(self, state_key: str, valid_moves: list, player: int = 1):
        """Return a move using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        # Exploit
        table      = self.q_table.get(player, {})
        state_mvs  = table.get(state_key, {})
        best_move  = None
        best_q     = -float("inf")
        for m in valid_moves:
            if isinstance(m, dict):
                mk = f"{m['type']}_{m['r']}_{m['c']}"
            else:
                mk = f"{m[0]}_{m[1]}_{m[2]}"
            q = state_mvs.get(mk, 0.0)
            if q > best_q:
                best_q = q; best_move = m
        return best_move if best_move is not None else random.choice(valid_moves)

    def increment_games(self) -> None:
        self.games_played += 1

    # ------------------------------------------------------------------
    # Stats / export for dashboard
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        total_moves = sum(
            len(mv)
            for table in self.q_table.values()
            for mv in table.values()
        )
        return {
            "states_known":     len(self.q_table[1]) + len(self.q_table[2]),
            "moves_learned":    total_moves,
            "games_played":     self.games_played,
            "learning_rate":    self.alpha,
            "discount_factor":  self.gamma,
            "exploration_rate": round(self.epsilon, 4),
        }

    def export_top_moves(self, top_n: int = 10) -> list:
        """Return the top N (move, q_value) entries across both player tables."""
        all_entries = []
        for player, table in self.q_table.items():
            for state_key, moves in table.items():
                for move_key, q_val in moves.items():
                    all_entries.append({
                        "player":  player,
                        "state":   state_key[:12] + "…",
                        "move":    move_key,
                        "q_value": round(q_val, 4),
                    })
        all_entries.sort(key=lambda x: x["q_value"], reverse=True)
        return all_entries[:top_n]
