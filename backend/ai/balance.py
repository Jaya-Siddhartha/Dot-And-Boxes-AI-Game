"""
backend/ai/balance.py — Game Outcome Tracker v6.0

Simplified from v5's FairnessController:
  - No longer injects handicaps or biases into evaluation
  - Acts purely as a statistics tracker for game outcomes
  - Maintains a sliding window of the last N games
  - Reports win/draw/loss percentages for dashboard display
"""


class FairnessController:
    """
    Tracks AI vs AI game outcomes over a sliding window.
    
    Provides statistics on win/draw/loss distribution
    for monitoring game balance. Does NOT modify AI behavior.
    """

    _WINDOW_SIZE = 30   # Track last 30 games

    def __init__(self):
        self._outcomes: list = []   # list of winner values (1, 2, or 0=draw)

    def record_result(self, winner: int) -> None:
        """Record a game outcome. winner: 1=P1 wins, 2=P2 wins, 0=draw."""
        self._outcomes.append(winner)
        if len(self._outcomes) > self._WINDOW_SIZE:
            self._outcomes = self._outcomes[-self._WINDOW_SIZE:]

    def get_stats(self) -> dict:
        """Return current balance statistics."""
        total = len(self._outcomes)
        if total == 0:
            return {
                "games_tracked": 0,
                "p1_wins": 0,
                "p2_wins": 0,
                "draws": 0,
                "p1_win_pct": 0.0,
                "p2_win_pct": 0.0,
                "draw_pct": 0.0,
            }

        p1_wins = sum(1 for w in self._outcomes if w == 1)
        p2_wins = sum(1 for w in self._outcomes if w == 2)
        draws   = sum(1 for w in self._outcomes if w == 0)

        return {
            "games_tracked": total,
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "draws": draws,
            "p1_win_pct": round(100 * p1_wins / total, 1),
            "p2_win_pct": round(100 * p2_wins / total, 1),
            "draw_pct":   round(100 * draws / total, 1),
        }

    def get_handicap(self, player: int) -> float:
        """Legacy compatibility — always returns 0.0 (no handicaps)."""
        return 0.0

    def reset(self) -> None:
        """Clear all tracked outcomes."""
        self._outcomes.clear()
