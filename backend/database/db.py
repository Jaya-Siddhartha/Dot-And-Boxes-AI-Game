"""
backend/database/db.py — Async SQLite layer using aiosqlite.

Tables:
  games  — one row per completed game
  moves  — one row per move with AI metrics

New in v2:
  get_win_stats() — aggregated stats for dashboard
"""
import aiosqlite
import json
import os
from datetime import datetime

DB_PATH = "data/games.db"


async def get_db() -> aiosqlite.Connection:
    os.makedirs("data", exist_ok=True)
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


async def init_db() -> None:
    """Create tables if they don't exist yet."""
    os.makedirs("data", exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                mode        TEXT NOT NULL,        -- hvh | hvai | aivai
                strategy    TEXT,                 -- minimax | alphabeta | adaptive
                rows        INTEGER,
                cols        INTEGER,
                winner      INTEGER,              -- 1 | 2 | 0 (draw)
                score_p1    INTEGER,
                score_p2    INTEGER,
                started_at  TEXT,
                ended_at    TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS moves (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id     INTEGER REFERENCES games(id) ON DELETE CASCADE,
                move_num    INTEGER,
                player      INTEGER,
                move_type   TEXT,
                move_r      INTEGER,
                move_c      INTEGER,
                nodes       INTEGER,
                pruned      INTEGER,
                exec_time   REAL,
                q_value     REAL,
                strategy    TEXT
            )
        """)
        await db.commit()


async def save_game(mode: str, strategy: str, rows: int, cols: int,
                    winner: int, score_p1: int, score_p2: int,
                    moves_data: list, started_at: str) -> int:
    """Persist a completed game and its moves. Returns the new game id."""
    async with aiosqlite.connect(DB_PATH) as db:
        ended_at = datetime.utcnow().isoformat()
        cursor = await db.execute(
            """INSERT INTO games (mode, strategy, rows, cols, winner, score_p1, score_p2, started_at, ended_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (mode, strategy, rows, cols, winner, score_p1, score_p2, started_at, ended_at)
        )
        game_id = cursor.lastrowid
        for m in moves_data:
            await db.execute(
                """INSERT INTO moves
                     (game_id, move_num, player, move_type, move_r, move_c,
                      nodes, pruned, exec_time, q_value, strategy)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (game_id, m.get("move_num"), m.get("player"),
                 m.get("move_type"), m.get("move_r"), m.get("move_c"),
                 m.get("nodes", 0), m.get("pruned", 0),
                 m.get("exec_time", 0.0), m.get("q_value"),
                 m.get("strategy"))
            )
        await db.commit()
        return game_id


async def list_games(limit: int = 50) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM games ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_game(game_id: int) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM games WHERE id=?", (game_id,))
        game = await cur.fetchone()
        if not game:
            return {}
        cur2 = await db.execute(
            "SELECT * FROM moves WHERE game_id=? ORDER BY move_num", (game_id,)
        )
        moves = await cur2.fetchall()
        return {"game": dict(game), "moves": [dict(m) for m in moves]}


async def get_win_stats() -> dict:
    """Aggregated gameplay statistics for the dashboard."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Total games
        cur = await db.execute("SELECT COUNT(*) as cnt FROM games")
        row = await cur.fetchone()
        total_games = row["cnt"] if row else 0

        # Wins by player
        cur = await db.execute(
            "SELECT winner, COUNT(*) as cnt FROM games GROUP BY winner"
        )
        rows = await cur.fetchall()
        wins = {str(r["winner"]): r["cnt"] for r in rows}

        # Avg nodes by strategy (from AI moves only)
        cur = await db.execute(
            """SELECT strategy, AVG(nodes) as avg_nodes, AVG(exec_time) as avg_time,
                      COUNT(*) as move_count
               FROM moves WHERE nodes > 0
               GROUP BY strategy"""
        )
        rows = await cur.fetchall()
        strategy_stats = [dict(r) for r in rows]

        # Games by mode
        cur = await db.execute(
            "SELECT mode, COUNT(*) as cnt FROM games GROUP BY mode"
        )
        rows = await cur.fetchall()
        mode_counts = {r["mode"]: r["cnt"] for r in rows}

        return {
            "total_games":    total_games,
            "wins_by_player": wins,
            "strategy_stats": strategy_stats,
            "games_by_mode":  mode_counts,
        }
