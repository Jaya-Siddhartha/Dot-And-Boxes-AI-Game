from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

from backend.engine.game import GameState


@dataclass
class GameSession:
    session_id: str
    game_state: GameState = field(default_factory=lambda: GameState(rows=4, cols=4))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    websockets: List[WebSocket] = field(default_factory=list)
    aivai_task: Optional[asyncio.Task] = None
    session_meta: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen_at: datetime = field(default_factory=datetime.utcnow)

    def touch(self) -> None:
        self.last_seen_at = datetime.utcnow()

    def reset_meta(self, mode: str, strategy: str, difficulty: str) -> None:
        self.session_meta = {
            "mode": mode,
            "strategy": strategy,
            "difficulty": difficulty,
            "started_at": datetime.utcnow().isoformat(),
            "moves": [],
            "move_num": 0,
            "game_saved": False,
        }
        self.touch()


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, GameSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: Optional[str]) -> GameSession:
        key = (session_id or "default").strip() or "default"
        async with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = GameSession(session_id=key)
                session.reset_meta("hvai", "alphabeta", "hard")
                self._sessions[key] = session
            session.touch()
            return session

    async def register_ws(self, session: GameSession, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket not in session.websockets:
                session.websockets.append(websocket)
            session.touch()

    async def unregister_ws(self, session: GameSession, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in session.websockets:
                session.websockets.remove(websocket)
            session.touch()

    async def broadcast(self, session: GameSession, payload: dict) -> None:
        dead: List[WebSocket] = []
        for ws in list(session.websockets):
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in session.websockets:
                        session.websockets.remove(ws)
        session.touch()
