# Dots & Boxes AI Battle Suite

Full-stack Dots & Boxes game built with FastAPI, vanilla JavaScript, SQLite, and multiple AI strategies.

This project supports:
- Human vs Human
- Human vs AI
- AI vs AI
- Algorithm comparison and game history

It has been updated to isolate live gameplay per browser session, so multiple users can use the app at the same time on one deployed server process without sharing the same board state.

## What Changed

Recent updates in this codebase:
- Fixed broken game-over popup behavior
- Reduced end-of-game UI latency by pushing result events before persistence work
- Fixed strategy selection so low difficulty no longer silently downgrades to random play
- Made frontend depth selection map more predictably to backend AI search
- Added per-session live game isolation for concurrent users
- Scoped WebSocket updates per user session instead of broadcasting one shared board to everyone

## Current Architecture

### Frontend
- `frontend/index.html`
- `frontend/script.js`
- `frontend/styles.css`

The frontend is a vanilla JS single-page app. Each browser stores a stable `session_id` in `localStorage` and sends it with:
- all REST API requests
- the WebSocket connection

That session id is what keeps one user's game separate from another user's game.

### Backend
- `backend/app.py`
- `backend/api/routes.py`
- `backend/session_manager.py`
- `backend/engine/game.py`
- `backend/ai/strategies.py`
- `backend/ai/heuristics.py`
- `backend/learning/qlearning.py`
- `backend/database/db.py`

The backend is FastAPI-based and uses:
- in-memory per-session live game state
- SQLite for persistent history and stats
- a shared Q-learning data file for learning progress

## Multi-User Behavior

### Safe Right Now

The app now supports multiple simultaneous users on the same running server process because:
- each session has its own `GameState`
- each session has its own async lock
- each session has its own AI-vs-AI task
- each session has its own WebSocket subscriber list

This means:
- User A can start a `3x3` game
- User B can start a `5x5` game
- User A's moves do not affect User B's board
- real-time state updates only go to the matching session

### Important Limitation

Live session state is still in memory.

That means this version is suitable for:
- localhost use
- demo deployments
- single-server deployments
- one-process hosting

It is not yet sufficient for:
- horizontal scaling across multiple app instances
- load-balanced production clusters
- crash-safe live session recovery after process restart

If the server restarts, live games in memory are lost. Saved history in SQLite remains.

## AI Strategies

### Minimax
- full search without pruning
- useful for learning and comparison
- slower than Alpha-Beta

### Alpha-Beta
- pruned search
- faster than Minimax
- better default for interactive use

### Adaptive AI
- Alpha-Beta plus Q-learning feedback
- uses learned values to influence move selection

### Difficulty and Depth

The code now respects the selected strategy more faithfully.

Important behavior:
- chosen strategy remains the chosen strategy
- low difficulty no longer replaces Minimax or Alpha-Beta with random behavior
- UI depth is now used more directly for search depth in interactive play

## Features

- Per-session live gameplay
- Real-time WebSocket board updates
- Game-over modal with winner/draw handling
- Algorithm comparison
- Game history and replay
- AI metrics
- Q-learning persistence
- AI vs AI simulation

## Project Structure

```text
Ai_project/
├─ app.py
├─ requirements.txt
├─ README.md
├─ backend/
│  ├─ app.py
│  ├─ session_manager.py
│  ├─ api/
│  │  └─ routes.py
│  ├─ ai/
│  │  ├─ balance.py
│  │  ├─ heuristics.py
│  │  └─ strategies.py
│  ├─ database/
│  │  └─ db.py
│  ├─ engine/
│  │  └─ game.py
│  └─ learning/
│     └─ qlearning.py
├─ frontend/
│  ├─ index.html
│  ├─ script.js
│  ├─ styles.css
│  └─ assets/
└─ data/
   ├─ games.db
   └─ learning_data.json
```

## Requirements

- Python 3.10+
- pip
- modern browser

## Installation

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Locally

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000`
- or `http://localhost:8000`

## Development Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Use only one worker for this version.

Do not run:

```powershell
uvicorn backend.app:app --workers 2
```

Why:
- live sessions are in process memory
- multiple workers would each have different in-memory session stores
- users would get inconsistent live behavior

## API Overview

### REST

- `GET /api/state`
- `POST /api/start-game`
- `POST /api/reset`
- `POST /api/move`
- `POST /api/ai-move`
- `POST /api/ai-vs-ai`
- `GET /api/suggest`
- `POST /api/comparison`
- `GET /api/history`
- `GET /api/history/{id}`
- `GET /api/stats`
- `GET /api/learning-stats`
- `GET /api/balance-stats`

### WebSocket

- `WS /api/ws`

### Session Requirement

Live gameplay endpoints use a `session_id` query parameter.

Example:

```text
/api/start-game?session_id=sess-123
```

The frontend already does this automatically. If you build another client, you must send a stable `session_id` yourself.

## Persistence

### Stored Persistently
- completed game history in `data/games.db`
- Q-learning data in `data/learning_data.json`

### Not Stored Persistently
- active live boards
- active AI-vs-AI runtime tasks
- active WebSocket subscriptions

## Concurrency Model

The backend now uses:
- one `GameSession` object per session id
- one `asyncio.Lock` per session
- one optional AI-vs-AI task per session
- one WebSocket group per session

This prevents:
- all users seeing the same board
- one user's move changing another user's game
- one AI-vs-AI simulation overwriting another user's game

## Deployment Guidance

### Works Well For
- localhost
- single VPS
- Render or Railway style single-instance deploys
- internal demo environments

### Recommended Minimum Deployment Stack
- FastAPI app
- one Uvicorn worker
- reverse proxy like Nginx or Caddy
- process manager or container restart policy

### Recommended Command

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1
```

## For Real Production Scale

If you expect many real users, multiple devices, or multiple server instances, add these before calling it production-grade:

- Redis for shared session storage
- Redis pub/sub or a message broker for cross-instance WebSocket fanout
- Postgres instead of local SQLite
- background job queue for long AI work
- session expiry and cleanup
- rate limiting
- structured logging
- monitoring and alerts
- reverse proxy timeouts and health checks
- load testing
- crash recovery strategy

Without Redis or shared state, two separate app instances will not share live sessions.

## Troubleshooting

### Localhost keeps loading forever

Likely causes:
- old Python/Uvicorn process still running
- more than one process bound to port `8000`
- a stale hung server instance

Check port usage:

```powershell
cmd /c netstat -ano | findstr :8000
```

Kill a stuck process:

```powershell
taskkill /PID <pid> /F
```

Start the app again:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### Popup not showing correctly

This was addressed in the current version. The UI now:
- reacts from final state updates
- avoids duplicate game-over modals
- pushes result events earlier from the backend

### Multiple users affecting the same game

This should now be fixed if:
- they are using different browsers or browser sessions
- the frontend is sending a stable `session_id`

If you build another client manually and omit `session_id`, sessions can collide.

### App crashes under multi-instance deployment

This version is not yet designed for shared live state across multiple server instances.
Use Redis-backed session storage first.

## Validation Notes

The updated code was validated for:
- backend Python compilation
- frontend JavaScript syntax
- session isolation between two different `session_id` values

## Future Improvements

- Redis-backed live sessions
- Postgres migration
- real user authentication
- multiplayer room invites
- spectating mode
- stronger AI benchmarking tools
- cleanup job for abandoned sessions
- deployment Dockerfiles and Compose setup
- CI tests for concurrent session behavior

## License

MIT
