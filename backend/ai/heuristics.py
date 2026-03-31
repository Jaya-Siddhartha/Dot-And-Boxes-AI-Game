"""
backend/ai/heuristics.py — Dots & Boxes Heuristic Evaluation v6.0

Clean redesign:
  1. No draw_bias — draws emerge naturally from balanced play + small noise
  2. Proper terminal evaluation (win/loss/draw)
  3. Non-terminal evaluation:
     - Score difference (game-phase weighted)
     - Chain/loop parity (Long Chain Rule — THE key Dots & Boxes heuristic)
     - Three-sided box danger assessment
     - Safe move availability
  4. O(n) chain detection via set-based BFS
  5. Move ordering: captures → safe → risky (guaranteed non-empty)
"""

from collections import deque


# ---------------------------------------------------------------------------
# Fast line-count helper
# ---------------------------------------------------------------------------

def count_lines_for_box(state, r: int, c: int) -> int:
    """Count how many sides of box (r,c) are already drawn."""
    return (
        state.horizontal_lines[r][c]
        + state.horizontal_lines[r + 1][c]
        + state.vertical_lines[r][c]
        + state.vertical_lines[r][c + 1]
    )


# ---------------------------------------------------------------------------
# Chain / loop detection — O(n) via set-based BFS
# ---------------------------------------------------------------------------

def _find_chains(state) -> tuple:
    """
    Find chains and loops among 3-sided boxes.

    Returns (chains, loops) where:
      chains = list of chain lengths
      loops  = list of loop lengths
    """
    rows, cols = state.rows, state.cols
    hl, vl, boxes = state.horizontal_lines, state.vertical_lines, state.boxes

    three_sided: set = set()
    for r in range(rows):
        for c in range(cols):
            if boxes[r][c] == 0 and (
                hl[r][c] + hl[r + 1][c] + vl[r][c] + vl[r][c + 1]
            ) == 3:
                three_sided.add((r, c))

    visited: set = set()
    chains = []
    loops  = []

    for start in three_sided:
        if start in visited:
            continue

        comp_list = []
        comp_set  = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            r, c = queue.popleft()
            comp_list.append((r, c))
            comp_set.add((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in three_sided and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        n = len(comp_list)
        if n >= 4:
            is_loop = all(
                sum(
                    1 for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if (r + dr, c + dc) in comp_set
                ) == 2
                for r, c in comp_list
            )
        else:
            is_loop = False

        if is_loop:
            loops.append(n)
        else:
            chains.append(n)

    return chains, loops


# ---------------------------------------------------------------------------
# Main evaluation function — clean design, no draw_bias
# ---------------------------------------------------------------------------

def evaluate_state(state) -> float:
    """
    Evaluate a Dots & Boxes state from Player 1's perspective.

    Terminal states:
      Win  = +1000 + score_diff
      Loss = -1000 + score_diff
      Draw = 0

    Non-terminal evaluation combines:
      - Score difference (phase-weighted — stronger in endgame)
      - Chain parity heuristic (Long Chain Rule)
      - Three-sided box danger assessment
      - Safe move availability bonus
      - Two-sided box penalty (potential future danger)
    """
    p1 = state.scores[1]
    p2 = state.scores[2]
    score_diff   = p1 - p2
    total_boxes  = state.rows * state.cols

    # ── Terminal state ──────────────────────────────────────────────
    if state.is_game_over:
        if p1 > p2:
            return 1000.0 + score_diff
        elif p2 > p1:
            return -1000.0 + score_diff
        else:
            return 0.0  # Draw — perfectly neutral

    # ── Non-terminal ────────────────────────────────────────────────
    rows, cols = state.rows, state.cols
    hl = state.horizontal_lines
    vl = state.vertical_lines
    boxes = state.boxes

    three_sided = 0
    two_sided   = 0
    one_sided   = 0
    zero_sided  = 0

    for r in range(rows):
        for c in range(cols):
            if boxes[r][c] == 0:
                lines = hl[r][c] + hl[r + 1][c] + vl[r][c] + vl[r][c + 1]
                if lines == 3:
                    three_sided += 1
                elif lines == 2:
                    two_sided += 1
                elif lines == 1:
                    one_sided += 1
                else:
                    zero_sided += 1

    chains, loops = _find_chains(state)
    long_chains   = sum(1 for length in chains if length >= 3)
    short_chains  = sum(1 for length in chains if length <= 2)
    control_total = long_chains + len(loops)

    # ── Game phase (0.0 = early, 1.0 = endgame) ────────────────────
    boxes_captured = p1 + p2
    phase = boxes_captured / max(total_boxes, 1)

    # ── 1. Score difference — gets more important in endgame ───────
    score_weight = 8.0 + phase * 8.0   # 8 early, 16 in endgame
    score_component = score_diff * score_weight

    # ── 2. Chain parity (Long Chain Rule) ──────────────────────────
    # In Dots & Boxes, control of long chains determines the winner.
    # Current player benefits if control_total is odd (gets last chain).
    parity_val = 0.0
    if control_total > 0:
        if state.current_player == 1:
            parity_val = 5.0 if (control_total % 2 == 1) else -5.0
        else:
            parity_val = 5.0 if (control_total % 2 == 0) else -5.0

    # Scale parity down slightly in early game (less certain)
    parity_val *= (0.5 + 0.5 * phase)

    # ── 3. Three-sided box danger ──────────────────────────────────
    # Three-sided boxes are captures waiting to happen.
    # If it's the opponent's turn, they capture; if ours, we capture.
    if state.current_player == 1:
        # P1's perspective: 3-sided boxes will be captured by P1
        danger = three_sided * 2.5
    else:
        # Opponent captures them
        danger = -three_sided * 2.5

    # ── 4. Safe move availability ──────────────────────────────────
    # Having safe moves (don't create 3-sided boxes) is valuable
    safe_bonus = 0.0
    if zero_sided + one_sided > 0:
        safe_bonus = min(zero_sided + one_sided * 0.5, 5.0) * 0.4

    # ── 5. Two-sided penalty (future danger) ───────────────────────
    two_sided_penalty = two_sided * 0.3

    result = (
        score_component
        + parity_val
        + danger
        + safe_bonus
        - two_sided_penalty
    )

    return result


# ---------------------------------------------------------------------------
# Move ordering — NEVER returns empty list when valid moves exist
# ---------------------------------------------------------------------------

def get_capturing_moves(state) -> list:
    """
    Return all valid moves ordered by strategic priority:
      1. Capturing moves  (immediately complete a box)
      2. Safe moves       (do NOT create a 3-sided box for opponent)
      3. Risky moves      (gives opponent a 3-sided box)

    Guaranteed non-empty if state.get_valid_moves() is non-empty.
    """
    capturing = []
    safe      = []
    risky     = []

    rows, cols = state.rows, state.cols
    hl, vl, boxes = state.horizontal_lines, state.vertical_lines, state.boxes
    valid_moves = state.get_valid_moves()

    for move in valid_moves:
        m_type, r, c = move['type'], move['r'], move['c']
        would_complete   = False
        would_create_3rd = False

        if m_type == 'h':
            for br, bc in ((r - 1, c), (r, c)):
                if 0 <= br < rows and 0 <= bc < cols and boxes[br][bc] == 0:
                    lines = hl[br][bc] + hl[br + 1][bc] + vl[br][bc] + vl[br][bc + 1]
                    if lines == 3: would_complete   = True
                    if lines == 2: would_create_3rd = True
        else:
            for br, bc in ((r, c - 1), (r, c)):
                if 0 <= br < rows and 0 <= bc < cols and boxes[br][bc] == 0:
                    lines = hl[br][bc] + hl[br + 1][bc] + vl[br][bc] + vl[br][bc + 1]
                    if lines == 3: would_complete   = True
                    if lines == 2: would_create_3rd = True

        if would_complete:
            capturing.append(move)
        elif not would_create_3rd:
            safe.append(move)
        else:
            risky.append(move)

    return capturing + safe + risky


def get_capturing_moves_fast(state) -> list:
    """Same as get_capturing_moves but returns (type, r, c) tuples."""
    capturing = []
    safe      = []
    risky     = []

    rows, cols = state.rows, state.cols
    hl, vl, boxes = state.horizontal_lines, state.vertical_lines, state.boxes
    valid_moves = state.get_valid_moves_fast()

    for m_type, r, c in valid_moves:
        would_complete   = False
        would_create_3rd = False

        if m_type == 'h':
            for br, bc in ((r - 1, c), (r, c)):
                if 0 <= br < rows and 0 <= bc < cols and boxes[br][bc] == 0:
                    lines = hl[br][bc] + hl[br + 1][bc] + vl[br][bc] + vl[br][bc + 1]
                    if lines == 3: would_complete   = True
                    if lines == 2: would_create_3rd = True
        else:
            for br, bc in ((r, c - 1), (r, c)):
                if 0 <= br < rows and 0 <= bc < cols and boxes[br][bc] == 0:
                    lines = hl[br][bc] + hl[br + 1][bc] + vl[br][bc] + vl[br][bc + 1]
                    if lines == 3: would_complete   = True
                    if lines == 2: would_create_3rd = True

        if would_complete:
            capturing.append((m_type, r, c))
        elif not would_create_3rd:
            safe.append((m_type, r, c))
        else:
            risky.append((m_type, r, c))

    return capturing + safe + risky


def is_capturing_move(state, m_type: str, r: int, c: int) -> bool:
    """Check if a move would immediately capture a box."""
    rows, cols = state.rows, state.cols
    hl, vl, boxes = state.horizontal_lines, state.vertical_lines, state.boxes
    if m_type == 'h':
        for br, bc in ((r - 1, c), (r, c)):
            if 0 <= br < rows and 0 <= bc < cols and boxes[br][bc] == 0:
                if hl[br][bc] + hl[br + 1][bc] + vl[br][bc] + vl[br][bc + 1] == 3:
                    return True
    else:
        for br, bc in ((r, c - 1), (r, c)):
            if 0 <= br < rows and 0 <= bc < cols and boxes[br][bc] == 0:
                if hl[br][bc] + hl[br + 1][bc] + vl[br][bc] + vl[br][bc + 1] == 3:
                    return True
    return False
