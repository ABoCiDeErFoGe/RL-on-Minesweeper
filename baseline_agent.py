"""Baseline rule-based Minesweeper agent.

This module implements a simple deterministic agent based on local inference
rules used in human Minesweeper play.

State encoding used (same as `Game.get_game_state()`):
- negative values indicate unrevealed/flag states:
    - -1 : unrevealed / blank
    - -2 : flagged (flag placed)
    - -9/-10/-11 : various bomb states (revealed/ death / flagged bomb)
- 0-8 : revealed numbers indicating adjacent bombs

Rules implemented (per revealed numbered cell):
- Let `blanks` be surrounding squares with value < 0 (unrevealed)
- Let `flags` be the count of surrounding cells considered flagged (-2 or -11)
- If len(blanks) == (number - flags): place flags on all blanks (right-click)
- If number == flags: safe to left-click (open) all surrounding unrevealed cells

If no deterministic action is found this step, the agent falls back to a
single random unrevealed left-click.

The function `run_episode` returns a dict with keys `steps`, `reward`, `done`,
`history`, and `final_state` to allow the GUI to render the final board and the
sequence of actions taken.
"""

import random
import time
from typing import Dict, Any, List, Tuple

from Game import MSEnv


def _neighbors(r: int, c: int, w: int, h: int) -> List[Tuple[int, int]]:
    # Return all valid 8-neighbor coordinates around (r, c).
    # Coordinates are 0-based row/col indexes; `w` is width (#columns), `h` is height (#rows).
    coords = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                coords.append((nr, nc))
    return coords


class BaselineAgent:
    def __init__(self, env: MSEnv) -> None:
        self.env = env

    def select_action(self, state: List[List[int]]) -> Tuple[List[Tuple[int, int, str]] , List[Tuple[int, int, str]]]:
        # Scan the board for a cell where we can place a flag based on local rules.
        w = self.env.game.w
        h = self.env.game.h
        
         # Collect candidate actions for this step. Actions are tuples (x, y, mode)
        # where x,y are 1-based coordinates and mode is 'left' or 'right'
        actions = []

        # Scan every revealed numbered cell and check local constraints
        for r in range(h):
            for c in range(w):
                try:
                    val = state[r][c]
                except Exception:
                    val = -1
                # only interested in revealed numbered cells (0..8)
                if not isinstance(val, int) or val <= 0:
                    continue
                
                neigh = _neighbors(r, c, w, h)
                blanks = []
                flags = 0
                # Count blanks (unrevealed) and flags around this cell
                for (nr, nc) in neigh:
                    try:
                        v = state[nr][nc]
                    except Exception:
                        v = -1
                    if v == -1:
                        blanks.append((nr, nc))
                    if v in (-2, -11):
                        flags += 1

                req = val - flags

                # If the remaining required flags equal the number of blank
                # neighbors, we can safely place flags on every blank.
                if req > 0 and len(blanks) == req:
                    for (nr, nc) in blanks:
                        actions.append((nc + 1, nr + 1, 'right'))  # convert to 1-based
                if val == flags and len(blanks) > 0:
                    for (nr, nc) in blanks:
                        actions.append((nc + 1, nr + 1, 'left'))  # convert to 1-based

            # Deduplicate actions while preserving order
            seen = set()
            dedup_flag_actions = []
            dedup_click_actions = []
            for (x, y, m) in actions:
                key = (x, y, m)
                if key in seen:
                    continue
                seen.add(key)
                if m == 'right':
                    dedup_flag_actions.append((x, y, m))
                else:
                    dedup_click_actions.append((x, y, m))

        return (dedup_flag_actions, dedup_click_actions)
    
    def run_episode(self, difficulty: str = None, max_steps: int = 1000, delay: float = 0.0) -> Dict[str, Any]:
        # Start a new game (or reuse default difficulty) and obtain initial board.
        # `state` is a 2D list of ints as returned by Game.get_game_state().
        state = self.env.reset(difficulty)
        w = self.env.game.w
        h = self.env.game.h

        steps = 0
        total_reward = 0
        history = []
        done = False
        random_click = 0

        # Main loop: attempt to apply deterministic rules until the game ends
        while steps < max_steps and not done:
            # Collect candidate actions for this step. Actions are tuples (x, y, mode)
            # where x,y are 1-based coordinates and mode is 'left' or 'right'.
            (flag_actions, click_actions) = self.select_action(state)
            
            dedup_actions = flag_actions + click_actions
            # If no deterministic actions found, pick a random unrevealed cell
            if not dedup_actions:
                unrevealed = []
                for r in range(h):
                    for c in range(w):
                        try:
                            v = state[r][c]
                        except Exception:
                            v = -1
                        if v < 0:
                            unrevealed.append((c + 1, r + 1))
                if not unrevealed:
                    # nothing left to click
                    break
                choice = random.choice(unrevealed)
                dedup_actions = [(choice[0], choice[1], 'left')]
                random_click += 1

            # Execute selected actions sequentially; after each action update
            # the local `state` and record the transition in `history`.
            for (x, y, mode) in dedup_actions:
                try:
                    state, reward, done, info = self.env.step((x, y, mode))
                except Exception as e:
                    # surface execution errors in the returned result
                    return {"error": str(e)}

                history.append(((x, y, mode), reward, done))
                total_reward += reward
                steps += 1

                if delay:
                    time.sleep(delay)

                # stop early if episode ended
                if done or steps >= max_steps:
                    break

        # Return a summary including the final board state for rendering
        return {"steps": steps, "reward": total_reward, "done": bool(done), "history": history, "final_state": state, "random_clicks": random_click, "win": info.get('win', False)}


if __name__ == "__main__":
    print("baseline_agent module: import BaselineAgent and run from the Playwright worker thread where a Game instance exists.")
