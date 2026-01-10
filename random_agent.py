"""Random agent utilities for playing Minesweeper via the existing Game/MSEnv.

Usage notes:
- This agent expects a `Game` instance whose Playwright `page` was created
  in the Playwright worker thread (i.e., where the browser was launched).
- Typical usage (from the worker thread where `game` exists):

    from Game import Game, MSEnv
    from random_agent import RandomAgent

    env = MSEnv(game)
    agent = RandomAgent(env)
    result = agent.run_episode(difficulty="beginner", max_steps=500, delay=0.05)
    print(result)

The agent is intentionally simple: it randomly selects unrevealed
cells (preferring unrevealed over already-opened) and mostly left-clicks,
occasionally right-clicking to place flags.
"""

import random
import time
from typing import Dict, Any

from Game import MSEnv


class RandomAgent:
    def __init__(self, env: MSEnv, right_click_prob: float = 0.05) -> None:
        self.env = env
        self.right_click_prob = float(right_click_prob)

    def run_episode(self, difficulty: str = None, max_steps: int = 1000, delay: float = 0.0) -> Dict[str, Any]:
        """Run one episode using purely random actions.

        Returns a dictionary with keys: `steps`, `reward`, `done`, `history`.
        `history` is a list of ((x,y,mode), reward, done) tuples.
        """
        state = self.env.reset(difficulty)
        w = self.env.game.w
        h = self.env.game.h

        steps = 0
        total_reward = 0
        history = []
        done = False

        while steps < max_steps and not done:
            # collect unrevealed cells (value < 0)
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

            x, y = random.choice(unrevealed)
            mode = "right" if random.random() < self.right_click_prob else "left"

            state, reward, done, info = self.env.step((x, y, mode))
            history.append(((x, y, mode), reward, done))
            total_reward += reward
            steps += 1

            if delay:
                time.sleep(delay)

        return {"steps": steps, "reward": total_reward, "done": bool(done), "history": history, "final_state": state}


if __name__ == "__main__":
    print("random_agent.py is a library module. Import RandomAgent and run from the Playwright worker thread where a Game instance exists.")
