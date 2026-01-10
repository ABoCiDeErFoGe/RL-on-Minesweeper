from typing import List, Optional

import numpy as np
from config import REWARD_BOMB_DEATH, REWARD_STEP, REWARD_WIN


class Game:
    """Wrapper around a Playwright page that exposes Minesweeper actions

    Coordinates used by this class are 1-based to match the site's element ids.
    """

    def __init__(self, page) -> None:
        self.page = page
        self.w: int = -1
        self.h: int = -1

    def new_game(self, difficulty: str) -> None:
        """Start a new game on the website and set board dimensions."""
        # normalize difficulty, accept empty string as 'expert'
        diff = (difficulty or "").lower()
        if diff == "beginner":
            self.w, self.h = 9, 9
        elif diff == "intermediate":
            self.w, self.h = 16, 16
        else:  # expert
            diff = "expert"
            self.w, self.h = 30, 16

        url = f"https://minesweeperonline.com/#{diff}"
        print(f"Navigating to {url}")
        # navigate and wait for the game container to be available
        print("Calling goto...")
        self.page.goto(url)
        try:
            print("Waiting for network to be idle...")
            self.page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            print("Warning: wait_for_load_state timed out or failed")
        # Some pages require an explicit reload to pick up the new board state
        try:
            print("Reloading page to ensure fresh state...")
            self.page.reload()
        except Exception:
            print("Warning: reload failed or unavailable")

        try:
            print("Waiting for #game selector...")
            self.page.wait_for_selector("#game", timeout=10000)
        except Exception:
            print("Error: #game selector not found after navigation/reload")
            # if the selector doesn't appear, raise to signal failure
            raise

    def _check_bounds(self, x: int, y: int) -> None:
        if x < 1 or x > self.w or y < 1 or y > self.h:
            raise ValueError("Coordinates out of bounds")

    def handle_click(self, x: int, y: int) -> List[List[int]]:
        """Left-click the square at 1-based coordinates (x,y) and return the updated game state."""
        self._check_bounds(x, y)
        square_id = f"{y}_{x}"
        print(f"Clicking on square {square_id}")
        self.page.locator("#game").locator(f'[id="{square_id}"]').click()
        return self.get_game_state()

    def handle_right_click(self, x: int, y: int) -> None:
        """Right-click (flag/unflag) the square at 1-based coordinates (x,y)."""
        self._check_bounds(x, y)
        square_id = f"{y}_{x}"
        print(f"Right clicking on square {square_id}")
        self.page.locator("#game").locator(f'[id="{square_id}"]').click(button="right")

    def get_game_state(self) -> List[List[int]]:
        """Read the current board from the page and return a 2D integer list.

        Encoding:
        -1: unrevealed (blank)
        -2: flagged
         0-8: revealed numbers
        -9: bomb revealed
       -10: bomb death
       -11: bomb flagged
        """
        if self.w == -1 or self.h == -1:
            raise ValueError("Game not initialized. Please start a new game first.")

        board = np.zeros((self.h, self.w), dtype=int)
        game_container = self.page.locator("#game")

        elements = game_container.evaluate("""
            (root) => {
                return [...root.querySelectorAll("*")]
                    .filter(el => [...el.classList].some(cls => cls.startsWith("square")))
                    .map(el => ({ id: el.id || null, classes: [...el.classList] }));
            }
            """)

        for el in elements:
            try:
                row, col = map(int, el["id"].split("_"))
            except Exception:
                continue
            # convert to 0-based
            row -= 1
            col -= 1
            if row < 0 or row >= self.h or col < 0 or col >= self.w:
                continue

            # Find a meaningful class for the square
            classes: List[str] = el.get("classes", [])
            cls = next((c for c in classes if c != "square" and not c.startswith("square")), None)
            if cls is None:
                board[row, col] = -1
            else:
                board[row, col] = self.square_encode(cls)

        return board.tolist()

    def square_encode(self, cls: str) -> int:
        cls = cls or ""
        if cls == "blank":
            return -1
        if cls == "bombflagged":
            return -11
        if cls == "bombdeath":
            return -10
        if cls == "bombrevealed":
            return -9
        # flagged or miss-flagged
        if cls in ("flagged", "missflagged", "bombmisflagged"):
            return -2
        if cls.startswith("open"):
            try:
                num = int(cls.replace("open", ""))
                return num
            except Exception:
                return -1
        return -1

    def get_result(self) -> int:
        """Return game result based on the face element's class:

        - 1: win (face has class 'facewin')
        - 0: still playing (face has class 'facesmile' or unknown)
        - -1: lost (face has class 'facedead')
        """
        try:
            face_cls = self.page.locator("#face").get_attribute("class")
        except Exception:
            return 0

        if not face_cls:
            return 0
        face_cls = face_cls.lower()
        if "facewin" in face_cls:
            return 1
        if "facedead" in face_cls:
            return -1
        # treat smile/other as ongoing
        return 0


class MSEnv:
    """A lightweight RL-style environment wrapper around `Game`.

    Action format: a tuple (x, y, mode) where x and y are 1-based coordinates
    and mode is either 'left' or 'right'. Example: (3, 5, 'left')

    The environment provides:
    - reset(difficulty=None) -> state
    - step(action) -> (state, reward, done, info)
    """

    def __init__(self, game: Game, difficulty: str = "beginner") -> None:
        self.game = game
        self.default_difficulty = difficulty

    def reset(self, difficulty: Optional[str] = None) -> List[List[int]]:
        """Start a new game and return the initial state."""
        diff = difficulty if difficulty is not None else self.default_difficulty
        self.game.new_game(diff)
        state = self.game.get_game_state()
        return state

    def step(self, action) -> (List[List[int]], int, bool, dict):
        """Apply an action and return (state, reward, done, info).

        Reward/done policy (simple):
        - If a left click results in a bomb death (-10) anywhere: done=True, reward=-1
        - Otherwise reward=0 and done=False
        - Right-clicks (flags) return reward 0 and do not mark done here
        """
        # validate action
        try:
            x, y, mode = action
        except Exception:
            raise ValueError("Action must be a tuple (x, y, mode)")

        if mode == "left":
            state = self.game.handle_click(x, y)
        elif mode == "right":
            # perform right click (flag/unflag)
            self.game.handle_right_click(x, y)
            state = self.game.get_game_state()
        else:
            raise ValueError("mode must be 'left' or 'right'")

        # prefer checking the page face for definitive result (win/loss)
        result = self.game.get_result()
        if result == 1:
            done = True
            reward = REWARD_WIN
        elif result == -1:
            done = True
            reward = REWARD_BOMB_DEATH
        else:
            # fallback: inspect state for bomb death squares
            done = any(-10 in row for row in state)
            reward = REWARD_BOMB_DEATH if done else REWARD_STEP
        info = {}
        return state, reward, done, info

        
        
                