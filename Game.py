from typing import List, Optional

import numpy as np


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
        if difficulty == "beginner":
            self.w, self.h = 9, 9
        elif difficulty == "intermediate":
            self.w, self.h = 16, 16
        else:  # expert
            self.w, self.h = 30, 16

        self.page.goto(f"https://minesweeperonline.com/#{difficulty}")

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

        
        
                