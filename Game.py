import numpy as np

# Bind with the minesweeperonline.com website, need to have 
# a playwright page object to work with

class Game:
    
    w = -1
    h = -1
    
    # take a playwright page object as input
    def __init__(self, page):
        self.page = page
        
    # difficulty: "beginner": 9x9, "intermediate": 16x16, "" for expert: 16x30    
    def new_game(self, difficulty):
        if difficulty == "beginner":
            self.w, self.h = 9, 9
        elif difficulty == "intermediate":
            self.w, self.h = 16, 16
        else:  # expert
            self.w, self.h = 30, 16
        self.page.goto("https://minesweeperonline.com/#" + difficulty)
        
        
    def handle_click(self, x, y):
        
        if (x < 0 or x > self.w or y < 0 or y > self.h):
            raise ValueError("Coordinates out of bounds")
        
        # click on the square by their id
        square_id = f"{y}_{x}"
        print(f"Clicking on square {square_id}")
        self.page.locator("#game").locator(f'[id="{square_id}"]').click()
        return self.get_game_state()
        
    # convert the game to 2d array representation 
       
    def get_game_state(self):
        
        if self.w == -1 or self.h == -1:
            raise ValueError("Game not initialized. Please start a new game first.")
        
        # initialize empty board
        board = np.zeros((self.h, self.w), dtype=int)
        
        # locate the div with id "game"
        game_container = self.page.locator("#game")
        
        # only look at the squares and ignore the rest
        elements = game_container.evaluate("""
            (root) => {
                return [...root.querySelectorAll("*")]
                    .filter(el =>
                        [...el.classList].some(cls => cls.startsWith("square"))
                    )
                    .map(el => ({
                        id: el.id || null,
                        classes: [...el.classList]
                    }));
            }
            """)

        for el in elements:
            row, col = map(int, el['id'].split('_'))
            print(row, col)
            row -= 1  # adjust for 0-based indexing
            col -= 1  # adjust for 0-based indexing
            # if out of bound: continue
            if row < 0 or row >= self.h or col < 0 or col >= self.w:
                continue
            board[row, col] = self.square_encode(el['classes'][1])
        return board.tolist()
    
    # -1 for blank(unrevealed), -2 for flagged, 0-8 for revealed squares
    # -9 for bombrevealed, -10 for bombdeath, -11 for bombflagged
    def square_encode(self, string:str) -> int:
        if 'blank' == string:
            return -1
        elif 'bombflagged' == string:
            return -11
        elif 'bombdeath' == string:
            return -10
        elif 'bombrevealed' == string:
            return -9
        elif 'flagged' == string:
            return -2
        else:
            if string.startswith('open'):
                num = int(string.replace('open', ''))
                return num
            else:
                return -1

        
        
                