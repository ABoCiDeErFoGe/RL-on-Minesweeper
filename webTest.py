from playwright.sync_api import sync_playwright
import random
from Game import Game
# randomly click on squares    
def random_click(container):
    for i in range(8):
        # generate random coordinates between 1 - 9 inclusive
        x = random.randint(1, 9)
        y = random.randint(1, 9)
        
        # click on the square by their id
        square_id = f"{y}_{x}"
        print(f"Clicking on square {square_id}")
        container.locator('[id="7_5"]').click()


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    game = Game(page)
    game.new_game("beginner")


    container = page.locator("#game")
    random_click(container)
    for row in game.get_game_state():
        print(row)
    page.pause()
