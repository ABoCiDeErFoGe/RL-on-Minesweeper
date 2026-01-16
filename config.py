# UI and game constants
ROWS = 16
COLUMNS = 30
CELL_SIZE = 35

# Emoji / symbols
FLAG_CHAR = "\U0001F6A9"
BOMB_CHAR = "💣"
CHECK_CHAR = "\u2714"

# Minesweeper number colors
NUMBER_COLOR_MAP = {
    1: "#0000FF",
    2: "#008000",
    3: "#FF0000",
    4: "#00008B",
    5: "#840000",
    6: "#00FFFF",
    7: "#000000",
    8: "#808080",
}

# Rewards for RL environment (tweakable)
REWARD_BOMB_DEATH = -1
REWARD_STEP = 0
REWARD_WIN = 1

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4
