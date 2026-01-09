import threading
import queue
import tkinter as tk
from playwright.sync_api import sync_playwright
from Game import Game
from config import ROWS, COLUMNS, CELL_SIZE, FLAG_CHAR, BOMB_CHAR, CHECK_CHAR, NUMBER_COLOR_MAP

class App:
    def __init__(self):
        self.cells = []
        self.root = tk.Tk()
        self.root.title("Controller")

        self.initialize_interface()

        # Event used to signal the worker thread to stop Playwright
        self._stop_playwright = threading.Event()
        # Queue for tasks that must run on the Playwright thread
        self._playwright_tasks = queue.Queue()

        # Start the worker thread for Playwright
        self.worker_thread = threading.Thread(target=self.initialize_playwright, daemon=True)
        self.worker_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def initialize_playwright(self):
        """Initialize Playwright in a worker thread."""
        try:
            self.update_status("Status: Initializing Playwright...")
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page()
            # create Game context using the Playwright page
            try:
                self.game = Game(self.page)
                self.game.new_game("beginner")
                # obtain game state in worker thread (uses Playwright page)
                try:
                    state = self.game.get_game_state()
                except Exception:
                    state = None

                # schedule GUI update on main thread to render the grid
                if state is not None:
                    self.root.after(0, lambda s=state: self.build_grid(self.grid_container, game_status=s))
                    self.update_status("Status: Ready (Game created and displayed)")
                else:
                    self.update_status("Status: Ready (Game created, state unavailable)")
            except Exception:
                self.game = None
                self.update_status("Status: Ready (Game creation failed)")
            
        except Exception:
            # If initialization fails, ensure status reflects it and exit thread
            self.update_status("Status: Playwright failed to start")
            return

        # Process tasks using a blocking get(); `None` is the shutdown sentinel.
        try:
            while True:
                task = self._playwright_tasks.get()  # blocking
                # shutdown sentinel
                if task is None:
                    break

                func = task.get('func')
                args = task.get('args', ())
                kwargs = task.get('kwargs', {})
                done = task.get('done')
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    result = e

                if callable(done):
                    # schedule the done callback on the main thread
                    self.root.after(0, (lambda res, cb: cb(res)), result, done)

        finally:
            try:
                self.update_status("Status: Shutting down Playwright...")
                if hasattr(self, 'browser'):
                    self.browser.close()
                if hasattr(self, 'playwright'):
                    self.playwright.stop()
                self.update_status("Status: Playwright stopped")
            except Exception:
                pass

    def update_status(self, message):
        """Update the status label in a thread-safe manner."""
        self.root.after(0, lambda: self.status_label.config(text=message))

    def _place_label(self, cell, text, fg=None, bold=False, bind_right_click=False, coord=None):
        """Create and place a centered Label inside `cell` with font sized to CELL_SIZE.

        If `bind_right_click` is True the label will forward right-clicks to `on_right_click` using `coord`.
        Returns the created Label.
        """
        font_size = max(8, int(CELL_SIZE * 0.6))
        font_spec = (None, font_size, "bold") if bold else (None, font_size)
        lbl = tk.Label(cell, text=text, bg=cell.cget('bg'), fg=fg, font=font_spec)
        lbl.place(relx=0.5, rely=0.5, anchor="center")
        if bind_right_click and coord is not None:
            rr, cc = coord
            lbl.bind("<Button-3>", lambda _e, r=rr, c=cc: self.on_right_click(r, c))
        return lbl

    def on_click(self):
        self.page.locator("text=More information").click()

    def on_close(self):
        # Signal the worker thread to stop Playwright and wait for it to exit
        self.update_status("Status: Exiting...")
        # wake the worker by pushing a sentinel task; worker will exit when it sees None
        try:
            self._playwright_tasks.put(None)
        except Exception:
            # fallback to setting the event if queue isn't available
            if hasattr(self, '_stop_playwright'):
                self._stop_playwright.set()

        # Poll until the worker thread has finished cleanup, then destroy the root
        self.root.after(100, self._check_worker_and_destroy)

    def _check_worker_and_destroy(self):
        if self.worker_thread.is_alive():
            self.root.after(100, self._check_worker_and_destroy)
        else:
            self.root.destroy()

    def run(self):
        self.root.mainloop()
        
    def build_grid(self, container, game_status=None):
        """Creates a ROWS x COLUMNS grid using Frame widgets and renders `game_status`.

        `game_status` is an optional 2D list of integers. If provided, cells will be
        displayed in a minesweeper style according to the integer codes.
        """
        # Clear any existing cells
        for child in container.winfo_children():
            child.destroy()
        self.cells.clear()

        # Outer border on the container
        container.configure(highlightthickness=2, highlightbackground="#444")

        # determine provided status dimensions
        status_rows = len(game_status) if game_status else 0
        status_cols = len(game_status[0]) if status_rows > 0 else 0

        for r in range(ROWS):
            row_cells = []
            for c in range(COLUMNS):
                # decide value for this cell from game_status, if present
                if r < status_rows and c < status_cols:
                    val = game_status[r][c]
                else:
                    val = None

                # background rules
                if val is None:
                    bg = "white"
                elif val == -1 or val == -11:
                    bg = "#D3D3D3"
                elif val == -10:
                    bg = "red"
                else:
                    bg = "#808080"

                cell = tk.Frame(
                    container,
                    width=CELL_SIZE,
                    height=CELL_SIZE,
                    bg=bg,
                    highlightthickness=1,
                    highlightbackground="#999",
                )
                cell.grid(row=r, column=c)
                cell.grid_propagate(False)  # keep the fixed pixel size
                cell.bind("<Button-1>", lambda _e, rr=r, cc=c: self.on_cell_click(rr, cc))
                # right click to place/remove flag
                cell.bind("<Button-3>", lambda _e, rr=r, cc=c: self.on_right_click(rr, cc))

                # compute a font size proportional to the cell size
                font_size = max(8, int(CELL_SIZE * 0.6))

                # render content according to game_status value
                if val is not None:
                    # show numbers 1-8 (0 usually shown as blank)
                    if isinstance(val, int) and val >= 1:
                        fg = NUMBER_COLOR_MAP.get(val, "black")
                        self._place_label(cell, str(val), fg=fg, bold=True)
                    elif val == -11:
                        # flag
                        self._place_label(cell, FLAG_CHAR, bind_right_click=True, coord=(r, c))
                    elif val in (-9, -10):
                        # bomb
                        self._place_label(cell, BOMB_CHAR)
                    elif val == -2:
                        # cross emoji ❌ U+274C
                        self._place_label(cell, "\u274C", fg="#FF0000", bold=True)

                row_cells.append(cell)
            self.cells.append(row_cells)


    def on_cell_click(self, row, col):
        """When a cell is clicked, enqueue a Playwright task to call Game.handle_click
        and then rebuild the grid from the returned state. If any error occurs,
        do nothing.
        """
        if not (0 <= row < ROWS and 0 <= col < COLUMNS):
            return

        # Do nothing if game isn't ready
        if not hasattr(self, 'game') or self.game is None:
            return

        try:
            def _done(result):
                # Runs on the main thread via root.after
                if isinstance(result, Exception):
                    return
                try:
                    if result is not None:
                        self.build_grid(self.grid_container, game_status=result)
                except Exception:
                    pass

            # Game.handle_click expects (x, y) as 1-based coordinates (col+1, row+1)
            task = {'func': self.game.handle_click, 'args': (col + 1, row + 1), 'done': _done}
            self._playwright_tasks.put(task)
        except Exception:
            return

    def on_right_click(self, row, col):
        """Toggle a flag on the clicked cell and notify the Game via handle_right_click.

        The flag is shown/removed immediately in the UI. The Game.handle_right_click
        call is enqueued to the Playwright thread but the grid is not rebuilt.
        """
        if not (0 <= row < ROWS and 0 <= col < COLUMNS):
            return

        # Do nothing if game isn't ready
        if not hasattr(self, 'game') or self.game is None:
            return

        cell = self.cells[row][col]

        # detect existing flag label (exact emoji)
        FLAG = "\U0001F6A9"
        existing_flag = None
        for child in cell.winfo_children():
            try:
                if isinstance(child, tk.Label) and child.cget("text") == FLAG:
                    existing_flag = child
                    break
            except Exception:
                continue

        if existing_flag is not None:
            # remove flag immediately
            try:
                existing_flag.destroy()
            except Exception:
                pass
        else:
            # add a flag label sized to the cell
            font_size = max(8, int(CELL_SIZE * 0.6))
            try:
                lbl = tk.Label(cell, text=FLAG, bg=cell.cget('bg'), font=(None, font_size))
                lbl.place(relx=0.5, rely=0.5, anchor="center")
                # bind right-click on the label so subsequent right-clicks reach the handler
                lbl.bind("<Button-3>", lambda _e, rr=row, cc=col: self.on_right_click(rr, cc))
            except Exception:
                pass

        # enqueue Playwright task to call Game.handle_right_click if available
        try:
            func = getattr(self.game, 'handle_right_click', None)
            if callable(func):
                task = {'func': func, 'args': (col + 1, row + 1), 'done': None}
                self._playwright_tasks.put(task)
        except Exception:
            return

    def _do_new_game(self, difficulty: str):
        """Run on the Playwright thread: start a new game and return its state."""
        if not hasattr(self, 'game') or self.game is None:
            raise RuntimeError("Game not initialized")
        # difficulty is a string: 'beginner', 'intermediate', or '' for expert
        self.game.new_game(difficulty)
        return self.game.get_game_state()

    def _on_difficulty_selected(self, label: str):
        """GUI callback when a difficulty is chosen from the dropdown.

        Maps the displayed label to the difficulty string and enqueues a Playwright task
        that starts a new game and returns the state; when complete the grid is rebuilt.
        """
        diff_value = self._diff_map.get(label, "beginner")
        if not hasattr(self, 'game') or self.game is None:
            return

        def _done(result):
            if isinstance(result, Exception):
                self.update_status("Status: Failed to change difficulty")
                return
            try:
                if result is not None:
                    # debug print of the returned game_status for troubleshooting
                    try:
                        rows = len(result)
                        cols = len(result[0]) if rows > 0 else 0
                    except Exception:
                        rows = cols = 0
                    print(f"DEBUG: game_status ({rows}x{cols}): ")
                    print(result)
                    self.build_grid(self.grid_container, game_status=result)
                    self.update_status(f"Status: Difficulty set to {label}")
            except Exception:
                pass

        task = {'func': self._do_new_game, 'args': (diff_value,), 'done': _done}
        self._playwright_tasks.put(task)

    def initialize_interface(self):
        # Left column: contains grid on top and status label below
        self.left_column = tk.Frame(self.root)
        # add outer padding around the left column (around the grid)
        self.left_column.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # grid frame sits inside the left column
        self.grid_frame = tk.Frame(self.left_column)
        self.grid_frame.pack(side="top", fill="both", expand=True)

        self.grid_container = tk.Frame(self.grid_frame, bg="#f5f5f5")
        self.grid_container.pack()
        self.build_grid(self.grid_container)

        # Right: vertical buttons
        # Right: vertical buttons
        self.sidebar = tk.Frame(self.root)
        self.sidebar.pack(side="right", fill="y", padx=8, pady=8)

        # Difficulty selector
        diff_frame = tk.Frame(self.sidebar)
        diff_frame.pack(fill="x", pady=(0,6))
        tk.Label(diff_frame, text="Difficulty:").pack(side="left")
        # display labels mapped to values passed to Game.new_game
        self._diff_map = {"Beginner": "beginner", "Intermediate": "intermediate", "Expert": "expert"}
        self._diff_var = tk.StringVar(value="Beginner")
        options = list(self._diff_map.keys())
        diff_menu = tk.OptionMenu(diff_frame, self._diff_var, *options, command=self._on_difficulty_selected)
        diff_menu.pack(side="right")

        tk.Button(self.sidebar, text="Start").pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Reset", command=lambda: self.build_grid(self.grid_container)).pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Quit", command=self.on_close).pack(fill="x", pady=2)
        
        # Label indicating the progress, placed below the grid in the left column
        self.status_label = tk.Label(self.left_column, text="Status: Initializing...", anchor="w")
        self.status_label.pack(side="bottom", fill="x", padx=4, pady=(6,0))
        

App().run()