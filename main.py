import threading
import queue
import tkinter as tk
from playwright.sync_api import sync_playwright
from Game import Game

ROWS = 16
COLUMNS = 30
CELL_SIZE = 35

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
            self.page.goto("https://minesweeperonline.com")
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
                elif val == -1:
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

                # compute a font size proportional to the cell size
                font_size = max(8, int(CELL_SIZE * 0.6))

                # render content according to game_status value
                if val is not None:
                    # show numbers 1-8 (0 usually shown as blank)
                    if isinstance(val, int) and val >= 1:
                        # Minesweeper color mapping for numbers
                        color_map = {
                            1: "#0000FF",  # blue
                            2: "#008000",  # green
                            3: "#FF0000",  # red
                            4: "#00008B",  # dark blue
                            5: "#840000",  # maroon
                            6: "#00FFFF",  # cyan/teal
                            7: "#000000",  # black
                            8: "#808080",  # gray
                        }
                        fg = color_map.get(val, "black")
                        lbl = tk.Label(cell, text=str(val), bg=bg, fg=fg, font=(None, font_size, "bold"))
                        lbl.place(relx=0.5, rely=0.5, anchor="center")
                    elif val == -2:
                        # triangular flag emoji U+1F6A9
                        lbl = tk.Label(cell, text="\U0001F6A9", bg=bg, font=(None, font_size))
                        lbl.place(relx=0.5, rely=0.5, anchor="center")
                    elif val in (-9, -10):
                        # bomb emoji
                        lbl = tk.Label(cell, text="💣", bg=bg, font=(None, font_size))
                        lbl.place(relx=0.5, rely=0.5, anchor="center")
                    elif val == -11:
                        # checkmark U+2714 (green)
                        lbl = tk.Label(cell, text="\u2714", bg=bg, fg="#008200", font=(None, font_size, "bold"))
                        lbl.place(relx=0.5, rely=0.5, anchor="center")

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

        tk.Button(self.sidebar, text="Start").pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Reset", command=lambda: self.build_grid(self.grid_container)).pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Quit", command=self.root.destroy).pack(fill="x", pady=2)
        
        # Label indicating the progress, placed below the grid in the left column
        self.status_label = tk.Label(self.left_column, text="Status: Initializing...", anchor="w")
        self.status_label.pack(side="bottom", fill="x", padx=4, pady=(6,0))
        

App().run()