"""Minesweeper GUI controller.

This module runs a Tkinter GUI that displays a Minesweeper board and drives
interactions on the live website via a Playwright instance. Playwright runs in
a dedicated worker thread; tasks that need browser access are passed to the
worker using `self._playwright_tasks` (a `queue.Queue`). The worker executes
callables and schedules GUI updates back onto the main thread using
`root.after(0, ...)` so all GUI ops remain thread-safe.

The GUI can run small agents (random, baseline) in the Playwright worker and
then render the final state plus a click-order footer for analysis.
"""

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
        """Initialize Playwright in a worker thread.

        The worker thread owns Playwright and all browser/page interactions. We
        instantiate Playwright here and then run a blocking loop to process
        tasks submitted via `self._playwright_tasks`.
        """
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
        # Each task is a dict with 'func', optional 'args'/'kwargs', and
        # a 'done' callback that will be invoked on the GUI thread via
        # `root.after(0, ...)` with the result of `func`.
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
                    # All browser interactions happen here inside the worker
                    result = func(*args, **kwargs)
                except Exception as e:
                    # Propagate exception object to the GUI callback for reporting
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
        # Always schedule label updates on the main (GUI) thread.
        self.root.after(0, lambda: self.status_label.config(text=message))

    def _place_label(self, cell, text, fg=None, bold=False, bind_right_click=False, coord=None):
        """Create and place a centered Label inside `cell` with font sized to CELL_SIZE.

        If `bind_right_click` is True the label will forward right-clicks to `on_right_click` using `coord`.
        Returns the created Label.
        """
        # compute font size proportional to cell size; guarantees minimum readability
        font_size = max(8, int(CELL_SIZE * 0.55))
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
        
    def build_grid(self, container, game_status=None, click_history=None):
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

        # add click-order footers if provided: `click_history` is a mapping
        # keyed by (row, col) in 0-based coordinates with value = click order.
        # Footers are small labels placed near the bottom-center of each cell.
        try:
            if click_history:
                footer_font_size = max(6, int(CELL_SIZE * 0.3))
                for (rr, cc), order in click_history.items():
                    if 0 <= rr < len(self.cells) and 0 <= cc < len(self.cells[rr]):
                        cell = self.cells[rr][cc]
                        try:
                            lbl = tk.Label(cell, text=str(order), bg="white", fg='black', font=(None, footer_font_size))
                            lbl.place(relx=0.5, rely=0.8, anchor='center')
                        except Exception:
                            # ignore rendering errors for individual footers
                            pass
        except Exception:
            # ignore errors building footers
            pass


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

    def _run_random_agent(self, difficulty: str = None, max_steps: int = 500, delay: float = 0.02, right_click_prob: float = 0.05, episodes: int = 1):
        """Wrapper that delegates to the generic `_run_agent` adaptor for RandomAgent.

        Kept as a thin shim so existing enqueue code remains unchanged.
        """
        return self._run_agent(
            module_name="random_agent",
            class_name="RandomAgent",
            agent_init_kwargs={"right_click_prob": right_click_prob},
            run_kwargs={"difficulty": difficulty, "max_steps": max_steps, "delay": delay, "episodes": episodes},
        )

    def _run_baseline_agent(self, difficulty: str = None, max_steps: int = 1000, delay: float = 0.0, episodes: int = 1):
        """Wrapper that delegates to the generic `_run_agent` adaptor for BaselineAgent."""
        return self._run_agent(
            module_name="baseline_agent",
            class_name="BaselineAgent",
            agent_init_kwargs={},
            run_kwargs={"difficulty": difficulty, "max_steps": max_steps, "delay": delay, "episodes": episodes},
        )

    def _run_agent(self, module_name: str, class_name: str, agent_init_kwargs: dict = None, run_kwargs: dict = None):
        """Generic worker-side adaptor to run any agent class.

        - Dynamically imports `module_name` and looks up `class_name`.
        - Instantiates the agent with the environment `MSEnv(self.game)` plus
          any `agent_init_kwargs`.
        - Calls `agent.run_episode(**run_kwargs)` and returns its result.

        This keeps worker logic for different agents centralized and avoids
        duplicating import/instantiation/run code.
        """
        if not hasattr(self, 'game') or self.game is None:
            raise RuntimeError("Game not initialized")

        import importlib
        from Game import MSEnv

        mod = importlib.import_module(module_name)
        AgentClass = getattr(mod, class_name)

        env = MSEnv(self.game)
        init_kwargs = agent_init_kwargs or {}
        agent = AgentClass(env, **init_kwargs) if init_kwargs else AgentClass(env)

        run_kwargs = run_kwargs or {}

        # support running multiple episodes by passing 'episodes' in run_kwargs
        episodes = int(run_kwargs.pop('episodes', 1)) if 'episodes' in run_kwargs else 1
        if episodes <= 1:
            return agent.run_episode(**run_kwargs)

        # run multiple episodes sequentially and collect results
        results = []
        final_state = None
        for i in range(episodes):
            res = agent.run_episode(**run_kwargs)
            results.append(res)
            # track final_state from last episode if present
            final_state = res.get('final_state') if isinstance(res, dict) else None

        return {'episodes': episodes, 'results': results, 'final_state': final_state}

    def _click_map_from_history(self, history):
        """Convert agent `history` into a mapping {(r,c): order} for footers.

        `history` is a list of ((x,y,mode), reward, done) tuples where x,y are
        1-based coordinates. Returns dict keyed by 0-based (row,col).
        """
        click_map = {}
        for idx, entry in enumerate(history, start=1):
            try:
                action = entry[0]
                x, y = action[0], action[1]
                rr = y - 1
                cc = x - 1
                # only keep first occurrence (order)
                if (rr, cc) not in click_map:
                    click_map[(rr, cc)] = idx
            except Exception:
                continue
        return click_map

    def _handle_agent_result(self, agent_name: str, result):
        """Common done-callback for agents: render final state and update status."""
        if isinstance(result, Exception):
            self.update_status(f"Status: {agent_name} failed")
            print(f"{agent_name} error:", result)
            return
        try:
            # Support both single-run result dicts and multi-episode aggregates
            if isinstance(result, dict) and 'results' in result:
                # aggregate across episodes
                results = result.get('results', [])
                total_steps = sum(r.get('steps', 0) for r in results if isinstance(r, dict))
                total_reward = sum(r.get('reward', 0) for r in results if isinstance(r, dict))
                any_done = any(r.get('done', False) for r in results if isinstance(r, dict))
                # use last episode's history/state for display
                last = results[-1] if results else {}
                hist = last.get('history', []) if isinstance(last, dict) else []
                click_map = self._click_map_from_history(hist)
                final_state = result.get('final_state')
                if final_state is not None:
                    self.build_grid(self.grid_container, game_status=final_state, click_history=click_map)
                self.update_status(f"{agent_name} finished ({result.get('episodes',len(results))}): steps={total_steps}, reward={total_reward}, done={any_done}")
            else:
                steps = result.get('steps', 0)
                reward = result.get('reward', 0)
                done = result.get('done', False)
                hist = result.get('history', [])
                click_map = self._click_map_from_history(hist)
                final_state = result.get('final_state')
                if final_state is not None:
                    self.build_grid(self.grid_container, game_status=final_state, click_history=click_map)
                self.update_status(f"{agent_name} finished: steps={steps}, reward={reward}, done={done}")
        except Exception:
            pass

    def _enqueue_agent(self, worker_func, kwargs: dict, agent_name: str):
        """Put an agent-running task on the Playwright worker and attach common done handler."""
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return

        task = {
            'func': worker_func,
            'kwargs': kwargs,
            'done': (lambda res, name=agent_name: self._handle_agent_result(name, res)),
        }
        try:
            self._playwright_tasks.put(task)
            self.update_status(f"Status: {agent_name} started...")
        except Exception:
            self.update_status(f"Status: Failed to start {agent_name}")

    def start_random_agent(self, max_steps: int = 500, delay: float = 0.02, right_click_prob: float = 0.05):
        """Enqueue a task to start the random agent on the Playwright thread and display result in the GUI."""
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return

        # determine difficulty from UI selection and enqueue agent
        diff_value = self._diff_map.get(self._diff_var.get(), "beginner")
        # determine episodes: custom entry takes precedence
        episodes = None
        try:
            custom = self._episodes_entry.get().strip()
            if custom:
                episodes = int(custom)
        except Exception:
            episodes = None

        if episodes is None:
            try:
                episodes = int(self._episodes_var.get())
            except Exception:
                episodes = 1

        # validate
        if episodes <= 0:
            self.update_status("Status: Invalid episodes number")
            return

        self._enqueue_agent(
            self._run_random_agent,
            {'difficulty': diff_value, 'max_steps': max_steps, 'delay': delay, 'right_click_prob': right_click_prob, 'episodes': episodes},
            'Random agent',
        )

    def start_baseline_agent(self, max_steps: int = 1000, delay: float = 0.0):
        """Enqueue a task to start the BaselineAgent on the Playwright thread and display result in the GUI."""
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return

        # determine difficulty from UI selection and enqueue agent
        diff_value = self._diff_map.get(self._diff_var.get(), "beginner")
        # determine episodes
        episodes = None
        try:
            custom = self._episodes_entry.get().strip()
            if custom:
                episodes = int(custom)
        except Exception:
            episodes = None

        if episodes is None:
            try:
                episodes = int(self._episodes_var.get())
            except Exception:
                episodes = 1

        if episodes <= 0:
            self.update_status("Status: Invalid episodes number")
            return

        self._enqueue_agent(
            self._run_baseline_agent,
            {'difficulty': diff_value, 'max_steps': max_steps, 'delay': delay, 'episodes': episodes},
            'Baseline agent',
        )

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

        # Episodes selector (predefined + custom entry)
        episodes_frame = tk.Frame(self.sidebar)
        episodes_frame.pack(fill="x", pady=(6,6))
        tk.Label(episodes_frame, text="Episodes:").pack(side="left")
        # predefined options
        self._episodes_var = tk.StringVar(value="1")
        episode_options = ["1", "5", "10", "50"]
        episodes_menu = tk.OptionMenu(episodes_frame, self._episodes_var, *episode_options)
        episodes_menu.pack(side="right")
        # custom entry below
        custom_frame = tk.Frame(self.sidebar)
        custom_frame.pack(fill="x")
        tk.Label(custom_frame, text="Custom:").pack(side="left")
        self._episodes_entry = tk.Entry(custom_frame, width=6)
        self._episodes_entry.pack(side="right")

        tk.Button(self.sidebar, text="Start").pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Random Agent", command=lambda: self.start_random_agent()).pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Baseline Agent", command=lambda: self.start_baseline_agent()).pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Reset", command=lambda: self.build_grid(self.grid_container)).pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Quit", command=self.on_close).pack(fill="x", pady=2)
        
        # Label indicating the progress, placed below the grid in the left column
        self.status_label = tk.Label(self.left_column, text="Status: Initializing...", anchor="w")
        self.status_label.pack(side="bottom", fill="x", padx=4, pady=(6,0))
        
App().run()