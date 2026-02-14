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
import importlib
from Game import MSEnv
from utils import extract_random_clicks
from episode_progress_display import EpisodeProgressDisplay
from config import ROWS, COLUMNS, CELL_SIZE, FLAG_CHAR, BOMB_CHAR, NUMBER_COLOR_MAP

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
        # registry for concurrent runs keyed by run_id
        self._runs = {}
        self._run_counter = 0

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

    def _run_random_agent(self, difficulty: str = None, max_steps: int = 500, delay: float = 0.02, right_click_prob: float = 0.05, episodes: int = 1, agent_name: str = None, **kwargs):
        """Wrapper that delegates to the generic `_run_agent` adaptor for RandomAgent.

        Kept as a thin shim so existing enqueue code remains unchanged.
        """
        rw = {"difficulty": difficulty, "max_steps": max_steps, "delay": delay, "episodes": episodes}
        if agent_name:
            rw['agent_name'] = agent_name
        return self._run_agent(
            module_name="random_agent",
            class_name="RandomAgent",
            agent_init_kwargs={"right_click_prob": right_click_prob},
            run_kwargs=rw,
        )

    def _run_baseline_agent(self, difficulty: str = None, max_steps: int = 1000, delay: float = 0.0, episodes: int = 1, agent_name: str = None, **kwargs):
        """Wrapper that delegates to the generic `_run_agent` adaptor for BaselineAgent."""
        rw = {"difficulty": difficulty, "max_steps": max_steps, "delay": delay, "episodes": episodes}
        if agent_name:
            rw['agent_name'] = agent_name
        return self._run_agent(
            module_name="baseline_agent",
            class_name="BaselineAgent",
            agent_init_kwargs={},
            run_kwargs=rw,
        )

    def _run_rl_agent(self, difficulty: str = None, max_steps: int = 100000, delay: float = 0.0, episodes: int = 1, agent_name: str = None, **kwargs):
        """Run the DQN RL agent for `episodes` and report progress via App.update_agent_visualization.

        This runs on the Playwright worker thread and uses `run_num_episodes`'s
        `progress_update` callback to stream per-episode info back to the GUI.
        """
        # use top-level importlib and Game.MSEnv

        # Delegate to the generic agent runner which handles visualization and progress
        rw = {"difficulty": difficulty, "max_steps": max_steps, "delay": delay, "episodes": episodes}
        if agent_name:
            rw['agent_name'] = agent_name
        return self._run_agent(module_name="RL_agent", class_name="DQNAgent", agent_init_kwargs=None, run_kwargs=rw)

    def _run_hybrid_agent(self, difficulty: str = None, max_steps: int = 100000, delay: float = 0.0, episodes: int = 1, agent_name: str = None, **kwargs):
        """Run the Hybrid_Agent which combines the baseline agent with DQN.

        Instantiates `BaselineAgent` and passes it to `Hybrid_Agent`.
        """
        # Delegate to generic runner, passing baseline spec so _run_agent
        # can create and inject the baseline instance for Hybrid_Agent.
        rw = {"difficulty": difficulty, "max_steps": max_steps, "delay": delay, "episodes": episodes}
        if agent_name:
            rw['agent_name'] = agent_name

        init_kw = {'_baseline': {'module': 'baseline_agent', 'class': 'BaselineAgent', 'kwargs': {}}}
        return self._run_agent(module_name="RL_agent", class_name="Hybrid_Agent", agent_init_kwargs=init_kw, run_kwargs=rw)

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

        env = MSEnv(self.game)
        # centralized resolution/instantiation helper handles baseline injection
        AgentClass, agent, resolved_module, resolved_class = self._instantiate_agent(env, module_name=module_name, class_name=class_name, agent_init_kwargs=agent_init_kwargs, worker_callable=None, task_kwargs=run_kwargs, agent_name=None)

        run_kwargs = run_kwargs or {}

        # support running multiple episodes by passing 'episodes' in run_kwargs
        episodes = int(run_kwargs.pop('episodes', 1)) if 'episodes' in run_kwargs else 1
        # optional agent name for per-episode display (may be provided by caller)
        agent_name = run_kwargs.pop('agent_name', class_name)

        if episodes <= 1:
            return agent.run_episode(**run_kwargs)

        # per-run EpisodeProgressDisplay will be created instead of a global visualization

        # create a per-run EpisodeProgressDisplay and composite progress callback
        ep_display = self._create_episode_display(agent_name, run_kwargs.get('difficulty'), episodes)

        def _progress_cb(info):
            try:
                # construct a minimal result dict for _handle_episode_progress
                res = {
                    'steps': info.get('length'),
                    'reward': info.get('reward'),
                    'done': info.get('done', False),
                    'history': [],
                    'random_clicks': extract_random_clicks(info),
                    'win': info.get('win', False),
                }

                # schedule short summary update on GUI thread via a small worker thread
                try:
                    idx = info.get('episode')
                    def _notify_thread(index, result_obj, total, name=agent_name):
                        import threading
                        def _do_notify():
                            try:
                                self.root.after(0, lambda: self._handle_episode_progress(name, index, result_obj, total))
                            except Exception:
                                pass
                        t = threading.Thread(target=_do_notify, daemon=True)
                        t.start()
                    _notify_thread(idx, res, episodes)
                except Exception:
                    pass

                # no global visualization; per-run display handles plotting
            except Exception:
                pass

        # composite progress: update the EpisodeProgressDisplay and the app summary/plots
        def _composite_progress(info):
            try:
                if ep_display is not None:
                    try:
                        ep_display.progress_update(info)
                    except Exception:
                        pass
                _progress_cb(info)
            except Exception:
                pass

        # let the agent run multiple episodes and provide per-episode progress
        results = agent.run_num_episodes(int(episodes), progress_update=_composite_progress, **run_kwargs)

        final_state = None
        if isinstance(results, list) and results:
            final_state = results[-1].get('final_state') if isinstance(results[-1], dict) else None

        return {'episodes': episodes, 'results': results, 'final_state': final_state}

    def _run_agent_with_own_playwright(self, module_name: str = None, worker_callable=None, task_kwargs: dict = None, run_id: str = None, agent_name: str = None):
        """Run an agent in a dedicated thread/process with its own Playwright instance.

        - `module_name` optional: if provided, used to import the agent class by name.
        - `worker_callable` is the original worker function (may be None); we will
          import the agent class based on run kwargs instead.
        - `task_kwargs` contains run parameters including `_run_id` and run args.
        This function starts Playwright, creates a browser/page, constructs a
        Game(page)/MSEnv and instantiates the agent class, then calls
        `agent.run_num_episodes(...)` and schedules UI updates to the main thread.
        """
        tk_master = getattr(self, 'root', None)
        # normalize inputs
        task_kwargs = dict(task_kwargs or {})
        run_args = dict(task_kwargs)
        # extract run parameters
        difficulty = run_args.pop('difficulty', None)
        max_steps = run_args.pop('max_steps', None)
        delay = run_args.pop('delay', None)
        episodes = int(run_args.pop('episodes', 1)) if 'episodes' in run_args else 1
        agent_name = agent_name or run_args.pop('agent_name', None) or module_name or 'agent'

        # create per-run EpisodeProgressDisplay on the GUI thread
        ep_display = self._create_episode_display(agent_name, difficulty, episodes)

        # start Playwright and run the agent
        playwright = None
        browser = None
        page = None
        game = None
        env = None
        agent = None
        results = None
        try:
            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            # create Game context and environment
            game = Game(page)
            env = MSEnv(game)

            # centralize agent resolution and instantiation
            AgentClass, agent, resolved_module, resolved_class = self._instantiate_agent(env, module_name=module_name, class_name=None, agent_init_kwargs=task_kwargs.get('agent_init_kwargs'), worker_callable=worker_callable, task_kwargs=task_kwargs, agent_name=agent_name)
            try:
                print(f"[DEBUG] own_playwright selected AgentClass={resolved_module}.{resolved_class}, baseline_instantiated={hasattr(agent, '__class__')}")
            except Exception:
                pass

            # define progress callback used by agent.run_num_episodes
            def progress_cb(info: dict):
                try:
                    # attach run id so main thread can record per-run results
                    if run_id is not None:
                        info['_run_id'] = run_id
                    # forward to per-run display
                    try:
                        if ep_display is not None:
                            ep_display.progress_update(info)
                    except Exception:
                        pass
                    # schedule short summary update on GUI thread
                    try:
                        # build compact result for summary
                        res = {
                            'steps': info.get('length'),
                            'reward': info.get('reward'),
                            'done': info.get('done', False),
                            'history': [],
                            'random_clicks': extract_random_clicks(info),
                            'win': info.get('win', False),
                        }
                        if run_id is not None:
                            res['_run_id'] = run_id
                        # call _handle_episode_progress on GUI thread
                        try:
                            self.root.after(0, lambda idx=info.get('episode'), r=res: self._handle_episode_progress(agent_name, idx, r, episodes))
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass

            # run the agent episodes (this blocks until finished)
            try:
                results = agent.run_num_episodes(episodes, difficulty=difficulty, max_steps=max_steps, delay=delay, progress_update=progress_cb)
            except Exception as e:
                results = [{'error': str(e)}]

            # prepare final result and schedule final UI update/cleanup on GUI thread
            final_state = None
            if isinstance(results, list) and results:
                try:
                    final_state = results[-1].get('final_state') if isinstance(results[-1], dict) else None
                except Exception:
                    final_state = None
            final_result = {'episodes': episodes, 'results': results, 'final_state': final_state}
            if run_id is not None:
                final_result['_run_id'] = run_id

            try:
                self.root.after(0, lambda: self._handle_agent_result(agent_name, final_result, run_id))
            except Exception:
                pass

        except Exception as e:
            try:
                # report failure
                final_result = {'error': str(e)}
                if run_id is not None:
                    final_result['_run_id'] = run_id
                try:
                    self.root.after(0, lambda: self._handle_agent_result(agent_name, final_result, run_id))
                except Exception:
                    pass
            except Exception:
                pass
        finally:
            try:
                if page is not None:
                    try:
                        page.close()
                    except Exception:
                        pass
                if browser is not None:
                    try:
                        browser.close()
                    except Exception:
                        pass
                if playwright is not None:
                    try:
                        playwright.stop()
                    except Exception:
                        pass
            except Exception:
                pass

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

    def _handle_episode_progress(self, agent_name: str, episode_index: int, result, total_episodes: int):
        """Update the episode info frame for a single finished episode.

        This runs on the GUI thread (it's scheduled via `root.after`). It records
        per-episode summary into `self._current_agent_run['results']` so the
        final aggregated summary can be computed when all episodes finish.
        """
        try:
            # record result in per-run state if run_id provided in result
            run_id = None
            if isinstance(result, dict):
                run_id = result.get('_run_id', None)
            if run_id is None:
                # fall back to global current run for backward compatibility
                run_state = getattr(self, '_current_agent_run', None)
            else:
                run_state = self._runs.get(run_id)

            if isinstance(run_state, dict):
                try:
                    rec = {}
                    if isinstance(result, dict):
                        rec['steps'] = int(result.get('steps', 0))
                        rec['reward'] = float(result.get('reward', 0))
                        rec['done'] = bool(result.get('done', False))
                        rec['random_clicks'] = extract_random_clicks(result)
                    else:
                        rec['steps'] = 0
                        rec['reward'] = 0.0
                        rec['done'] = False
                        rec['random_clicks'] = 0
                    run_state['results'].append(rec)
                except Exception:
                    pass

            # build a concise per-episode display string
            try:
                if isinstance(result, dict):
                    steps = result.get('steps', 0)
                    done = result.get('done', False)
                    rc = extract_random_clicks(result)
                    outcome = 'Win' if done and result.get('reward', 0) > 0 else ('Lose' if done else 'Incomplete')
                    text = f"Episode {episode_index}/{total_episodes}: {outcome}, steps={steps}, random_clicks={rc}"
                else:
                    text = f"Episode {episode_index}/{total_episodes}: completed"
            except Exception:
                text = f"Episode {episode_index}/{total_episodes}: result"

            try:
                self._episode_fields['last'].config(text=text)
            except Exception:
                pass
        except Exception:
            pass

    def _update_episode_summary_from_results(self, results_list):
        """Compute aggregate metrics from a list of per-episode result dicts and update the summary label.

        Designed to be easily extended with new metrics in the future.
        """
        try:
            if not isinstance(results_list, list):
                return
            n = len(results_list)
            if n == 0:
                return
            wins = 0
            total_steps = 0
            total_random = 0
            for r in results_list:
                if not isinstance(r, dict):
                    continue
                if r.get('done', False) and r.get('reward', 0) > 0:
                    wins += 1
                total_steps += int(r.get('steps', 0))
                total_random += extract_random_clicks(r)

            win_rate = wins / n * 100.0
            avg_steps = total_steps / n if n else 0
            avg_random = total_random / n if n else 0

            summary_text = f"Win rate: {win_rate:.1f}%  |  Avg steps: {avg_steps:.1f}  |  Avg random clicks: {avg_random:.1f}"
            try:
                self._episode_fields['summary'].config(text=summary_text)
            except Exception:
                pass
        except Exception:
            pass

    def _handle_agent_result(self, agent_name: str, result, run_id: str = None):
        """Common done-callback for agents: render final state and update status."""
        if isinstance(result, Exception):
            self.update_status(f"Status: {agent_name} failed")
            print(f"{agent_name} error:", result)
            return
        try:
            # Support both single-run result dicts and multi-episode aggregates
            # If run_id provided and registry exists, use stored per-run results for summary
            run_state = None
            if run_id is not None:
                run_state = self._runs.get(run_id)

            if isinstance(result, dict) and 'results' in result:
                # aggregate across episodes
                results = result.get('results', [])
            elif isinstance(run_state, dict):
                results = run_state.get('results', [])
            else:
                results = [result] if isinstance(result, dict) else []

            total_steps = sum(r.get('steps', 0) for r in results if isinstance(r, dict))
            total_reward = sum(r.get('reward', 0) for r in results if isinstance(r, dict))
            any_done = any(r.get('done', False) for r in results if isinstance(r, dict))

            # use last episode's history/state for display
            last = results[-1] if results else result if isinstance(result, dict) else {}
            hist = last.get('history', []) if isinstance(last, dict) else []
            click_map = self._click_map_from_history(hist)
            final_state = result.get('final_state') if isinstance(result, dict) else None
            if final_state is not None:
                self.build_grid(self.grid_container, game_status=final_state, click_history=click_map)

            # update aggregated summary frame
            self._update_episode_summary_from_results(results)
            if run_state is not None:
                # cleanup run registry
                try:
                    del self._runs[run_id]
                except Exception:
                    pass

            if isinstance(result, dict) and 'results' in result:
                self.update_status(f"{agent_name} finished ({result.get('episodes',len(results))}): steps={total_steps}, reward={total_reward}, done={any_done}")
            else:
                steps = result.get('steps', 0) if isinstance(result, dict) else 0
                reward = result.get('reward', 0) if isinstance(result, dict) else 0
                done = result.get('done', False) if isinstance(result, dict) else False
                self.update_status(f"{agent_name} finished: steps={steps}, reward={reward}, done={done}")
        except Exception:
            pass

    def _enqueue_agent(self, worker_func, kwargs: dict, agent_name: str):
        """Put an agent-running task on the Playwright worker and attach common done handler."""
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return
        # create a per-run registry entry and spawn a thread to enqueue the worker task
        total_eps = int(kwargs.get('episodes', 1)) if isinstance(kwargs, dict) else 1
        try:
            self._run_counter += 1
            run_id = f"run_{self._run_counter}"
            self._runs[run_id] = {'agent_name': agent_name, 'total_episodes': total_eps, 'results': []}
            # clear episode info fields (shared short-summary)
            try:
                self._episode_fields['last'].config(text="")
                self._episode_fields['summary'].config(text="")
            except Exception:
                pass
        except Exception:
            run_id = None

        # prepare the worker task; include run_id in kwargs so worker-side progress callbacks
        # can include it in per-episode info objects
        task_kwargs = dict(kwargs or {})
        if run_id is not None:
            task_kwargs['_run_id'] = run_id

        task = {'func': worker_func, 'kwargs': task_kwargs, 'done': (lambda res, name=agent_name, rid=run_id: self._handle_agent_result(name, res, rid)),}

        # Start a dedicated run thread which creates its own Playwright instance
        def _run_thread():
            try:
                # import and run agent inside its own Playwright/browser/page
                self._run_agent_with_own_playwright(module_name=worker_func.__name__ if isinstance(worker_func, type) else None,
                                                     worker_callable=worker_func,
                                                     task_kwargs=task_kwargs,
                                                     run_id=run_id,
                                                     agent_name=agent_name)
            except Exception:
                self.update_status(f"Status: Failed to start {agent_name}")

        t = threading.Thread(target=_run_thread, daemon=True)
        t.start()

    def start_random_agent(self, max_steps: int = 500, delay: float = 0.02, right_click_prob: float = 0.05):
        """Enqueue a task to start the random agent on the Playwright thread and display result in the GUI."""
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return

        # determine difficulty from UI selection and enqueue agent
        diff_value = self._diff_map.get(self._diff_var.get(), "beginner")
        # determine episodes: custom entry takes precedence
        
        episodes = self._get_episodes_from_ui()
        
        self._enqueue_agent(
            self._run_random_agent,
            {'difficulty': diff_value, 'max_steps': max_steps, 'delay': delay, 'right_click_prob': right_click_prob, 'episodes': episodes, 'agent_name': 'Random agent'},
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
        
        episodes = self._get_episodes_from_ui()
        
        self._enqueue_agent(
            self._run_baseline_agent,
            {'difficulty': diff_value, 'max_steps': max_steps, 'delay': delay, 'episodes': episodes, 'agent_name': 'Baseline agent'},
            'Baseline agent',
        )

    def start_rl_agent(self, max_steps: int = 100000, delay: float = 0.0):
        """Enqueue the RL agent training run on the Playwright worker and show visualization."""
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return

        # per-run EpisodeProgressDisplay will be created by the runner

        # determine difficulty and episodes (custom entry takes precedence)
        diff_value = self._diff_map.get(self._diff_var.get(), "beginner")
        episodes = self._get_episodes_from_ui()

        self._enqueue_agent(
            self._run_rl_agent,
            {'difficulty': diff_value, 'max_steps': max_steps, 'delay': delay, 'episodes': episodes, 'agent_name': 'RL agent'},
            'RL agent',
        )

    def start_hybrid_agent(self, max_steps: int = 100000, delay: float = 0.0):
        """Enqueue the Hybrid agent (baseline + DQN) to run on the Playwright worker."""
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return

        # per-run EpisodeProgressDisplay will be created by the runner

        diff_value = self._diff_map.get(self._diff_var.get(), "beginner")
        episodes = self._get_episodes_from_ui()

        self._enqueue_agent(
            self._run_hybrid_agent,
            {'difficulty': diff_value, 'max_steps': max_steps, 'delay': delay, 'episodes': episodes, 'agent_name': 'Hybrid agent'},
            'Hybrid agent'
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
        tk.Button(self.sidebar, text="Train RL Agent", command=lambda: self.start_rl_agent()).pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Hybrid Agent", command=lambda: self.start_hybrid_agent()).pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Reset", command=lambda: self.build_grid(self.grid_container)).pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Quit", command=self.on_close).pack(fill="x", pady=2)
        
        # Label indicating the progress, placed below the grid in the left column
        # Episode info frame: shows per-episode and aggregated statistics
        self.episode_info_frame = tk.Frame(self.left_column, relief='groove', bd=1)
        tk.Label(self.episode_info_frame, text="Episode Info:").pack(anchor='w', padx=4)
        self._episode_fields = {}
        self._episode_fields['last'] = tk.Label(self.episode_info_frame, text="", anchor='w')
        self._episode_fields['last'].pack(fill='x', padx=6)
        self._episode_fields['summary'] = tk.Label(self.episode_info_frame, text="", anchor='w')
        self._episode_fields['summary'].pack(fill='x', padx=6, pady=(0,4))
        self.episode_info_frame.pack(side="bottom", fill="x", padx=4, pady=(6,0))

        self.status_label = tk.Label(self.left_column, text="Status: Initializing...", anchor="w")
        self.status_label.pack(side="bottom", fill="x", padx=4, pady=(6,0))

    def _get_episodes_from_ui(self):
        """Read episode count from UI (custom entry or dropdown). Returns int."""
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

        return max(1, int(episodes))

    def _create_episode_display(self, agent_name: str, difficulty: str, episodes: int):
        """Create an EpisodeProgressDisplay on the Tk main thread and populate static labels.

        Returns the created display or None.
        """
        holder = {'disp': None}
        def _create_on_main():
            try:
                holder['disp'] = EpisodeProgressDisplay(title=f"{agent_name} Progress", master=self.root, agent_name=agent_name, difficulty=difficulty, total_episodes=episodes)
            except Exception:
                holder['disp'] = None

        try:
            self.root.after(0, _create_on_main)
            import time
            deadline = time.time() + 5.0
            while time.time() < deadline and holder['disp'] is None:
                time.sleep(0.05)
            disp = holder.get('disp')
            if disp is not None:
                try:
                    disp.set_agent(agent_name)
                except Exception:
                    pass
                try:
                    if difficulty is not None:
                        disp.set_difficulty(difficulty)
                except Exception:
                    pass
                try:
                    disp.set_run_progress(0, episodes)
                except Exception:
                    pass
            return disp
        except Exception:
            return None

    def _instantiate_agent(self, env, module_name: str = None, class_name: str = None, agent_init_kwargs: dict = None, worker_callable=None, task_kwargs: dict = None, agent_name: str = None):
        """Resolve and instantiate an agent class given various hints.

        Returns a tuple (AgentClass, agent_instance, resolved_module_name, resolved_class_name).
        """
        task_kwargs = dict(task_kwargs or {})
        # determine target module/class
        target_module = None
        target_class_name = class_name

        if module_name and module_name in ('RL_agent', 'baseline_agent', 'random_agent'):
            target_module = importlib.import_module(module_name)
            if module_name == 'RL_agent':
                # prefer Hybrid if agent_name or task hints include it
                if (agent_name and 'hybrid' in agent_name.lower()) or ('hybrid' in (task_kwargs.get('agent_name') or '').lower()):
                    target_class_name = 'Hybrid_Agent'
                else:
                    target_class_name = target_class_name or 'DQNAgent'
            elif module_name == 'baseline_agent':
                target_class_name = target_class_name or 'BaselineAgent'
            elif module_name == 'random_agent':
                target_class_name = target_class_name or 'RandomAgent'

        # fallback to worker_callable name mapping
        if target_module is None and worker_callable is not None:
            name = getattr(worker_callable, '__name__', '') or ''
            nlow = name.lower()
            if 'hybrid' in nlow:
                target_module = importlib.import_module('RL_agent')
                target_class_name = 'Hybrid_Agent'
            elif 'rl' in nlow:
                target_module = importlib.import_module('RL_agent')
                target_class_name = 'DQNAgent'
            elif 'baseline' in nlow:
                target_module = importlib.import_module('baseline_agent')
                target_class_name = 'BaselineAgent'
            elif 'random' in nlow:
                target_module = importlib.import_module('random_agent')
                target_class_name = 'RandomAgent'

        # last resort
        if target_module is None:
            target_module = importlib.import_module('RL_agent')
            target_class_name = target_class_name or 'DQNAgent'

        AgentClass = getattr(target_module, target_class_name)

        # baseline handling
        init_kwargs = dict(agent_init_kwargs) if isinstance(agent_init_kwargs, dict) else {}
        baseline_instance = None
        if '_baseline' in init_kwargs:
            bspec = init_kwargs.pop('_baseline')
            if isinstance(bspec, dict):
                try:
                    bm = importlib.import_module(bspec.get('module'))
                    BClass = getattr(bm, bspec.get('class'))
                    bkwargs = bspec.get('kwargs', {}) or {}
                    baseline_instance = BClass(env, **bkwargs) if bkwargs else BClass(env)
                except Exception:
                    baseline_instance = None
            else:
                baseline_instance = bspec

        # if class needs baseline and none provided, try to instantiate default
        try:
            if baseline_instance is None and target_class_name == 'Hybrid_Agent':
                try:
                    bm = importlib.import_module('baseline_agent')
                    BClass = getattr(bm, 'BaselineAgent')
                    baseline_instance = BClass(env)
                except Exception:
                    baseline_instance = None
        except Exception:
            baseline_instance = None

        # instantiate agent
        if baseline_instance is not None:
            agent = AgentClass(env, baseline_instance)
        else:
            try:
                init_kw = task_kwargs.get('agent_init_kwargs') or init_kwargs or {}
                agent = AgentClass(env, **init_kw) if init_kw else AgentClass(env)
            except Exception:
                agent = AgentClass(env)

        return AgentClass, agent, target_module.__name__, target_class_name

    # Global visualization removed; per-run EpisodeProgressDisplay windows are used instead.
        
App().run()