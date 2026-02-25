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
from tkinter import messagebox, filedialog
from playwright.sync_api import sync_playwright
from Game import Game
import importlib
from Game import MSEnv
from utils import extract_random_clicks
from episode_progress_display import EpisodeProgressDisplay
from config import ROWS, COLUMNS, CELL_SIZE, FLAG_CHAR, BOMB_CHAR, NUMBER_COLOR_MAP
from config import BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR
import tkinter.messagebox as messagebox

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
        def _do_update():
            try:
                if hasattr(self, 'status_label') and self.status_label is not None:
                    try:
                        self.status_label.config(text=message)
                        return
                    except Exception:
                        pass
                # fallback: update window title so user still sees a status
                try:
                    self.root.title(message)
                except Exception:
                    # final fallback: print to stdout
                    try:
                        print(message)
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            self.root.after(0, _do_update)
        except Exception:
            # if scheduling fails, try immediate update
            _do_update()

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

    def _create_agent_runner(self, agent_type: str, difficulty: str = None, max_steps: int = None, 
                            delay: float = 0.0, episodes: int = 1, agent_name: str = None, **kwargs):
        """Unified agent runner factory for all agent types."""
        agent_configs = {
            'random': {
                'module': 'random_agent',
                'class': 'RandomAgent',
                'max_steps': 500,
                'init_kwargs': {'right_click_prob': kwargs.get('right_click_prob', 0.05)}
            },
            'baseline': {
                'module': 'baseline_agent',
                'class': 'BaselineAgent',
                'max_steps': 1000,
                'init_kwargs': {}
            },
            'rl': {
                'module': 'RL_agent',
                'class': 'DQNAgent',
                'max_steps': 100000,
                'init_kwargs': self._build_agent_init_kwargs(None)
            },
            'hybrid': {
                'module': 'RL_agent',
                'class': 'Hybrid_Agent',
                'max_steps': 100000,
                'init_kwargs': self._build_agent_init_kwargs({
                    '_baseline': {'module': 'baseline_agent', 'class': 'BaselineAgent', 'kwargs': {}}
                })
            }
        }
        
        config = agent_configs.get(agent_type)
        if not config:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        run_kwargs = {
            "difficulty": difficulty,
            "max_steps": max_steps or config['max_steps'],
            "delay": delay,
            "episodes": episodes
        }
        if agent_name:
            run_kwargs['agent_name'] = agent_name
            
        return self._run_agent(
            module_name=config['module'],
            class_name=config['class'],
            agent_init_kwargs=config['init_kwargs'],
            run_kwargs=run_kwargs
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

        # Update per-run display with agent hyperparameters if available.
        # Ensure both the agent and the display exist, and schedule the
        # update on the Tk main thread to avoid threading/timing issues.
        try:
            self._schedule_ep_display_hyperparams(ep_display, agent)
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

            # If loading from checkpoint, initialize game with checkpoint's difficulty first
            # This ensures the network dimensions match the checkpoint
            checkpoint_path = task_kwargs.get('checkpoint_path')
            if checkpoint_path:
                try:
                    import torch
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    saved_difficulty = checkpoint.get('difficulty', 'beginner')
                    # Initialize game with correct difficulty before creating agent
                    game.new_game(saved_difficulty)
                except Exception as e:
                    print(f"Warning: Failed to pre-initialize game from checkpoint difficulty: {e}")

            # Extract agent type from task_kwargs and determine module/class
            agent_type = task_kwargs.get('agent_type')
            target_module_name = module_name
            target_class_name = None
            
            if agent_type:
                # Map agent type to module/class using same logic as _create_agent_runner
                agent_configs = {
                    'random': {'module': 'random_agent', 'class': 'RandomAgent'},
                    'baseline': {'module': 'baseline_agent', 'class': 'BaselineAgent'},
                    'rl': {'module': 'RL_agent', 'class': 'DQNAgent'},
                    'hybrid': {'module': 'RL_agent', 'class': 'Hybrid_Agent'}
                }
                config = agent_configs.get(agent_type)
                if config:
                    target_module_name = config['module']
                    target_class_name = config['class']

            # centralize agent resolution and instantiation
            AgentClass, agent, resolved_module, resolved_class = self._instantiate_agent(
                env, 
                module_name=target_module_name, 
                class_name=target_class_name, 
                agent_init_kwargs=task_kwargs.get('agent_init_kwargs'), 
                worker_callable=worker_callable, 
                task_kwargs=task_kwargs, 
                agent_name=agent_name
            )
            try:
                pass
            except Exception:
                pass

            # If we created a per-run display, populate its hyperparameters
            try:
                self._schedule_ep_display_hyperparams(ep_display, agent)
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
            # Cleanup Playwright resources
            for resource in [('page', page), ('browser', browser), ('playwright', playwright)]:
                if resource[1]:
                    try:
                        getattr(resource[1], 'close' if resource[0] != 'playwright' else 'stop')()
                    except Exception:
                        pass

    def _click_map_from_history(self, history):
        """Convert agent history to a mapping {(row,col): order} for footers."""
        click_map = {}
        for idx, entry in enumerate(history, start=1):
            if len(entry) > 0 and len(entry[0]) >= 2:
                x, y = entry[0][0], entry[0][1]
                click_map.setdefault((y - 1, x - 1), idx)  # Convert to 0-based
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
        
        # Get results list
        run_state = self._runs.get(run_id) if run_id else None
        if isinstance(result, dict) and 'results' in result:
            results = result.get('results', [])
        elif run_state:
            results = run_state.get('results', [])
        else:
            results = [result] if isinstance(result, dict) else []

        # Calculate aggregate stats
        total_steps = sum(r.get('steps', 0) for r in results if isinstance(r, dict))
        total_reward = sum(r.get('reward', 0) for r in results if isinstance(r, dict))
        any_done = any(r.get('done', False) for r in results if isinstance(r, dict))

        # Render final state from last episode
        last = results[-1] if results else (result if isinstance(result, dict) else {})
        if isinstance(last, dict):
            hist = last.get('history', [])
            click_map = self._click_map_from_history(hist)
            final_state = result.get('final_state') if isinstance(result, dict) else None
            if final_state:
                self.build_grid(self.grid_container, game_status=final_state, click_history=click_map)

        # Update summary
        self._update_episode_summary_from_results(results)
        
        # Cleanup
        if run_id and run_id in self._runs:
            del self._runs[run_id]

        # Status message
        if isinstance(result, dict) and 'results' in result:
            episodes = result.get('episodes', len(results))
            self.update_status(f"{agent_name} finished ({episodes}): steps={total_steps}, reward={total_reward}, done={any_done}")
        else:
            steps = result.get('steps', 0) if isinstance(result, dict) else 0
            reward = result.get('reward', 0) if isinstance(result, dict) else 0
            done = result.get('done', False) if isinstance(result, dict) else False
            self.update_status(f"{agent_name} finished: steps={steps}, reward={reward}, done={done}")

    def _enqueue_agent(self, worker_func, kwargs: dict, agent_name: str):
        """Enqueue an agent task in its own thread with dedicated Playwright instance."""
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return
        
        # Create run registry entry
        self._run_counter += 1
        run_id = f"run_{self._run_counter}"
        total_eps = int(kwargs.get('episodes', 1))
        self._runs[run_id] = {'agent_name': agent_name, 'total_episodes': total_eps, 'results': []}
        
        # Clear episode info UI
        if hasattr(self, '_episode_fields'):
            self._episode_fields.get('last', tk.Label()).config(text="")
            self._episode_fields.get('summary', tk.Label()).config(text="")

        # Prepare task kwargs
        task_kwargs = dict(kwargs)
        task_kwargs['_run_id'] = run_id
        
        # Merge hyperparams
        merged = self._build_agent_init_kwargs(task_kwargs.get('agent_init_kwargs'))
        if merged:
            task_kwargs['agent_init_kwargs'] = merged

        # Start dedicated thread
        def _run_thread():
            try:
                self._run_agent_with_own_playwright(
                    module_name=getattr(worker_func, '__name__', None),
                    worker_callable=worker_func,
                    task_kwargs=task_kwargs,
                    run_id=run_id,
                    agent_name=agent_name
                )
            except Exception as e:
                self.update_status(f"Status: Failed to start {agent_name}: {e}")

        threading.Thread(target=_run_thread, daemon=True).start()

    def start_random_agent(self, max_steps: int = 500, delay: float = 0.02, right_click_prob: float = 0.05):
        """Start random agent."""
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return
        
        diff_value = self._diff_map.get(self._diff_var.get(), "beginner")
        episodes = self._get_episodes_from_ui()
        
        self._enqueue_agent(
            self._create_agent_runner,
            {
                'agent_type': 'random',
                'difficulty': diff_value,
                'max_steps': max_steps,
                'delay': delay,
                'episodes': episodes,
                'agent_name': 'Random agent',
                'right_click_prob': right_click_prob
            },
            'Random agent'
        )

    def start_baseline_agent(self, max_steps: int = 1000, delay: float = 0.0):
        """Start baseline agent."""
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return
        
        diff_value = self._diff_map.get(self._diff_var.get(), "beginner")
        episodes = self._get_episodes_from_ui()
        
        self._enqueue_agent(
            self._create_agent_runner,
            {
                'agent_type': 'baseline',
                'difficulty': diff_value,
                'max_steps': max_steps,
                'delay': delay,
                'episodes': episodes,
                'agent_name': 'Baseline agent'
            },
            'Baseline agent'
        )

    def start_rl_agent(self, max_steps: int = 100000, delay: float = 0.0):
        """Start RL agent."""
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return
        
        diff_value = self._diff_map.get(self._diff_var.get(), "beginner")
        episodes = self._get_episodes_from_ui()

        self._enqueue_agent(
            self._create_agent_runner,
            {
                'agent_type': 'rl',
                'difficulty': diff_value,
                'max_steps': max_steps,
                'delay': delay,
                'episodes': episodes,
                'agent_name': 'RL agent'
            },
            'RL agent'
        )

    def start_hybrid_agent(self, max_steps: int = 100000, delay: float = 0.0):
        """Start hybrid agent."""
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return
        
        diff_value = self._diff_map.get(self._diff_var.get(), "beginner")
        episodes = self._get_episodes_from_ui()

        self._enqueue_agent(
            self._create_agent_runner,
            {
                'agent_type': 'hybrid',
                'difficulty': diff_value,
                'max_steps': max_steps,
                'delay': delay,
                'episodes': episodes,
                'agent_name': 'Hybrid agent'
            },
            'Hybrid agent'
        )

    def start_saved_agent(self):
        """Start a previously saved agent from checkpoint."""
        if not self._checkpoint_loaded or not self._loaded_checkpoint_path:
            messagebox.showwarning("No Checkpoint", "Please load a checkpoint first")
            return
        
        if not hasattr(self, 'game') or self.game is None:
            self.update_status("Status: Game not ready")
            return
        
        episodes = self._get_episodes_from_ui()
        
        # Get difficulty from checkpoint (already set in UI)
        diff_value = self._diff_map.get(self._diff_var.get(), "beginner")
        
        # Get agent type from loaded checkpoint
        agent_type_map = {
            'DQNAgent': 'rl',
            'Hybrid_Agent': 'hybrid',
            'BaselineAgent': 'baseline',
            'RandomAgent': 'random'
        }
        agent_type = agent_type_map.get(self._loaded_agent_class, 'rl')
        
        # Queue the agent with checkpoint path (ONLY for saved agent)
        self._enqueue_agent(
            self._create_agent_runner,
            {
                'agent_type': agent_type,
                'difficulty': diff_value,
                'max_steps': 100000,
                'delay': 0.0,
                'episodes': episodes,
                'agent_name': f'Saved {agent_type.capitalize()} Agent',
                'checkpoint_path': self._loaded_checkpoint_path  # Only passed for saved agent
            },
            f'Saved {agent_type.capitalize()} Agent'
        )

    def _load_agent_checkpoint(self):
        """Open file dialog to select and load a checkpoint file."""
        import torch
        import os
        
        try:
            # Open file dialog in current working directory
            filepath = filedialog.askopenfilename(
                title="Select checkpoint file",
                filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")],
                initialdir=os.getcwd()
            )
            
            if not filepath:
                return  # User cancelled
            
            # Load checkpoint to verify it's valid and extract info
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Verify required keys exist
            if 'agent_class' not in checkpoint:
                messagebox.showerror("Invalid Checkpoint", "Checkpoint missing 'agent_class' field")
                return
            
            # Store checkpoint info
            self._loaded_checkpoint_path = filepath
            self._loaded_agent_class = checkpoint.get('agent_class')
            self._checkpoint_loaded = True
            
            # Enable the Saved Agent button
            if hasattr(self, '_saved_agent_btn'):
                self._saved_agent_btn.config(state='normal')
            
            # Display notification with filename
            filename = os.path.basename(filepath)
            self._checkpoint_label.config(text=f"Loaded:\n{filename}", fg='green')
            self.update_status(f"Status: Checkpoint loaded ({self._loaded_agent_class})")
            
        except Exception as e:
            messagebox.showerror("Error Loading Checkpoint", f"Failed to load checkpoint: {str(e)}")
            self._checkpoint_label.config(text="", fg='red')

    def _apply_checkpoint_settings(self):
        """Apply checkpoint settings (difficulty and hyperparameters) to the UI."""
        import torch
        
        if not self._checkpoint_loaded or not self._loaded_checkpoint_path:
            return
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self._loaded_checkpoint_path, map_location='cpu')
            
            # Extract settings from checkpoint
            saved_difficulty = checkpoint.get('difficulty', 'beginner')
            saved_config = checkpoint.get('config', {})
            
            # Update difficulty selector
            try:
                # Find the key that matches the saved difficulty
                for label, value in self._diff_map.items():
                    if value == saved_difficulty:
                        self._diff_var.set(label)
                        break
            except Exception:
                pass
            
            # Update hyperparameters from checkpoint config
            try:
                for param_name, param_value in saved_config.items():
                    if param_name in self._hp_vars:
                        self._hp_vars[param_name].set(str(param_value))
            except Exception:
                pass
            
            # Update status
            agent_class_name = checkpoint.get('agent_class', 'Unknown')
            self.update_status(f"Status: Loaded {agent_class_name} settings (difficulty: {saved_difficulty})")
            
        except Exception as e:
            pass

    def _run_saved_agent(self):
        """Load saved agent checkpoint and run it with checkpoint settings."""
        import torch
        from Game import MSEnv
        
        if not self._checkpoint_loaded or not self._loaded_checkpoint_path:
            messagebox.showwarning("No Checkpoint Loaded", "Please load a checkpoint first")
            return
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self._loaded_checkpoint_path, map_location='cpu')
            
            # Extract settings from checkpoint
            saved_difficulty = checkpoint.get('difficulty', 'beginner')
            saved_config = checkpoint.get('config', {})
            agent_class_name = checkpoint.get('agent_class')
            
            # Update difficulty selector and disable it
            try:
                # Find the key that matches the saved difficulty
                for label, value in self._diff_map.items():
                    if value == saved_difficulty:
                        self._diff_var.set(label)
                        break
                # Disable difficulty selector since checkpoint specifies it
                self._diff_menu.config(state='disabled')
            except Exception:
                pass
            
            # Update hyperparameters from checkpoint config
            try:
                for param_name, param_value in saved_config.items():
                    if param_name in self._hp_vars:
                        self._hp_vars[param_name].set(str(param_value))
            except Exception:
                pass
            
            # Select the appropriate agent type based on agent_class
            agent_type_map = {
                'DQNAgent': 'rl',
                'Hybrid_Agent': 'hybrid',
                'BaselineAgent': 'baseline',
                'RandomAgent': 'random'
            }
            
            agent_type = agent_type_map.get(agent_class_name, 'rl')
            
            # Select and run the agent with the checkpoint path
            if not hasattr(self, 'game') or self.game is None:
                self.update_status("Status: Game not ready")
                return
            
            # Get episodes from UI
            episodes = self._get_episodes_from_ui()
            
            # Prepare run parameters with checkpoint path
            diff_value = saved_difficulty
            task_kwargs = {
                'agent_type': agent_type,
                'difficulty': diff_value,
                'max_steps': 100000,
                'delay': 0.0,
                'episodes': episodes,
                'agent_name': f'Saved {agent_type.capitalize()} Agent',
                'checkpoint_path': self._loaded_checkpoint_path  # Pass checkpoint to agent
            }
            
            # Would run the agent here, but user said no need to implement this yet
            self.update_status(f"Status: Ready to run saved {agent_class_name}")
            
        except Exception as e:
            messagebox.showerror("Error Running Saved Agent", f"Failed to run saved agent: {str(e)}")

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
            if result is not None:
                self.build_grid(self.grid_container, game_status=result)
                self.update_status(f"Status: Difficulty set to {label}")

        task = {
            'func': lambda d: self.game.new_game(d) or self.game.get_game_state(),
            'args': (diff_value,),
            'done': _done
        }
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

        # Episode info frame (shows per-episode and aggregated statistics)
        # Created here so it's present on startup and only once.
        self.episode_info_frame = tk.Frame(self.left_column, relief='groove', bd=1)
        tk.Label(self.episode_info_frame, text="Episode Info:").pack(anchor='w', padx=4)
        self._episode_fields = {}
        self._episode_fields['last'] = tk.Label(self.episode_info_frame, text="", anchor='w')
        self._episode_fields['last'].pack(fill='x', padx=6)
        self._episode_fields['summary'] = tk.Label(self.episode_info_frame, text="", anchor='w')
        self._episode_fields['summary'].pack(fill='x', padx=6, pady=(0,4))
        self.episode_info_frame.pack(side="bottom", fill="x", padx=4, pady=(6,0))

        # Status label placed below the grid in the left column
        self.status_label = tk.Label(self.left_column, text="Status: Initializing...", anchor="w")
        self.status_label.pack(side="bottom", fill="x", padx=4, pady=(6,0))

        # Right: vertical buttons
        # Right: vertical buttons
        self.sidebar = tk.Frame(self.root)
        self.sidebar.pack(side="right", fill="y", padx=8, pady=8)

        # Agent selection buttons (2x2 grid) and Start control
        self._agent_buttons = {}
        self.selected_agent = None
        
        # Checkpoint tracking variables
        self._loaded_checkpoint_path = None
        self._checkpoint_loaded = False
        self._loaded_agent_class = None

        agent_frame = tk.Frame(self.sidebar)
        agent_frame.pack(fill='x', pady=(4,6))
        btn = tk.Button(agent_frame, text="Random", width=12)
        btn.config(command=lambda k='random', b=btn: self._select_agent(k, b))
        btn.grid(row=0, column=0, padx=4, pady=2)
        self._agent_buttons['random'] = btn

        btn = tk.Button(agent_frame, text="Pure rule", width=12)
        btn.config(command=lambda k='baseline', b=btn: self._select_agent(k, b))
        btn.grid(row=0, column=1, padx=4, pady=2)
        self._agent_buttons['baseline'] = btn

        # second row
        btn = tk.Button(agent_frame, text="Pure CNN", width=12)
        btn.config(command=lambda k='rl', b=btn: self._select_agent(k, b))
        btn.grid(row=1, column=0, padx=4, pady=2)
        self._agent_buttons['rl'] = btn

        btn = tk.Button(agent_frame, text="Hybrid", width=12)
        btn.config(command=lambda k='hybrid', b=btn: self._select_agent(k, b))
        btn.grid(row=1, column=1, padx=4, pady=2)
        self._agent_buttons['hybrid'] = btn

        # third row - checkpoint buttons
        btn = tk.Button(agent_frame, text="Load Agent", width=12, command=self._load_agent_checkpoint)
        btn.grid(row=2, column=0, padx=4, pady=2)
        self._load_agent_btn = btn
        
        btn = tk.Button(agent_frame, text="Saved Agent", width=12)
        btn.config(command=lambda k='saved', b=btn: self._select_agent(k, b), state='disabled')
        btn.grid(row=2, column=1, padx=4, pady=2)
        self._agent_buttons['saved'] = btn
        self._saved_agent_btn = btn

        # Checkpoint notification label (placed below checkpoint buttons)
        self._checkpoint_label = tk.Label(agent_frame, text="", fg='green', font=(None, 8), wraplength=100)
        self._checkpoint_label.grid(row=3, column=0, columnspan=2, padx=4, pady=(2, 4))

        # Difficulty selector (placed below agent selection)
        diff_frame = tk.Frame(self.sidebar)
        diff_frame.pack(fill="x", pady=(0,6))
        tk.Label(diff_frame, text="Difficulty:").pack(side="left")
        # display labels mapped to values passed to Game.new_game
        self._diff_map = {"Beginner": "beginner", "Intermediate": "intermediate", "Expert": "expert"}
        self._diff_var = tk.StringVar(value="Beginner")
        options = list(self._diff_map.keys())
        self._diff_menu = tk.OptionMenu(diff_frame, self._diff_var, *options, command=self._on_difficulty_selected)
        self._diff_menu.pack(side="right")

        # Start button (below difficulty)
        start_btn = tk.Button(self.sidebar, text="Start", command=self.start_selected_agent)
        start_btn.pack(fill="x", pady=(6,6))

        # default selection -> Random
        try:
            self._select_agent('random', self._agent_buttons['random'])
        except Exception:
            self.selected_agent = 'random'

        # Episodes heading
        tk.Label(self.sidebar, text="Episodes:", font=(None, 10, 'bold')).pack(anchor='w', pady=(6,2), padx=2)

        # Training episodes selector (dropdown + custom entry) - legacy logic uses this
        self._train_episodes_var = tk.StringVar(value="1")
        episode_options = ["1", "5", "10", "50"]
        train_frame = tk.Frame(self.sidebar)
        train_frame.pack(fill="x", pady=(2,4))
        tk.Label(train_frame, text="Training:").pack(side="left")
        train_select = tk.Frame(train_frame)
        train_select.pack(side="right")
        train_menu = tk.OptionMenu(train_select, self._train_episodes_var, *episode_options)
        train_menu.pack(side='left')
        self._train_episodes_entry = tk.Entry(train_select, width=6)
        self._train_episodes_entry.pack(side='left', padx=(6,0))

        # Testing episodes selector (dropdown + entry), default 0
        test_frame = tk.Frame(self.sidebar)
        test_frame.pack(fill="x", pady=(4,6))
        tk.Label(test_frame, text="Testing:").pack(side="left")
        # include 0 as an option for testing
        self._test_episodes_var = tk.StringVar(value="0")
        test_select = tk.Frame(test_frame)
        test_select.pack(side='right')
        test_options = ["0", "1", "5", "10", "50"]
        test_menu = tk.OptionMenu(test_select, self._test_episodes_var, *test_options)
        test_menu.pack(side='left')
        self._test_episodes_entry = tk.Entry(test_select, width=6)
        self._test_episodes_entry.insert(0, "0")
        self._test_episodes_entry.pack(side='left', padx=(6,0))
        
        # Hyperparameters panel (only affects upcoming RL/Hybrid agent)
        hp_frame = tk.LabelFrame(self.sidebar, text="Hyperparameters", padx=4, pady=4)
        hp_frame.pack(fill='x', pady=(6,6))

        # store references to entry vars
        self._hp_vars = {}

        def _add_hp_row(frame, label_text, var_name, default, width=8):
            row = tk.Frame(frame)
            row.pack(fill='x', pady=(2,2))
            tk.Label(row, text=label_text).pack(side='left')
            v = tk.StringVar(value=str(default))
            ent = tk.Entry(row, textvariable=v, width=width)
            ent.pack(side='right')
            self._hp_vars[var_name] = v

        _add_hp_row(hp_frame, 'Batch size', 'BATCH_SIZE', BATCH_SIZE)
        _add_hp_row(hp_frame, 'Gamma', 'GAMMA', GAMMA)
        _add_hp_row(hp_frame, 'Eps start', 'EPS_START', EPS_START)
        _add_hp_row(hp_frame, 'Eps end', 'EPS_END', EPS_END)
        _add_hp_row(hp_frame, 'Eps decay', 'EPS_DECAY', EPS_DECAY)
        _add_hp_row(hp_frame, 'Tau', 'TAU', TAU)
        _add_hp_row(hp_frame, 'LR', 'LR', LR)

        btns = tk.Frame(hp_frame)
        btns.pack(fill='x', pady=(4,0))
        apply_btn = tk.Button(btns, text='Apply', width=8, command=lambda: self._apply_hyperparams())
        apply_btn.pack(side='left')
        reset_btn = tk.Button(btns, text='Reset', width=8, command=lambda: self._reset_hyperparams())
        reset_btn.pack(side='right')

        # Error message for hyperparameter validation (wraps long text)
        # Use tk.Message which handles wrapping more reliably than Label
        try:
            self._hp_error_label = tk.Message(hp_frame, text="", fg="red", anchor='w', justify='left', width = 170)
            self._hp_error_label.pack(fill='x', pady=(4,0))
        except Exception:
            # fallback to Label if Message isn't available
            self._hp_error_label = tk.Label(hp_frame, text="", fg="red", anchor='w', justify='left', wraplength=170)
            self._hp_error_label.pack(fill='x', pady=(4,0))

        # active hyperparams dict (used when instantiating agents)
        self._agent_hyperparams = None

        # Lock the root window to current size so it is fixed
        try:
            self.root.update_idletasks()
            w = self.root.winfo_width()
            h = self.root.winfo_height()
            # set min/max to current size and disable resizing
            self.root.minsize(w, h)
            self.root.maxsize(w, h)
        except Exception:
            pass

    def _get_hyperparam_defaults(self):
        """Return default hyperparameter values from config."""
        return {
            'BATCH_SIZE': BATCH_SIZE,
            'GAMMA': GAMMA,
            'EPS_START': EPS_START,
            'EPS_END': EPS_END,
            'EPS_DECAY': EPS_DECAY,
            'TAU': TAU,
            'LR': LR,
        }
    
    def _reset_hyperparams(self):
        """Reset UI entries to defaults from config."""
        defaults = self._get_hyperparam_defaults()
        for k, v in defaults.items():
            if k in self._hp_vars:
                self._hp_vars[k].set(str(v))
        if hasattr(self, '_hp_error_label'):
            self._hp_error_label.config(text="")

    def _validate_hyperparam(self, name, validator, defaults):
        """Validate a single hyperparameter and return its value or None on error."""
        try:
            value = validator(self._hp_vars[name].get())
            return value
        except Exception as e:
            self._hp_vars[name].set(str(defaults[name]))
            if hasattr(self, '_hp_error_label'):
                self._hp_error_label.config(text=f"{name}: {e}")
            return None
    
    def _apply_hyperparams(self):
        """Validate hyperparameter entries and store them for next agent instantiation."""
        defaults = self._get_hyperparam_defaults()
        vals = {}
        
        if hasattr(self, '_hp_error_label'):
            self._hp_error_label.config(text="")
        
        # Define validators
        validators = {
            'BATCH_SIZE': lambda x: int(x) if int(x) >= 1 else (_ for _ in ()).throw(ValueError('must be >= 1')),
            'GAMMA': lambda x: float(x) if 0.0 <= float(x) <= 1.0 else (_ for _ in ()).throw(ValueError('must be in [0,1]')),
            'EPS_DECAY': lambda x: int(x) if int(x) >= 1 else (_ for _ in ()).throw(ValueError('must be >= 1')),
            'TAU': lambda x: float(x) if 0.0 <= float(x) <= 1.0 else (_ for _ in ()).throw(ValueError('must be in [0,1]')),
            'LR': lambda x: float(x) if float(x) > 0.0 else (_ for _ in ()).throw(ValueError('must be > 0')),
        }
        
        # Validate single params
        for name, validator in validators.items():
            value = self._validate_hyperparam(name, validator, defaults)
            if value is None:
                return
            vals[name] = value
        
        # Validate EPS_START/EPS_END together
        try:
            es = float(self._hp_vars['EPS_START'].get())
            ee = float(self._hp_vars['EPS_END'].get())
            if not (0.0 <= es <= 1.0 and 0.0 <= ee <= 1.0 and es >= ee):
                raise ValueError('must be in [0,1] and EPS_START >= EPS_END')
            vals['EPS_START'] = es
            vals['EPS_END'] = ee
        except Exception as e:
            self._hp_vars['EPS_START'].set(str(defaults['EPS_START']))
            self._hp_vars['EPS_END'].set(str(defaults['EPS_END']))
            if hasattr(self, '_hp_error_label'):
                self._hp_error_label.config(text=f"EPS: {e}")
            return
        
        self._agent_hyperparams = vals
        if hasattr(self, '_hp_error_label'):
            self._hp_error_label.config(text="")
        messagebox.showinfo('Hyperparams applied', 'Hyperparameters saved for next agent instantiation')

    def _get_episodes_from_ui(self):
        """Read episode count from UI (custom entry or dropdown). Returns int."""
        episodes = None
        try:
            custom = self._train_episodes_entry.get().strip()
            if custom:
                episodes = int(custom)
        except Exception:
            episodes = None

        if episodes is None:
            try:
                episodes = int(self._train_episodes_var.get())
            except Exception:
                episodes = 1

        return max(1, int(episodes))

    def _reset_hyperparams_ui(self):
        """Reset hyperparameter entry widgets to defaults from config.py."""
        defaults = {
            'BATCH_SIZE': str(self._get_config_value('BATCH_SIZE', '128')),
            'GAMMA': str(self._get_config_value('GAMMA', '0.99')),
            'EPS_START': str(self._get_config_value('EPS_START', '0.9')),
            'EPS_END': str(self._get_config_value('EPS_END', '0.05')),
            'EPS_DECAY': str(self._get_config_value('EPS_DECAY', '4000')),
            'TAU': str(self._get_config_value('TAU', '0.005')),
            'LR': str(self._get_config_value('LR', '3e-4')),            
        }
        for k, ent in self._hp_fields.items():
            try:
                ent.delete(0, 'end')
                ent.insert(0, defaults.get(k, ''))
            except Exception:
                pass
        # clear any existing error message
        try:
            if hasattr(self, '_hp_error_label'):
                self._hp_error_label.config(text="")
        except Exception:
            pass

    def _get_config_value(self, name: str, fallback: str):
        """Utility: read a value from the imported config module namespace."""
        try:
            from config import __dict__ as _cdict
        except Exception:
            _cdict = globals()
        try:
            return globals().get(name, fallback)
        except Exception:
            return fallback

    def _apply_hyperparams_from_ui(self):
        """Validate entries and store as `self._agent_hyperparams` dict applied to next agent."""
        try:
            vals = {}
            # BATCH_SIZE: int >=1
            bs = int(self._hp_fields['BATCH_SIZE'].get())
            if bs < 1:
                raise ValueError('BATCH_SIZE must be >= 1')
            vals['BATCH_SIZE'] = bs

            # GAMMA: float in [0,1]
            gamma = float(self._hp_fields['GAMMA'].get())
            if not (0.0 <= gamma <= 1.0):
                raise ValueError('GAMMA must be in [0,1]')
            vals['GAMMA'] = gamma

            # EPS_START / EPS_END: floats in [0,1], EPS_START >= EPS_END
            eps_s = float(self._hp_fields['EPS_START'].get())
            eps_e = float(self._hp_fields['EPS_END'].get())
            if not (0.0 <= eps_s <= 1.0 and 0.0 <= eps_e <= 1.0):
                raise ValueError('EPS_START and EPS_END must be in [0,1]')
            if eps_s < eps_e:
                raise ValueError('EPS_START must be >= EPS_END')
            vals['EPS_START'] = eps_s
            vals['EPS_END'] = eps_e

            # EPS_DECAY: int >=1
            ed = int(self._hp_fields['EPS_DECAY'].get())
            if ed < 1:
                raise ValueError('EPS_DECAY must be >= 1')
            vals['EPS_DECAY'] = ed

            # TAU: float in [0,1]
            tau = float(self._hp_fields['TAU'].get())
            if not (0.0 <= tau <= 1.0):
                raise ValueError('TAU must be in [0,1]')
            vals['TAU'] = tau

            # LR: float > 0
            lr = float(self._hp_fields['LR'].get())
            if not (lr > 0.0):
                raise ValueError('LR must be > 0')
            vals['LR'] = lr

            # store
            self._agent_hyperparams = vals
            self.update_status('Status: Hyperparameters applied for next agent instantiation')
        except Exception as e:
            try:
                messagebox.showerror('Invalid hyperparameter', str(e))
            except Exception:
                pass

    def _build_agent_init_kwargs(self, base: dict = None):
        """Merge base kwargs with app-level hyperparams."""
        init = dict(base) if isinstance(base, dict) else {}
        hp = getattr(self, '_agent_hyperparams', None)
        if hp and 'hyperparams' not in init:
            init['hyperparams'] = hp
        return init if init else None

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

    def _schedule_ep_display_hyperparams(self, ep_display, agent):
        """If available, schedule transfer of agent hyperparams to the per-run display."""
        try:
            if ep_display is None or agent is None:
                return
            
            # Get hyperparams from the agent
            cfg = None
            if hasattr(agent, 'get_config_dict'):
                try:
                    cfg = agent.get_config_dict()
                except Exception as e:
                    print(f"Warning: Failed to get config dict from agent: {e}")
                    return
            
            # Fallback: try reading attributes directly from agent
            if cfg is None and hasattr(agent, 'BATCH_SIZE'):
                try:
                    cfg = {
                        'BATCH_SIZE': getattr(agent, 'BATCH_SIZE', None),
                        'GAMMA': getattr(agent, 'GAMMA', None),
                        'EPS_START': getattr(agent, 'EPS_START', None),
                        'EPS_END': getattr(agent, 'EPS_END', None),
                        'EPS_DECAY': getattr(agent, 'EPS_DECAY', None),
                        'TAU': getattr(agent, 'TAU', None),
                        'LR': getattr(agent, 'LR', None),
                    }
                except Exception as e:
                    print(f"Warning: Failed to read hyperparams from agent attributes: {e}")
            
            if cfg:
                try:
                    self.root.after(0, lambda c=cfg, d=ep_display: d.set_hyper_params(c))
                except Exception as e:
                    print(f"Warning: Failed to schedule hyperparam update: {e}")
                    # Try direct call as fallback
                    try:
                        ep_display.set_hyper_params(cfg)
                    except Exception as e2:
                        print(f"Warning: Direct hyperparam update also failed: {e2}")
        except Exception as e:
            print(f"Warning: _schedule_ep_display_hyperparams failed: {e}")

    def _select_agent(self, key: str, button: tk.Button):
        """Select the agent identified by `key` and highlight the provided button.

        This centralizes selection/highlighting so all agent buttons behave
        consistently. `key` should be one of: 'random', 'baseline', 'rl', 'hybrid'.
        """
        try:
            # un-highlight all
            for k, b in getattr(self, '_agent_buttons', {}).items():
                try:
                    b.config(relief='raised')
                except Exception:
                    pass
            # highlight the selected button
            try:
                button.config(relief='sunken')
            except Exception:
                pass
            # store selection
            self.selected_agent = key
            
            # If saved agent is selected, load checkpoint settings
            if key == 'saved':
                try:
                    self._apply_checkpoint_settings()
                except Exception as e:
                    pass
            
            # Enable difficulty selector when non-saved agent is selected
            # Disable it when saved agent is selected (checkpoint determines difficulty)
            try:
                if hasattr(self, '_diff_menu'):
                    if key == 'saved':
                        self._diff_menu.config(state='disabled')
                    else:
                        self._diff_menu.config(state='normal')
            except Exception:
                pass
            
            # update status bar
            try:
                pretty = {
                    'random': 'Random agent', 
                    'baseline': 'Baseline agent', 
                    'rl': 'RL agent', 
                    'hybrid': 'Hybrid agent',
                    'saved': 'Saved agent'
                }.get(key, str(key))
                self.update_status(f"Selected: {pretty}")
            except Exception:
                pass
        except Exception:
            try:
                self.selected_agent = key
            except Exception:
                pass

    def start_selected_agent(self):
        """Dispatch to the configured start_* method for the selected agent."""
        key = getattr(self, 'selected_agent', None)
        if not key:
            self.update_status("Status: No agent selected")
            return
        mapping = {
            'random': self.start_random_agent,
            'baseline': self.start_baseline_agent,
            'rl': self.start_rl_agent,
            'hybrid': self.start_hybrid_agent,
            'saved': self.start_saved_agent,
        }
        fn = mapping.get(key)
        if fn is None:
            self.update_status(f"Status: Unknown agent '{key}'")
            return
        try:
            fn()
        except Exception as e:
            try:
                self.update_status(f"Status: Failed to start agent: {e}")
            except Exception:
                pass

    def _instantiate_agent(self, env, module_name: str = None, class_name: str = None, agent_init_kwargs: dict = None, worker_callable=None, task_kwargs: dict = None, agent_name: str = None):
        """Resolve and instantiate an agent class.
        
        Returns: (AgentClass, agent_instance, module_name, class_name)
        """
        # Agent registry for clean lookups
        AGENT_MAP = {
            'RL_agent': {'default': 'DQNAgent', 'hybrid_keyword': 'Hybrid_Agent'},
            'baseline_agent': {'default': 'BaselineAgent'},
            'random_agent': {'default': 'RandomAgent'}
        }
        
        task_kwargs = task_kwargs or {}
        
        # Determine target module and class
        if module_name and module_name in AGENT_MAP:
            target_module = importlib.import_module(module_name)
            config = AGENT_MAP[module_name]
            
            # Check for hybrid keyword in agent_name
            if 'hybrid_keyword' in config and agent_name and 'hybrid' in agent_name.lower():
                target_class_name = config['hybrid_keyword']
            else:
                target_class_name = class_name or config['default']
        
        # Fallback: parse worker_callable name
        elif worker_callable:
            name_lower = getattr(worker_callable, '__name__', '').lower()
            if 'hybrid' in name_lower:
                target_module = importlib.import_module('RL_agent')
                target_class_name = 'Hybrid_Agent'
            elif 'rl' in name_lower:
                target_module = importlib.import_module('RL_agent')
                target_class_name = 'DQNAgent'
            elif 'baseline' in name_lower:
                target_module = importlib.import_module('baseline_agent')
                target_class_name = 'BaselineAgent'
            elif 'random' in name_lower:
                target_module = importlib.import_module('random_agent')
                target_class_name = 'RandomAgent'
            else:
                target_module = importlib.import_module('RL_agent')
                target_class_name = 'DQNAgent'
        
        # Last resort default
        else:
            target_module = importlib.import_module('RL_agent')
            target_class_name = class_name or 'DQNAgent'

        AgentClass = getattr(target_module, target_class_name)
        
        # Handle baseline injection for Hybrid_Agent
        init_kwargs = dict(agent_init_kwargs) if agent_init_kwargs else {}
        baseline_instance = None
        
        if '_baseline' in init_kwargs:
            bspec = init_kwargs.pop('_baseline')
            if isinstance(bspec, dict):
                bm = importlib.import_module(bspec['module'])
                BClass = getattr(bm, bspec['class'])
                baseline_instance = BClass(env, **bspec.get('kwargs', {}))
            else:
                baseline_instance = bspec
        
        # Auto-create baseline for Hybrid_Agent if not provided
        if baseline_instance is None and target_class_name == 'Hybrid_Agent':
            bm = importlib.import_module('baseline_agent')
            baseline_instance = getattr(bm, 'BaselineAgent')(env)
        
        # Instantiate agent
        merged_kwargs = task_kwargs.get('agent_init_kwargs', init_kwargs)
        if baseline_instance:
            agent = AgentClass(env, baseline_instance, **merged_kwargs) if merged_kwargs else AgentClass(env, baseline_instance)
        else:
            agent = AgentClass(env, **merged_kwargs) if merged_kwargs else AgentClass(env)

        # Load checkpoint if provided (ONLY for saved agent path)
        checkpoint_path = task_kwargs.get('checkpoint_path')
        if checkpoint_path:
            try:
                # RL agents (DQNAgent, Hybrid_Agent) have load_checkpoint method
                if hasattr(agent, 'load_checkpoint'):
                    agent.load_checkpoint(checkpoint_path)
                else:
                    # For non-RL agents (Random, Baseline), checkpoint loading is not applicable
                    pass
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")

        return AgentClass, agent, target_module.__name__, target_class_name
   
App().run()