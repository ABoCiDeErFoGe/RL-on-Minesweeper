import threading
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class EpisodeProgressDisplay:
    """Window that displays episode progress and side-panel summary.

    Uses a `Toplevel` when provided a `master` to avoid creating multiple
    root `Tk()` windows. If `master` is None it falls back to creating its
    own `Tk()` so the class remains usable standalone.

    New optional init args:
    - `agent_name`: display the agent's name (static for the run)
    - `difficulty`: display difficulty string (static for the run)
    - `total_episodes`: integer total episodes for the run (used to show finished/total)
    """
    def __init__(self, title="Episode Progress", master=None, agent_name: str = None, difficulty: str = None, total_episodes: int = None):
        # internal records
        self.win_record = []
        self.random_click_record = []
        self.episode_length_record = []
        self.reward_record = []

        # window ownership: if master provided, use Toplevel(master), else create Tk
        if master is None:
            self._owns_root = True
            self.root = tk.Tk()
            self.window = self.root
        else:
            self._owns_root = False
            self.root = master
            self.window = tk.Toplevel(master)
        try:
            self.window.title(title)
        except Exception:
            pass

        # main container: left = plots, right = summary panel
        container = tk.Frame(self.window)
        container.pack(fill='both', expand=True)

        plot_frame = tk.Frame(container)
        plot_frame.pack(side='left', fill='both', expand=True)

        side_frame = tk.Frame(container, width=240, relief='groove', bd=1)
        side_frame.pack(side='right', fill='y')

        # matplotlib figure with 4 subplots
        self.fig = plt.Figure(figsize=(6, 6))
        self.ax_len = self.fig.add_subplot(411)
        self.ax_win = self.fig.add_subplot(412)
        self.ax_reward = self.fig.add_subplot(413)
        self.ax_random = self.fig.add_subplot(414)

        # canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # side panel info: top shows run metadata (Agent / Difficulty / Episodes)
        info_frame = tk.Frame(side_frame)
        info_frame.pack(fill='x', padx=6, pady=(6,4))

        tk.Label(info_frame, text='Agent:').grid(row=0, column=0, sticky='w')
        self.agent_label = tk.Label(info_frame, text=agent_name if agent_name is not None else 'N/A')
        self.agent_label.grid(row=0, column=1, sticky='w')

        tk.Label(info_frame, text='Difficulty:').grid(row=1, column=0, sticky='w')
        # append board size and mine count for known difficulties using a switch-like match
        try:
            if difficulty is None:
                diff_text = 'N/A'
            else:
                dlow = str(difficulty).lower()
                print(dlow)
                if dlow == 'beginner':
                    diff_text = f"{difficulty} (9x9, 10 mines)"
                elif dlow == 'intermediate':
                    diff_text = f"{difficulty} (16x16, 40 mines)"
                elif dlow == 'expert':
                    diff_text = f"{difficulty} (16x30, 99 mines)"
                else:
                    diff_text = str(difficulty)
        except Exception:
            print("oops")
            diff_text = difficulty if difficulty is not None else 'N/A'

        self.difficulty_label = tk.Label(info_frame, text=diff_text)
        self.difficulty_label.grid(row=1, column=1, sticky='w')

        tk.Label(info_frame, text='Episodes:').grid(row=2, column=0, sticky='w')
        total_text = f"0/{total_episodes}" if total_episodes is not None else "0/0"
        self.episodes_label = tk.Label(info_frame, text=total_text)
        self.episodes_label.grid(row=2, column=1, sticky='w')

        # summary labels
        tk.Label(side_frame, text='Summary', font=(None, 12, 'bold')).pack(pady=(6,4))
        self.win_rate_label = tk.Label(side_frame, text='Win rate: 0.0%')
        self.win_rate_label.pack(anchor='w', padx=8, pady=2)
        self.avg_random_label = tk.Label(side_frame, text='Avg random clicks: 0.0')
        self.avg_random_label.pack(anchor='w', padx=8, pady=2)
        self.avg_length_label = tk.Label(side_frame, text='Avg episode length: 0.0')
        self.avg_length_label.pack(anchor='w', padx=8, pady=2)
        self.avg_reward_label = tk.Label(side_frame, text='Avg reward: 0.0')
        self.avg_reward_label.pack(anchor='w', padx=8, pady=2)

        # Save status label
        self.save_status_label = tk.Label(side_frame, text='Save status: —', fg='gray', wraplength=160)
        self.save_status_label.pack(anchor='w', padx=8, pady=(6,2))

        # Hyperparameters panel (read-only labels)
        tk.Label(side_frame, text='Hyperparameters', font=(None, 12, 'bold')).pack(pady=(8,4))
        self._hyper_frame = tk.Frame(side_frame)
        self._hyper_frame.pack(fill='x', padx=6, pady=(0,6))

        # mapping display name -> attribute name on RL agent
        self._hp_map = [
            ('Batch size', 'BATCH_SIZE'),
            ('Gamma', 'GAMMA'),
            ('Eps start', 'EPS_START'),
            ('Eps end', 'EPS_END'),
            ('Eps decay', 'EPS_DECAY'),
            ('Tau', 'TAU'),
            ('Lr', 'LR'),
        ]

        self.hyperparam_labels = {}
        for i, (disp, _) in enumerate(self._hp_map):
            tk.Label(self._hyper_frame, text=f'{disp}:').grid(row=i, column=0, sticky='w')
            val = tk.Label(self._hyper_frame, text='N/A')
            val.grid(row=i, column=1, sticky='w')
            self.hyperparam_labels[disp] = val

        # keep internal episode counter for x-axis if incoming info lacks 'episode'
        self._next_ep_index = 1
        # run counters
        self._finished_episodes = 0
        self._total_episodes = int(total_episodes) if total_episodes is not None else None
        
        # Performance optimization settings
        self._max_display_points = 1000  # Max points to show in plots
        self._scatter_threshold = 200  # Use scatter only if fewer points than this
        self._plot_update_interval = 5  # Update plots every N episodes for large datasets
        self._episodes_since_plot_update = 0
        
        # Aggregation settings for high episode counts
        self._aggregation_window = 100  # Aggregate every 100 episodes
        self._aggregation_threshold = 500  # Start aggregating when exceeding this many episodes
        
        # Aggregated data (win rate, episode length, random clicks, rewards per 100 episodes)
        self._aggregated_windows = []  # List of (window_num, avg_win_rate, avg_length, avg_randoms, avg_reward)
        self._current_window_data = {  # Current 100-episode window being accumulated
            'wins': [],
            'lengths': [],
            'randoms': [],
            'rewards': []
        }

    def _apply_update(self, info: dict):
        """Apply update on the GUI thread (internal)."""
        # extract values from info
        ep = info.get('episode') if isinstance(info.get('episode', None), int) else None
        length = info.get('length', info.get('steps', None))
        try:
            length = int(length) if length is not None else None
        except Exception:
            length = None
        win = bool(info.get('win', False))
        reward = info.get('reward', 0)
        try:
            reward = float(reward)
        except Exception:
            reward = 0.0
        from utils import extract_random_clicks
        random_clicks = extract_random_clicks(info)

        # determine episode index
        if ep is None:
            ep = self._next_ep_index
        self._next_ep_index = max(self._next_ep_index, ep + 1)

        # update run progress if an episode index is provided
        try:
            if isinstance(ep, int):
                # finished episodes count becomes the reported episode index
                self._finished_episodes = ep
                if self._total_episodes is None:
                    # if total unknown, infer from counting
                    self._total_episodes = self._total_episodes or None
                # refresh episodes label
                try:
                    self.episodes_label.config(text=f"{self._finished_episodes}/{self._total_episodes if self._total_episodes is not None else len(self.episode_length_record)}")
                except Exception:
                    pass
        except Exception:
            pass

        # append records
        self.episode_length_record.append(length if length is not None else 0)
        self.reward_record.append(reward)
        self.random_click_record.append(random_clicks)
        self.win_record.append(1 if win else 0)
        
        # Handle aggregation for large datasets
        total_episodes = len(self.episode_length_record)

        episodes = list(range(1, len(self.episode_length_record) + 1))

        # Performance optimization: only update plots periodically for large datasets
        self._episodes_since_plot_update += 1
        total_episodes = len(self.episode_length_record)
        
        # Determine if we should update plots this iteration
        should_update_plots = (
            total_episodes <= 100 or  # Always update for small datasets
            self._episodes_since_plot_update >= self._plot_update_interval or  # Periodic updates
            total_episodes == self._total_episodes  # Always update on final episode
        )
        
        if not should_update_plots:
            # Skip plot update but still update summary stats
            self._update_summary_stats()
            return
        
        self._episodes_since_plot_update = 0  # Reset counter

        # Get plot data (either individual episodes or aggregated windows)
        all_episodes, all_lengths, all_rewards, all_randoms, all_wins, is_aggregated = self._get_plot_data()
        
        # Apply downsampling only if not using aggregated data
        if not is_aggregated:
            display_episodes, display_lengths, display_rewards, display_randoms, display_wins = self._downsample_data(
                all_episodes, 
                all_lengths, 
                all_rewards, 
                all_randoms, 
                all_wins
            )
        else:
            # Already downsampled via aggregation
            display_episodes, display_lengths, display_rewards, display_randoms, display_wins = (
                all_episodes, all_lengths, all_rewards, all_randoms, all_wins
            )

        # update plots
        try:
            # Determine if we should use scatter (expensive) or just lines
            use_scatter = (not is_aggregated) and (len(display_episodes) <= self._scatter_threshold)

            colors = ['#11f54e' if w else 'red' for w in display_wins] if use_scatter else None
            
            # Determine axis labels based on data type
            if is_aggregated:
                ep_label_suffix = " (100-ep windows)"
                win_title = "Win Rate / 100"
                len_title = "Avg Episode Length / 100"
                reward_title = "Avg Reward / 100"
                random_title = "Avg Random Clicks / 100"
            else:
                ep_label_suffix = ""
                win_title = "Individual Episodes"
                len_title = "Episode Length"
                reward_title = "Reward"
                random_title = "Random Clicks"
            
            self.ax_len.clear()
            self.ax_len.plot(display_episodes, display_lengths, color='0.75', linewidth=1)
            if use_scatter:
                self.ax_len.scatter(display_episodes, display_lengths, c=colors, edgecolors='k', s=20)
            self.ax_len.set_ylabel(len_title)
            self.ax_len.grid(True, alpha=0.3)

            self.ax_win.clear()
            if is_aggregated:
                self.ax_win.plot(display_episodes, display_wins, color='green', linewidth=1.5)
                self.ax_win.set_ylim(0, 100)
                self.ax_win.set_ylabel('Win rate (%)')
                self.ax_win.set_title(win_title, fontsize=7)
            else:
                total_wins = sum(self.win_record)
                total_losses = len(self.win_record) - total_wins
                self.ax_win.bar(['Lose', 'Win'], [total_losses, total_wins], color=['red','green'])
                self.ax_win.set_ylabel('Count')
                self.ax_win.set_title(win_title, fontsize=9)
            self.ax_win.grid(True, alpha=0.3)

            self.ax_reward.clear()
            self.ax_reward.plot(display_episodes, display_rewards, color='0.75', linewidth=1)
            if use_scatter:
                self.ax_reward.scatter(display_episodes, display_rewards, c=colors, edgecolors='k', s=20)
            self.ax_reward.set_ylabel(reward_title)
            self.ax_reward.grid(True, alpha=0.3)

            self.ax_random.clear()
            self.ax_random.plot(display_episodes, display_randoms, color='0.75', linewidth=1)
            if use_scatter:
                self.ax_random.scatter(display_episodes, display_randoms, c=colors, edgecolors='k', s=20)
            self.ax_random.set_ylabel(random_title)
            self.ax_random.set_xlabel(f'Episode{ep_label_suffix}')
            self.ax_random.grid(True, alpha=0.3)

            self.fig.tight_layout()
            self.canvas.draw_idle()
        except Exception:
            pass

        # update side labels
        self._update_summary_stats()

    def _accumulate_aggregation(self, win, length, random_clicks, reward, total_episodes):
        """Accumulate data into 100-episode windows for large datasets."""
        # Add current episode to current window
        self._current_window_data['wins'].append(1 if win else 0)
        self._current_window_data['lengths'].append(length if length is not None else 0)
        self._current_window_data['randoms'].append(random_clicks)
        self._current_window_data['rewards'].append(reward)
        
        # Check if current window is complete (100 episodes)
        if len(self._current_window_data['wins']) >= self._aggregation_window:
            # Calculate averages for this window
            avg_win_rate = (sum(self._current_window_data['wins']) / len(self._current_window_data['wins'])) * 100
            avg_length = sum(self._current_window_data['lengths']) / len(self._current_window_data['lengths'])
            avg_randoms = sum(self._current_window_data['randoms']) / len(self._current_window_data['randoms'])
            avg_reward = sum(self._current_window_data['rewards']) / len(self._current_window_data['rewards'])
            
            # Store aggregated window (window number starts at 1, representing episodes 1-100, 101-200, etc)
            window_num = len(self._aggregated_windows) + 1
            self._aggregated_windows.append((window_num, avg_win_rate, avg_length, avg_randoms, avg_reward))
            
            # Reset for next window
            self._current_window_data = {'wins': [], 'lengths': [], 'randoms': [], 'rewards': []}
    
    def _get_plot_data(self):
        """Get data to plot: individual episodes for small datasets, aggregated for large.
        
        Returns:
            (episodes, lengths, rewards, randoms, wins, use_aggregated)
        """
        total_episodes = len(self.episode_length_record)
        
        # Use aggregated data if we're above threshold, computed from full history
        if total_episodes >= self._aggregation_threshold:
            windows_nums = []
            win_rates = []
            lengths = []
            randoms = []
            rewards = []

            window = max(1, int(self._aggregation_window))
            for start in range(0, total_episodes, window):
                end = min(start + window, total_episodes)
                chunk_wins = self.win_record[start:end]
                chunk_lengths = self.episode_length_record[start:end]
                chunk_randoms = self.random_click_record[start:end]
                chunk_rewards = self.reward_record[start:end]

                if not chunk_wins:
                    continue

                # x-axis uses the ending episode number of each window
                windows_nums.append(end)
                win_rates.append((sum(chunk_wins) / len(chunk_wins)) * 100.0)
                lengths.append(sum(chunk_lengths) / len(chunk_lengths))
                randoms.append(sum(chunk_randoms) / len(chunk_randoms))
                rewards.append(sum(chunk_rewards) / len(chunk_rewards))

            return windows_nums, lengths, rewards, randoms, win_rates, True
        else:
            # Use individual episodes
            episodes = list(range(1, total_episodes + 1))
            return episodes, self.episode_length_record, self.reward_record, self.random_click_record, self.win_record, False

    def _downsample_data(self, episodes, lengths, rewards, randoms, wins):
        """Downsample data for efficient plotting with large datasets.
        
        Uses a sliding window approach: shows most recent episodes up to max_display_points.
        For very large datasets, applies additional intelligent downsampling.
        """
        n = len(episodes)
        
        # If dataset is small enough, return as-is
        if n <= self._max_display_points:
            return episodes, lengths, rewards, randoms, wins
        
        # Use sliding window: show most recent episodes
        start_idx = n - self._max_display_points
        
        return (
            episodes[start_idx:],
            lengths[start_idx:],
            rewards[start_idx:],
            randoms[start_idx:],
            wins[start_idx:]
        )
    
    def _update_summary_stats(self):
        """Update the summary statistics labels on the side panel."""
        try:
            n = len(self.win_record)
            if n == 0:
                return
            
            win_rate = (sum(self.win_record) / n * 100.0) if n else 0.0
            avg_random = (sum(self.random_click_record) / n) if n else 0.0
            avg_length = (sum(self.episode_length_record) / n) if n else 0.0
            avg_reward = (sum(self.reward_record) / n) if n else 0.0

            self.win_rate_label.config(text=f'Win rate: {win_rate:.1f}%')
            self.avg_random_label.config(text=f'Avg random clicks: {avg_random:.2f}')
            self.avg_length_label.config(text=f'Avg episode length: {avg_length:.2f}')
            self.avg_reward_label.config(text=f'Avg reward: {avg_reward:.2f}')
        except Exception:
            pass

    def progress_update(self, info: dict):
        """Public method to accept an episode info dict from any thread.

        Schedules the GUI update on the Tkinter main loop.
        """
        try:
            self.window.after(0, lambda: self._apply_update(info))
        except Exception:
            # fallback: call directly
            try:
                self._apply_update(info)
            except Exception:
                pass

    def set_agent(self, name: str):
        """Thread-safe setter for the Agent label (static for the run)."""
        try:
            self.window.after(0, lambda: self.agent_label.config(text=name))
        except Exception:
            try:
                self.agent_label.config(text=name)
            except Exception:
                pass

    def set_difficulty(self, diff: str):
        """Thread-safe setter for the Difficulty label (static for the run)."""
        def _do_set(d):
            try:
                if d is None:
                    text = 'N/A'
                else:
                    dlow = str(d).strip().lower()
                    if dlow == 'beginner':
                        text = f"{d} (9x9, 10 mines)"
                    elif dlow == 'intermediate':
                        text = f"{d} (16x16, 40 mines)"
                    elif dlow == 'expert':
                        text = f"{d} (16x30, 99 mines)"
                    else:
                        text = str(d)
                self.difficulty_label.config(text=text)
            except Exception:
                try:
                    self.difficulty_label.config(text=d if d is not None else 'N/A')
                except Exception:
                    pass

        try:
            self.window.after(0, lambda: _do_set(diff))
        except Exception:
            _do_set(diff)

    def set_hyper_params(self, hyperparams: dict):
        """Thread-safe setter: read hyperparameters from a dictionary and
        update the read-only labels in the hyperparameters panel.

        The `hyperparams` dict should use keys matching the attribute names
        in `self._hp_map` (e.g. 'BATCH_SIZE', 'GAMMA', 'EPS_START', ...).
        Missing keys or None values display as 'N/A'.
        """
        def _format_value(v):
            try:
                if isinstance(v, int):
                    return str(v)
                if isinstance(v, float):
                    return f"{v:.6g}"
                return str(v)
            except Exception:
                return 'N/A'

        def _do_set(hp):
            try:
                if hp is None:
                    for disp, _ in self._hp_map:
                        try:
                            self.hyperparam_labels[disp].config(text='N/A')
                        except Exception:
                            pass
                    return

                for disp, key in self._hp_map:
                    try:
                        val = hp.get(key) if isinstance(hp, dict) else None
                        text = _format_value(val) if val is not None else 'N/A'
                        self.hyperparam_labels[disp].config(text=text)
                    except Exception:
                        try:
                            self.hyperparam_labels[disp].config(text='N/A')
                        except Exception:
                            pass
            except Exception:
                pass

        # Apply immediately for scripts/tests without a running Tk mainloop,
        # and also schedule a GUI-thread update for normal thread-safe usage.
        try:
            _do_set(hyperparams)
        except Exception:
            pass
        try:
            self.window.after(0, lambda: _do_set(hyperparams))
        except Exception:
            pass

    def set_run_progress(self, finished: int, total: int):
        """Thread-safe setter for finished/total episodes display."""
        try:
            self._finished_episodes = int(finished)
        except Exception:
            self._finished_episodes = finished
        try:
            self._total_episodes = int(total) if total is not None else None
        except Exception:
            pass
        # schedule UI update
        try:
            text = f"{self._finished_episodes}/{self._total_episodes if self._total_episodes is not None else 0}"
            self.window.after(0, lambda: self.episodes_label.config(text=text))
        except Exception:
            try:
                self.episodes_label.config(text=text)
            except Exception:
                pass

    def set_save_status(self, enabled: bool, message: str = None):
        """Update save status label with color coding.
        
        Args:
            enabled: True if save was enabled, False if disabled
            message: Additional status message (e.g., filename or error)
        """
        try:
            if not enabled:
                text = "Save status: Disable"
                color = 'red'
            elif message:
                text = f"Save status: Enable - {message}"
                color = 'green'
            else:
                text = "Save status: Enable"
                color = 'green'
            
            def _update():
                try:
                    self.save_status_label.config(text=text, fg=color)
                except Exception:
                    pass
            
            self.window.after(0, _update)
        except Exception:
            pass

    def run(self):
        """Start the Tkinter main loop. Blocks."""
        if self._owns_root:
            self.root.mainloop()
        else:
            # when embedded as Toplevel, do not start a new main loop
            try:
                self.window.lift()
            except Exception:
                pass
