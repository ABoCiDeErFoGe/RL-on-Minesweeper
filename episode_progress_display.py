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
        self.difficulty_label = tk.Label(info_frame, text=difficulty if difficulty is not None else 'N/A')
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

        # keep internal episode counter for x-axis if incoming info lacks 'episode'
        self._next_ep_index = 1
        # run counters
        self._finished_episodes = 0
        self._total_episodes = int(total_episodes) if total_episodes is not None else None

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

        episodes = list(range(1, len(self.episode_length_record) + 1))

        # update plots
        try:
            self.ax_len.clear()
            self.ax_len.plot(episodes, self.episode_length_record, color='0.75', linewidth=1)
            colors = ['#11f54e' if w else 'red' for w in self.win_record]
            self.ax_len.scatter(episodes, self.episode_length_record, c=colors, edgecolors='k')
            self.ax_len.set_ylabel('Episode length')

            self.ax_win.clear()
            self.ax_win.bar(['Lose', 'Win'], [len(self.win_record) - sum(self.win_record), sum(self.win_record)], color=['red','green'])
            self.ax_win.set_ylabel('Count')

            self.ax_reward.clear()
            self.ax_reward.plot(episodes, self.reward_record, color='0.75', linewidth=1)
            self.ax_reward.scatter(episodes, self.reward_record, c=colors, edgecolors='k')
            self.ax_reward.set_ylabel('Reward')

            self.ax_random.clear()
            self.ax_random.plot(episodes, self.random_click_record, color='0.75', linewidth=1)
            self.ax_random.scatter(episodes, self.random_click_record, c=colors, edgecolors='k')
            self.ax_random.set_ylabel('Random clicks')
            self.ax_random.set_xlabel('Episode')

            self.canvas.draw_idle()
        except Exception:
            pass

        # update side labels
        try:
            n = len(self.win_record)
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
        try:
            self.window.after(0, lambda: self.difficulty_label.config(text=diff))
        except Exception:
            try:
                self.difficulty_label.config(text=diff)
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
