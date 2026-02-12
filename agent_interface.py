from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List


class agent_interface(ABC):
    """Abstract agent interface.

    - `run_episode` must be implemented by concrete agents.
    - `run_num_episodes` provided here follows the existing project's
      contract (calls `run_episode` repeatedly and optionally invokes
      `progress_update(info)` after each episode).
    """

    @abstractmethod
    def run_episode(self, difficulty: str = None, max_steps: int = 100000, delay: float = 0.0) -> Dict[str, Any]:
        raise NotImplementedError()

    def run_num_episodes(self, num_episodes: int, difficulty: str = None, max_steps: int = 100000, delay: float = 0.0, progress_update: Callable[[Dict[str, Any]], None] = None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i in range(1, int(num_episodes) + 1):
            res = self.run_episode(difficulty=difficulty, max_steps=max_steps, delay=delay)
            results.append(res)

            # determine win flag (mirror existing code)
            win_flag = False
            try:
                if res.get('win', False) and res.get('reward', 0) > 0:
                    win_flag = True
            except Exception:
                win_flag = False

            # include random_clicks (support legacy 'random_click') and done
            random_clicks = None
            done_flag = False
            if isinstance(res, dict):
                random_clicks = res.get('random_clicks', res.get('random_click', 0))
                done_flag = bool(res.get('done', False))

            info = {
                'episode': i,
                'length': len(res.get('history', [])) if isinstance(res, dict) else None,
                'win': bool(win_flag),
                'reward': res.get('reward', 0) if isinstance(res, dict) else None,
                'random_clicks': random_clicks,
                'done': done_flag,
            }
            try:
                if callable(progress_update):
                    progress_update(info)
            except Exception:
                pass

        return results
