from collections import namedtuple, deque
import math
import random
from torch import device, nn
from torch import optim
import torch
import torch.nn.functional as F

from Game import MSEnv

from config import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

class RL_agent:
    def __init__(self, env: MSEnv) -> None:
        self.env = env
        self.initialize_network()
        
    # reinitialize DQN based on the network
    def initialize_network(self):
        self.n_actions = self.env.game.w * self.env.game.h * 2  # left/right for each cell
        self.n_observations = self.env.game.w * self.env.game.h  # flattened board state
        self.policy_net = DQN(self.n_observations, self.n_actions)
        self.target_net = DQN(self.n_observations, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        # epsilon decay parameters
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * self.steps_done / EPS_DECAY)

        # obtain actionable mask (expected flat list length == n_actions)
        actionmask = self.env.game.get_actionable_mask()
        tensor_actionmask = torch.tensor(actionmask, dtype=torch.bool, device=self.device)

        # run network to get Q-values
        with torch.no_grad():
            result = self.policy_net(state)  # expected shape [1, n_actions]

        # ensure mask shape matches result: if result is 2D and mask is 1D,
        # unsqueeze to [1, n_actions] so boolean indexing broadcasts correctly.
        mask = tensor_actionmask
        if tensor_actionmask.dim() == 1 and result.dim() == 2:
            mask = tensor_actionmask.unsqueeze(0)

        if sample > eps_threshold:
            result_clone = result.clone()
            # apply actionable mask, set non-actionable to -inf
            result_clone[~mask] = -float('inf')
            return result_clone.max(1)[1].view(1, 1)
        else:
            # select random action from actionable actions (use first row if 2D)
            if mask.dim() == 2:
                valid = torch.nonzero(mask[0]).squeeze().tolist()
            else:
                valid = torch.nonzero(mask).squeeze().tolist()
            if isinstance(valid, int):
                valid = [valid]
            chosen_index = random.choice(valid)
            return torch.tensor([[chosen_index]], device=self.device, dtype=torch.long)
            
    def select_action_tuple(self, state, epsilon = 0.1):
        '''Select action as (x, y, mode) tuple'''
        action_index = self.select_action(state).item()
        cell_index = action_index // 2
        mode = 'left' if action_index % 2 == 0 else 'right'

        x = (cell_index % self.env.game.w) + 1        # 1-based coordinates
        y = (cell_index // self.env.game.w) + 1       # 1-based coordinates
        return (x, y, mode)
    
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # mask of non-final next states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)

        # concatenate next states that are not None (handle empty list)
        non_final_next_states_list = [s.to(self.device) for s in batch.next_state if s is not None]
        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.cat(non_final_next_states_list)
        else:
            non_final_next_states = torch.empty((0, self.n_observations), device=self.device)

        # move batches to our device
        state_batch = torch.cat([s.to(self.device) for s in batch.state])
        action_batch = torch.cat([a.to(self.device) for a in batch.action])
        reward_batch = torch.cat([r.to(self.device) for r in batch.reward]).to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states using the target network
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            if non_final_next_states.size(0) > 0:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

class DQNAgent(RL_agent):
    def __init__(self, env: MSEnv) -> None:
        super().__init__(env)
        
    def run_episode(self, difficulty: str = None, max_steps: int = 5000, delay: float = 0.0):
        """Run one episode using the DQN policy and train during the run.

        Returns a dict similar to other agents: keys `steps`, `reward`, `done`,
        `history`, and `final_state`, plus `random_clicks`.
        """
        # Start a new game and obtain initial board state (2D list)
        # Assume the difficulty is unchanged for now
        state = self.env.reset(difficulty=difficulty)
        self.initialize_network()  # reinitialize network for new difficulty
        w = self.env.game.w
        h = self.env.game.h

        steps = 0
        total_reward = 0
        history = []
        done = False

        # helper to convert board state (2D list) to tensor on device
        def state_to_tensor(s):
            flat = [int(item) for row in s for item in row]
            t = torch.tensor([flat], dtype=torch.float32, device=self.device)
            return t

        # initial tensor
        state_t = state_to_tensor(state)

        while steps < max_steps and not done:
            
            action_idx = int(self.select_action(state_t).item())
            
            # decode action index to (x,y,mode)
            cell_index = action_idx // 2
            mode = 'left' if (action_idx % 2) == 0 else 'right'
            x = (cell_index % w) + 1
            y = (cell_index // w) + 1

            # execute action in environment
            try:
                next_state, reward, done, info = self.env.step((x, y, mode))
            except Exception as e:
                # if execution failed, stop episode
                return {"steps": steps, "reward": total_reward, "done": True, "history": history, "final_state": state, "random_clicks": 0, "error": str(e)}

            # prepare tensors for replay memory
            action_tensor = torch.tensor([[action_idx]], dtype=torch.long, device=self.device)
            reward_tensor = torch.tensor([float(reward)], dtype=torch.float32, device=self.device)
            next_state_tensor = None if done else state_to_tensor(next_state)

            # push transition
            self.memory.push(state_t, action_tensor, next_state_tensor, reward_tensor)

            # record history
            history.append(((x, y, mode), reward, done))
            total_reward += reward
            steps += 1
            self.steps_done += 1

            # train step
            self.optimize_model()

            # soft update target network (polyak)
            with torch.no_grad():
                for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    tp.data.mul_(1.0 - TAU)
                    tp.data.add_(TAU * pp.data)

            # advance state
            state = next_state
            state_t = None if done else state_to_tensor(state)

            if delay and not done:
                import time
                time.sleep(delay)

        return {"steps": steps, "reward": total_reward, "done": bool(done), "history": history, "final_state": state, "random_clicks": "None", 'win': info.get('win', False)}

    def run_num_episodes(self, num_episodes: int, difficulty: str = None, max_steps: int = 100000, delay: float = 0.0, progress_update=None):
        """Run `num_episodes` episodes sequentially and call `progress_update(info)` after each.

        `progress_update` is an optional callable that receives a dict with keys:
        - `episode`: 1-based episode index
        - `length`: episode length (steps)
        - `win`: boolean indicating whether episode was won
        - `reward`: numeric reward from the episode

        Returns a list of the episode result dicts.
        """
        results = []
        for i in range(1, int(num_episodes) + 1):
            res = self.run_episode(difficulty=difficulty, max_steps=max_steps, delay=delay)
            results.append(res)

            # determine win flag
            win_flag = False
            try:
                if res.get('win', False) and res.get('reward', 0) > 0:
                    win_flag = True
            except Exception:
                win_flag = False

            info = {'episode': i, 'length': len(res.get('history', [])), 'win': bool(win_flag), 'reward': res.get('reward', 0)}
            try:
                if callable(progress_update):
                    progress_update(info)
            except Exception:
                pass

        return results

class Hybrid_Agent(RL_agent):
    def __init__(self, env: MSEnv, baseline_agent) -> None:
        super().__init__(env)
        self.baseline_agent = baseline_agent
        
    def run_episode(self, difficulty: str = None, max_steps: int = 5000, delay: float = 0.0):
                
        state = self.env.reset(difficulty)
        w = self.env.game.w
        h = self.env.game.h

        steps = 0
        total_reward = 0
        history = []
        done = False
        random_click = 0
        # helper to convert board state (2D list) to tensor on device
        def state_to_tensor(s):
            flat = [int(item) for row in s for item in row]
            t = torch.tensor([flat], dtype=torch.float32, device=self.device)
            return t

        # initial random left-click to start the board (do not record in memory)
        unrevealed = []
        for r in range(h):
            for c in range(w):
                try:
                    v = state[r][c]
                except Exception:
                    v = -1
                if v < 0:
                    unrevealed.append((c + 1, r + 1))
        if unrevealed:
            cx, cy = random.choice(unrevealed)
            try:
                state, reward, done, info = self.env.step((cx, cy, 'left'))
            except Exception as e:
                return {"steps": steps, "reward": total_reward, "done": True, "history": history, "final_state": state, "random_clicks": random_click, "error": str(e), "win": info.get('win', False)}
            history.append(((cx, cy, 'left'), reward, done))
            total_reward += reward
            steps += 1
            random_click += 1
            if delay and not done:
                import time
                time.sleep(delay)

        # Main loop: alternate baseline-rule actions and DQN actions
        state_t = state_to_tensor(state)

        while steps < max_steps and not done:
            # Apply baseline-agent deterministic actions repeatedly until none returned
            while True:
                try:
                    flag_actions, click_actions = self.baseline_agent.select_action(state)
                except Exception:
                    flag_actions, click_actions = ([], [])

                # if none returned, break to let DQN act
                if not flag_actions and not click_actions:
                    break

                # Execute flag actions (right clicks) first
                for (x, y, mode) in flag_actions:
                    try:
                        state, reward, done, info = self.env.step((x, y, mode))
                    except Exception as e:
                        return {"steps": steps, "reward": total_reward, "done": True, "history": history, "final_state": state, "random_clicks": random_click, "win": info.get('win', False), "error": str(e)}
                    history.append(((x, y, mode), reward, done))
                    total_reward += reward
                    steps += 1
                    if delay and not done:
                        import time
                        time.sleep(delay)
                    if done or steps >= max_steps:
                        break
                if done or steps >= max_steps:
                    break

                # Execute click actions (left clicks)
                for (x, y, mode) in click_actions:
                    try:
                        state, reward, done, info = self.env.step((x, y, mode))
                    except Exception as e:
                        return {"steps": steps, "reward": total_reward, "done": True, "history": history, "final_state": state, "random_clicks": random_click, "win": info.get('win', False), "error": str(e)}
                    history.append(((x, y, mode), reward, done))
                    total_reward += reward
                    steps += 1
                    if delay and not done:
                        import time
                        time.sleep(delay)
                    if done or steps >= max_steps:
                        break

                if done or steps >= max_steps:
                    break

                # loop to run baseline.select_action again

            if done or steps >= max_steps:
                break

            # Baseline produced no actions: let DQN decide next action
            try:
                # prepare tensors
                state_t = state_to_tensor(state)
                action_idx = int(self.select_action(state_t).item())
                # decode to (x,y,mode)
                cell_index = action_idx // 2
                mode = 'left' if (action_idx % 2) == 0 else 'right'
                x = (cell_index % w) + 1
                y = (cell_index // w) + 1

                # execute action in environment
                try:
                    next_state, reward, done, info = self.env.step((x, y, mode))
                except Exception as e:
                    return {"steps": steps, "reward": total_reward, "done": True, "history": history, "final_state": state, "random_clicks": random_click, "win": info.get('win', False), "error": str(e)}

                # prepare tensors for replay memory (only for DQN-decided actions)
                action_tensor = torch.tensor([[action_idx]], dtype=torch.long, device=self.device)
                reward_tensor = torch.tensor([float(reward)], dtype=torch.float32, device=self.device)
                next_state_tensor = None if done else state_to_tensor(next_state)

                # push transition to memory
                self.memory.push(state_t, action_tensor, next_state_tensor, reward_tensor)

                # record history and stats
                history.append(((x, y, mode), reward, done))
                # DQN-decided action; count as a random/dqn click
                random_click += 1
                total_reward += reward
                steps += 1
                self.steps_done += 1

                # train step and soft update
                self.optimize_model()
                with torch.no_grad():
                    for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                        tp.data.mul_(1.0 - TAU)
                        tp.data.add_(TAU * pp.data)

                # advance state
                state = next_state
                state_t = None if done else state_to_tensor(state)

                if delay and not done:
                    import time
                    time.sleep(delay)
            except Exception:
                # if select_action or training fails, stop safely
                break

        return {"steps": steps, "reward": total_reward, "done": bool(done), "history": history, "final_state": state, "random_clicks": random_click, "win": info.get('win', False)}
    
    def run_num_episodes(self, num_episodes: int, difficulty: str = None, max_steps: int = 100000, delay: float = 0.0, progress_update=None):
        """Run `num_episodes` episodes using the hybrid policy and call
        `progress_update(info)` after each episode (same contract as DQNAgent).

        Returns a list of per-episode result dicts.
        """
        results = []
        for i in range(1, int(num_episodes) + 1):
            res = self.run_episode(difficulty=difficulty, max_steps=max_steps, delay=delay)
            results.append(res)

            # determine win flag
            win_flag = False
            try:
                if res.get('win', False):
                    win_flag = True
            except Exception:
                win_flag = False

            info = {'episode': i, 'length': len(res.get('history', [])), 'win': bool(win_flag), 'reward': res.get('reward', 0), 'random_clicks': res.get('random_clicks', res.get('random_click', 0))}
            try:
                if callable(progress_update):
                    progress_update(info)
            except Exception:
                pass

        return results
        
        