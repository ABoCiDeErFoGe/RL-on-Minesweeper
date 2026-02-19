from collections import namedtuple, deque
import math
import random
from torch import device, nn
from torch import optim
import torch
import torch.nn.functional as F

from Game import MSEnv
from agent_interface import agent_interface

from utils import get_unrevealed_cells

from config import *

Transition = namedtuple('Transition',
            ('state','action','next_state','reward','mask'))


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
    def __init__(self, in_channels, height, width, n_actions):
        super(DQN, self).__init__()
        # simple conv stack that preserves spatial dims (padding=1)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # compute flattened size after convs (padding keeps h,w)
        conv_output_size = 64 * height * width

        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x expected shape: [batch, channels=1, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class RL_agent(agent_interface):
    def __init__(self, env: MSEnv) -> None:
        self.env = env
        self.initialize_network()
        
    # reinitialize DQN based on the network
    def initialize_network(self):
        self.w = self.env.game.w
        self.h = self.env.game.h
        self.n_actions = self.w * self.h  # each cell
        self.n_observations = self.w * self.h  # flattened board state
        # padding around board (border); agent must not be able to click border
        self.pad = 1
        self.padded_h = self.h + 2 * self.pad
        self.padded_w = self.w + 2 * self.pad
        # Policy and target are CNNs taking (batch, 3, H_padded, W_padded)
        self.policy_net = DQN(3, self.padded_h, self.padded_w, self.n_actions)
        self.target_net = DQN(3, self.padded_h, self.padded_w, self.n_actions)
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

        # obtain actionable mask (Game.get_actionable_mask may return one entry per cell)
        actionmask = self.env.game.get_actionable_mask()
        tensor_actionmask = torch.tensor(actionmask, dtype=torch.bool, device=self.device)

        # If mask was accidentally returned as per-action (2 entries per cell),
        # convert it to one-per-cell by OR-ing left/right entries.
        if tensor_actionmask.numel() == (self.n_actions * 2):
            tensor_actionmask = tensor_actionmask.view(-1, 2).any(1)

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
        # network outputs one Q per cell; default to left click
        cell_index = action_index
        mode = 'left'

        x = (cell_index % self.env.game.w) + 1        # 1-based coordinates
        y = (cell_index // self.env.game.w) + 1       # 1-based coordinates
        return (x, y, mode)

    def preprocess_state(self, state, pad: int = None):
        """Preprocess a 2D game state into 3 channels with padding.

        Returns a Python list of shape (3, padded_h, padded_w).

        Channels:
          - ch0: revealed numbers only (normalized to [0,1] by dividing by 8). Hidden/flagged -> 0
          - ch1: revealed mask (1 for any revealed cell, including zeros)
          - ch2: flagged mask (1 for flagged cells)

        Padding around the board is filled with zeros; those border positions
        are therefore non-actionable by mask logic.
        """
        import numpy as _np
        pad = self.pad if pad is None else pad
        h = self.h
        w = self.w
        ph = h + 2 * pad
        pw = w + 2 * pad

        ch0 = _np.zeros((ph, pw), dtype=_np.float32)
        ch1 = _np.zeros((ph, pw), dtype=_np.float32)
        ch2 = _np.zeros((ph, pw), dtype=_np.float32)

        for r in range(h):
            for c in range(w):
                try:
                    val = state[r][c]
                except Exception:
                    val = -1
                rr = r + pad
                cc = c + pad
                # revealed numbers: 0-8
                if isinstance(val, int) and val >= 0:
                    ch1[rr, cc] = 1.0
                    # normalize number to [0,1]
                    ch0[rr, cc] = float(val) / 8.0
                # flagged
                elif isinstance(val, int) and val in (-2, -11):
                    ch2[rr, cc] = 1.0

        stacked = _np.stack([ch0, ch1, ch2], axis=0)
        return stacked.tolist()
    
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch  = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=self.device,
            dtype=torch.bool
        )

        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        ).to(self.device)

        # gather masks aligned with non-final states
        next_masks = torch.stack([
            m for s, m in zip(batch.next_state, batch.mask) if s is not None
        ]).to(self.device)

        # Q(s,a)
        q_sa = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)

        with torch.no_grad():
            if non_final_next_states.size(0) > 0:

                # Double DQN
                next_policy_q = self.policy_net(non_final_next_states)

                # mask invalid
                next_policy_q[~next_masks] = -1e9

                next_actions = next_policy_q.argmax(1)

                next_target_q = self.target_net(non_final_next_states)

                next_target_q[~next_masks] = -1e9

                next_vals = next_target_q.gather(
                    1,
                    next_actions.unsqueeze(1)
                ).squeeze()

                next_state_values[non_final_mask] = next_vals

        expected = reward_batch + GAMMA * next_state_values

        loss = F.smooth_l1_loss(q_sa, expected.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
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
        
        w = self.env.game.w
        state = self.env.reset(difficulty=difficulty)
        
        if (w != self.env.game.w):
            self.initialize_network()  # reinitialize network for new difficulty
        w = self.env.game.w
        h = self.env.game.h

        steps = 0
        total_reward = 0
        history = []
        done = False

        # helper to convert board state (2D list) to tensor on device
        def state_to_tensor(s):
            chs = self.preprocess_state(s)
            t = torch.tensor([chs], dtype=torch.float32, device=self.device)
            return t

        # initial tensor
        state_t = state_to_tensor(state)

        while steps < max_steps and not done:
            action_idx = int(self.select_action(state_t).item())

            # network outputs one value per cell; action_idx is the cell index
            cell_index = action_idx
            mode = 'left'
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

            next_mask = self.env.game.get_actionable_mask()
            next_mask = torch.tensor(next_mask, dtype=torch.bool)

            self.memory.push(
                state_t,
                action_tensor,
                next_state_tensor,
                reward_tensor,
                next_mask
            )
            
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

    # run_num_episodes inherited from agent_interface

class Hybrid_Agent(RL_agent):
    def __init__(self, env: MSEnv, baseline_agent) -> None:
        super().__init__(env)
        self.baseline_agent = baseline_agent
        
    def run_episode(self, difficulty: str = None, max_steps: int = 5000, delay: float = 0.0):
                
        w = self.env.game.w
        state = self.env.reset(difficulty=difficulty)
        
        if (w != self.env.game.w):
            self.initialize_network()  # reinitialize network for new difficulty
        w = self.env.game.w
        h = self.env.game.h

        steps = 0
        total_reward = 0
        history = []
        done = False
        random_click = 0
        # helper to convert board state (2D list) to tensor on device
        def state_to_tensor(s):
            chs = self.preprocess_state(s)
            t = torch.tensor([chs], dtype=torch.float32, device=self.device)
            return t

        # initial random left-click to start the board (do not record in memory)
        unrevealed = get_unrevealed_cells(state)
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
                # decode to (x,y,mode) — network outputs per-cell Qs
                cell_index = action_idx
                mode = 'left'
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

                next_mask = self.env.game.get_actionable_mask()
                next_mask = torch.tensor(next_mask, dtype=torch.bool)

                self.memory.push(
                    state_t,
                    action_tensor,
                    next_state_tensor,
                    reward_tensor,
                    next_mask
                )
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
    
    # run_num_episodes inherited from agent_interface
        
        