from collections import namedtuple, deque
import math
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from Game import MSEnv
from agent_interface import agent_interface

from utils import get_unrevealed_cells

from config import *

from datetime import datetime 

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
    def __init__(self, env: MSEnv, hyperparams: dict = None) -> None:
        self.env = env
        # store relevant config values from config.py into the agent (defaults)
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TAU = TAU
        self.LR = LR

        # if hyperparams supplied, override defaults
        if hyperparams is not None and isinstance(hyperparams, dict):
            self.BATCH_SIZE = hyperparams.get("BATCH_SIZE", self.BATCH_SIZE)
            self.GAMMA = hyperparams.get("GAMMA", self.GAMMA)
            self.EPS_START = hyperparams.get("EPS_START", self.EPS_START)
            self.EPS_END = hyperparams.get("EPS_END", self.EPS_END)
            self.EPS_DECAY = hyperparams.get("EPS_DECAY", self.EPS_DECAY)
            self.TAU = hyperparams.get("TAU", self.TAU)
            self.LR = hyperparams.get("LR", self.LR)

        # a dict view of the config (constructed from agent attributes)
        self.config_dict = self.get_config_dict()

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
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        # epsilon decay parameters
        self.steps_done = 0
        # active network boolean: False -> policy (default), True -> target
        self.use_target_active = False

    def get_config_dict(self):
        """Return a dictionary of agent configuration values."""
        return {
            "BATCH_SIZE": self.BATCH_SIZE,
            "GAMMA": self.GAMMA,
            "EPS_START": self.EPS_START,
            "EPS_END": self.EPS_END,
            "EPS_DECAY": self.EPS_DECAY,
            "TAU": self.TAU,
            "LR": self.LR,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model + optimizer state and agent config to `path`."""
        checkpoint = {
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "difficulty": self.env.game.difficulty,
            "config": self.get_config_dict(),
            "agent_class": self.__class__.__name__,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, map_location=None) -> None:
        """Load checkpoint from `path` and restore networks/optimizer/steps.

        If the saved config differs from the current agent config, a warning
        is printed but loading proceeds.
        """
        ckpt = torch.load(path, map_location=map_location)
        
        # check if class matches (not strictly required, but likely a sign of user error if not)
        saved_class = ckpt.get("agent_class")
        if saved_class and saved_class != self.__class__.__name__:
            raise ValueError(f"Checkpoint agent class '{saved_class}' does not match current agent class '{self.__class__.__name__}'")
        
        if "policy_state" in ckpt:
            self.policy_net.load_state_dict(ckpt["policy_state"])
        if "target_state" in ckpt:
            self.target_net.load_state_dict(ckpt["target_state"])
        if "optimizer_state" in ckpt and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception:
                print("Warning: failed to load optimizer state; continuing")
        self.steps_done = ckpt.get("steps_done", self.steps_done)
        saved_cfg = ckpt.get("config")
        if saved_cfg and isinstance(saved_cfg, dict):
            self.BATCH_SIZE = saved_cfg.get("BATCH_SIZE", self.BATCH_SIZE)
            self.GAMMA = saved_cfg.get("GAMMA", self.GAMMA)
            self.EPS_START = saved_cfg.get("EPS_START", self.EPS_START)
            self.EPS_END = saved_cfg.get("EPS_END", self.EPS_END)
            self.EPS_DECAY = saved_cfg.get("EPS_DECAY", self.EPS_DECAY)
            self.TAU = saved_cfg.get("TAU", self.TAU)
            self.LR = saved_cfg.get("LR", self.LR)
            self.env.reset(saved_cfg.get("difficulty", None))
        else:
            print("Warning: no config found in checkpoint; using current agent config") 
        # recreate environment with saved difficulty if available
        saved_diff = ckpt.get("difficulty")
        if saved_diff is not None:
            self.env.reset(saved_diff)

    def predict_net_q(self, state, use_target: bool = None):
        """Run a forward pass on the selected network and return Q-values.

        Args:
            state: preprocessed state tensor (may be on CPU or device)
            use_target: if True use `self.target_net`, False use `self.policy_net`.
                        If None, falls back to `self.use_target_active`.

        Returns:
            Tensor of Q-values (no masking applied).
        """
        if use_target is None:
            use_target = bool(getattr(self, 'use_target_active', False))

        net = self.target_net if use_target else self.policy_net
        with torch.no_grad():
            return net(state.to(self.device))  # -> torch.Tensor

    def action_mask_tensor(self):
        """Return the actionable mask as a boolean tensor on the agent device.

        This normalizes masks that may be encoded as two entries per cell
        (e.g. left/right) into a single boolean per cell by OR-ing pairs.
        """
        actionmask = self.env.game.get_actionable_mask()
        tensor_actionmask = torch.tensor(actionmask, dtype=torch.bool, device=self.device)
        if tensor_actionmask.numel() == (self.n_actions * 2):
            tensor_actionmask = tensor_actionmask.view(-1, 2).any(1)
        return tensor_actionmask

    def select_action(self, state, use_target: bool = None):
        """Select an action index tensor using the policy or target network.

        Args:
            state: preprocessed state tensor (batch dim expected)
            use_target: if True, use `self.target_net` and act deterministically
                        (greedy). If False, use policy net with epsilon-greedy.

        Returns:
            torch.LongTensor shaped (1,1) containing the chosen action index.
        """
        # determine network choice from flag or active boolean
        if use_target is None:
            use_target = bool(getattr(self, 'use_target_active', False))

        # run the chosen network and obtain Q-values
        result = self.predict_net_q(state, use_target=use_target)

        # obtain actionable mask and ensure shape matches the network output
        tensor_actionmask = self.action_mask_tensor()
        # normalize mask shape to match network output (batch dim)
        if result.dim() == 2 and tensor_actionmask.dim() == 1:
            mask = tensor_actionmask.unsqueeze(0)
        else:
            mask = tensor_actionmask

        # If using target network, act greedily (deterministic)
        if use_target:
            res_clone = result.clone()
            res_clone[~mask] = -float('inf')
            return res_clone.max(1)[1].view(1, 1)

        # otherwise use epsilon-greedy on policy net
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)

        if sample > eps_threshold:
            result_clone = result.clone()
            # apply actionable mask, set non-actionable to -inf
            result_clone[~mask] = -float('inf')
            return result_clone.max(1)[1].view(1, 1)
        else:
            # select random action from actionable actions (use first row if 2D)
            if mask.dim() == 2:
                valid_idx = torch.nonzero(mask[0], as_tuple=False).view(-1).tolist()
            else:
                valid_idx = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
            if not valid_idx:
                raise RuntimeError("No valid actions available for random selection")
            chosen_index = random.choice(valid_idx)
            return torch.tensor([[chosen_index]], device=self.device, dtype=torch.long)
            
    def select_action_tuple(self, state, epsilon = 0.1, use_target: bool = None):
        '''Select action as (x, y, mode) tuple'''
        action_index = self.select_action(state, use_target=use_target).item()
        # network outputs one Q per cell; default to left click
        cell_index = action_index
        mode = 'left'

        x = (cell_index % self.env.game.w) + 1        # 1-based coordinates
        y = (cell_index // self.env.game.w) + 1       # 1-based coordinates
        return (x, y, mode)

    def set_use_target(self, flag: bool):
        """Set whether action selection should default to the target network.

        Args:
            flag: True to use the target network by default, False to use policy.
        """
        try:
            self.use_target_active = bool(flag)
        except Exception:
            raise ValueError('flag must be convertible to bool')

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
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch  = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=self.device,
            dtype=torch.bool
        )

        # Safely construct non-final next states and masks (handle empty lists)
        next_state_list = [s for s in batch.next_state if s is not None]
        if len(next_state_list) > 0:
            non_final_next_states = torch.cat(next_state_list).to(self.device)
        else:
            non_final_next_states = torch.empty(0, *state_batch.shape[1:], device=self.device)

        # gather masks aligned with non-final states
        next_masks_list = [m for s, m in zip(batch.next_state, batch.mask) if s is not None]
        if len(next_masks_list) > 0:
            next_masks = torch.stack(next_masks_list).to(self.device)
        else:
            next_masks = torch.empty(0, dtype=torch.bool, device=self.device)

        # Q(s,a)
        q_sa = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)

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

        expected = reward_batch + self.GAMMA * next_state_values

        loss = F.smooth_l1_loss(q_sa, expected.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

class DQNAgent(RL_agent):
    def __init__(self, env: MSEnv, hyperparams: dict = None) -> None:
        super().__init__(env, hyperparams=hyperparams)
        
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
        current_gamma = self.GAMMA

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
                reward = reward * current_gamma
                current_gamma *= self.GAMMA
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
                    tp.data.mul_(1.0 - self.TAU)
                    tp.data.add_(self.TAU * pp.data)

            # advance state
            state = next_state
            state_t = None if done else state_to_tensor(state)

            if delay and not done:
                import time
                time.sleep(delay)

        return {"steps": steps, "reward": total_reward, "done": bool(done), "history": history, "final_state": state, "random_clicks": "None", 'win': info.get('win', False)}

    # Expose checkpoint helpers on the concrete agent class
    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint (policy, target, optimizer, steps, config)."""
        super().save_checkpoint(path + "_pure_rl" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pth")

    def load_checkpoint(self, path: str, map_location=None) -> None:
        """Load agent checkpoint and restore networks/optimizer/steps."""
        super().load_checkpoint(path, map_location=map_location)
    
class Hybrid_Agent(RL_agent):
    def __init__(self, env: MSEnv, baseline_agent, hyperparams: dict = None) -> None:
        super().__init__(env, hyperparams=hyperparams)
        self.baseline_agent = baseline_agent
        
    def run_episode(self, difficulty: str = None, max_steps: int = 5000, delay: float = 0.0):
                
        w = self.env.game.w
        h = self.env.game.h
        state = self.env.reset(difficulty=difficulty)
        
        if (w != self.env.game.w or h != self.env.game.h):
            self.initialize_network()  # reinitialize network for new difficulty
        w = self.env.game.w
        h = self.env.game.h

        steps = 0
        total_reward = 0
        history = []
        done = False
        random_click = 0
        current_gamma = self.GAMMA
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
                    reward = reward * current_gamma
                    current_gamma *= self.GAMMA
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
                        tp.data.mul_(1.0 - self.TAU)
                        tp.data.add_(self.TAU * pp.data)

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

    # Expose checkpoint helpers on the concrete agent class
    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint (policy, target, optimizer, steps, config)."""
        super().save_checkpoint(path + "_hybrid_" + datetime.now().strftime("%Y%m%d_%H%M%S")+".pth")

    def load_checkpoint(self, path: str, map_location=None) -> None:
        """Load agent checkpoint and restore networks/optimizer/steps."""
        super().load_checkpoint(path, map_location=map_location)
        
        