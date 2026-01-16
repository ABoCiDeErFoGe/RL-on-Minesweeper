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
    
class DQNAgent:
    def __init__(self, env: MSEnv) -> None:
        self.env = env
        self.n_actions = env.game.w * env.game.h * 2  # left/right for each cell
        self.n_observations = env.game.w * env.game.h  # flattened board state
        self.policy_net = DQN(self.n_observations, self.n_actions)
        self.target_net = DQN(self.n_observations, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
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

        if sample > eps_threshold:
            with torch.no_grad():
                result = self.policy_net(state)
                result_clone = result.clone()
                # apply actionable mask, set non-actionable to -inf
                actionmask = torch.tensor(self.env.game.get_actionable_mask(), dtype=torch.bool)
                result_clone[~actionmask] = -float('inf')
                return result_clone.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)
            
    def select_action_tuple(self, state, epsilon = 0.1):
        '''Select action as (x, y, mode) tuple'''
        action_index = self.select_action(state, epsilon).item()
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
        