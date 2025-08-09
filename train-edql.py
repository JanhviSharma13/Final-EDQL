"""
Main training script implementing DQL-PER per Patil et al. (IET ITS 2025).

Usage:
    python train_edql.py

Make sure your PYTHONPATH includes the directory and that SUMO (and TraCI)
are installed if you want SUMO-backed environments. Otherwise the code falls
back to a simple random test environment for smoke tests.
"""
import os
import random
import glob
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
from datetime import datetime

# Local env imports (wrappers)
from edql_cp2_env import EDQLCP2Env
from edql_rakabganj_env import EDQLRakabGanjEnv
from edql_safdarjung_env import EDQLSafdarjungEnv
from edql_chandnichowk_env import EDQLChandniChowkEnv

# ---------------- Q-Network ----------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ------------- Prioritized Replay Buffer -------------
class PrioritizedReplayBuffer:
    """Proportional PER (simple array-based). Uses importance-sampling weights with beta annealing."""
    def __init__(self, capacity=2000, alpha=0.6, beta_start=0.4, beta_frames=5000):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.frame = 1

    def add(self, transition, priority=None):
        if priority is None:
            priority = self.priorities.max() if len(self.buffer) > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def _get_probabilities(self):
        length = len(self.buffer)
        scaled = self.priorities[:length] ** self.alpha
        probs = scaled / (scaled.sum() + 1e-8)
        return probs

    def sample(self, batch_size):
        n = len(self.buffer)
        if n == 0:
            return [], [], [], [], [], [], []

        probs = self._get_probabilities()
        indices = np.random.choice(n, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Beta annealing
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / max(1, self.beta_frames)))
        self.frame += 1

        weights = (n * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
            indices,
            weights
        )

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            if idx < self.capacity:
                self.priorities[idx] = abs(p) + 1e-6

    def __len__(self):
        return len(self.buffer)

# ---------------- EDQL Agent (paper-accurate) ----------------
class EDQLAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.5,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=2000,
        batch_size=32,
        alpha_per=0.6,
        beta_start=0.4,
        beta_frames=5000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Networks
        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)

        self.q1_target = QNetwork(state_dim, action_dim)
        self.q2_target = QNetwork(state_dim, action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers (we keep lr param; paper used high alpha value—in practice using Adam)
        self.opt1 = optim.Adam(self.q1.parameters(), lr=lr)
        self.opt2 = optim.Adam(self.q2.parameters(), lr=lr)

        # PER buffer (paper: D=2000)
        self.replay = PrioritizedReplayBuffer(capacity=buffer_size, alpha=alpha_per, beta_start=beta_start, beta_frames=beta_frames)

        # Exploration
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        # bookkeeping
        self.learn_step = 0

    def select_action(self, state):
        # epsilon-greedy with argmax over (Q1 + Q2) per paper's Algorithm 1
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q1_vals = self.q1(state_t)
            q2_vals = self.q2(state_t)
            qsum = q1_vals + q2_vals
            return int(qsum.argmax(1).item())

    def store(self, state, action, reward, next_state, done):
        # Add transition; initial priority will be set inside buffer.add
        self.replay.add((state, action, reward, next_state, float(done)))

    def update_targets(self):
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, indices, weights = self.replay.sample(self.batch_size)

        # Decide which network to update (50% chance each) per paper
        update_q1 = random.random() < 0.5

        if update_q1:
            # action selection using q1 (main), evaluation using q2_target
            with torch.no_grad():
                next_actions = self.q1(next_states).argmax(1)
                next_q = self.q2_target(next_states).gather(1, next_actions.unsqueeze(1))
                target_q = rewards + (self.gamma * next_q * (1.0 - dones))

            current_q = self.q1(states).gather(1, actions.unsqueeze(1))
            td_errors = (target_q - current_q)

            loss = (weights * (td_errors ** 2)).mean()

            self.opt1.zero_grad()
            loss.backward()
            self.opt1.step()

        else:
            # update q2: selection via q2, evaluation via q1_target
            with torch.no_grad():
                next_actions = self.q2(next_states).argmax(1)
                next_q = self.q1_target(next_states).gather(1, next_actions.unsqueeze(1))
                target_q = rewards + (self.gamma * next_q * (1.0 - dones))

            current_q = self.q2(states).gather(1, actions.unsqueeze(1))
            td_errors = (target_q - current_q)

            loss = (weights * (td_errors ** 2)).mean()

            self.opt2.zero_grad()
            loss.backward()
            self.opt2.step()

        # update priorities in replay buffer using magnitude of TD errors
        td_vals = td_errors.detach().abs().squeeze().cpu().numpy()
        # If batch_size is 1, make array consistent
        if td_vals.ndim == 0:
            td_vals = np.array([float(td_vals)])
        self.replay.update_priorities(indices, td_vals)

        self.learn_step += 1
        return float(loss.item())

    def save(self, path):
        torch.save({
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        data = torch.load(path)
        self.q1.load_state_dict(data['q1'])
        self.q2.load_state_dict(data['q2'])
        self.q1_target.load_state_dict(data['q1'])
        self.q2_target.load_state_dict(data['q2'])
        self.epsilon = data.get('epsilon', self.epsilon)

# ---------------- Training loop ----------------
def train_area(env_class, env_args, agent: EDQLAgent, area_name, episodes=10, max_steps=1000,
               target_update_freq=50, save_freq=100, log_freq=10, auto_load=True):

    # auto-load latest model
    start_ep = 0
    if auto_load:
        pattern = f"edql_model_episode_*_{area_name.lower()}.pth"
        files = glob.glob(pattern)
        if files:
            def ep_from_name(x):
                try:
                    parts = x.split('_')
                    return int(parts[3])
                except Exception:
                    return -1
            latest = max(files, key=ep_from_name)
            try:
                ep = ep_from_name(latest)
                if ep >= 0:
                    agent.load(latest)
                    start_ep = ep
                    episodes += start_ep
                    print(f"Loaded checkpoint {latest}, resuming from episode {start_ep}")
            except Exception:
                pass

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_name = f"training_log_{area_name.lower()}_{timestamp}.csv"

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", csv_name)
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_reward', 'steps', 'epsilon', 'loss',
            'td_mean', 'td_std', 'td_min', 'td_max',
            'travel_time', 'route_length', 'avg_speed', 'success', 'route_changes'
        ])

        for ep in range(start_ep, episodes):
            env = env_class(**env_args)
            state, info = env.reset()
            total_reward = 0.0
            steps = 0
            losses = []
            route_changes = 0

            for step in range(max_steps):
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store(state, action, reward, next_state, done)

                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

                if info.get("route_changed", False):
                    route_changes += 1

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

            if ep % target_update_freq == 0:
                agent.update_targets()

            agent.decay_epsilon()

            try:
                recent = agent.replay.priorities[:len(agent.replay.buffer)]
                td_mean = float(np.mean(recent))
                td_std = float(np.std(recent))
                td_min = float(np.min(recent))
                td_max = float(np.max(recent))
            except Exception:
                td_mean = td_std = td_min = td_max = 0.0

            travel_time = info.get("travel_time", None)
            route_length = info.get("route_length", None)
            avg_speed = info.get("avg_speed", None)
            success = 1 if info.get("success", False) else 0

            writer.writerow([
                ep, total_reward, steps, agent.epsilon, float(np.mean(losses) if losses else 0.0),
                td_mean, td_std, td_min, td_max,
                travel_time, route_length, avg_speed, success, route_changes
            ])

            if ep % log_freq == 0:
                print(f"{area_name} Ep {ep}/{episodes} "
                      f"R:{total_reward:.2f} steps:{steps} eps:{agent.epsilon:.3f} "
                      f"loss:{np.mean(losses) if losses else 0.0:.4f}")

            if ep % save_freq == 0:
                chk = f"edql_model_episode_{ep}_{area_name.lower()}.pth"
                agent.save(chk)

            env.close()

    agent.save(f"edql_model_episode_{episodes-1}_{area_name.lower()}.pth")
    print(f"Finished training {area_name}. Log saved to {log_path}")
    return log_path
if __name__ == '__main__':
    areas = [
        {"name": "CP2", "env": EDQLCP2Env, "args": {"net_file": "cp_cleaned.net.xml", "route_file": "cp2_dynamic.rou.xml", "use_gui": False, "max_steps": 2000}},
        {"name": "RakabGanj", "env": EDQLRakabGanjEnv, "args": {"net_file": "rakabganj_cleaned.net.xml", "route_file": "rakabganj_dynamic.rou.xml", "use_gui": False, "max_steps": 2000}},
        {"name": "Safdarjung", "env": EDQLSafdarjungEnv, "args": {"net_file": "safdarjung_cleaned.net.xml", "route_file": "safdarjung_dynamic.rou.xml", "use_gui": False, "max_steps": 2000}},
        {"name": "ChandniChowk", "env": EDQLChandniChowkEnv, "args": {"net_file": "chandnichowk_cleaned.net.xml", "route_file": "chandnichowk_dynamic.rou.xml", "use_gui": False, "max_steps": 2000}}
    ]

    for area in areas:
        print(f"Starting training for {area['name']}")
        agent = EDQLAgent(
            state_dim=4, action_dim=8, lr=0.001, gamma=0.95,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
            buffer_size=2000, batch_size=32,
            alpha_per=0.6, beta_start=0.4, beta_frames=5000
        )

        train_area(
            area['env'], area['args'], agent, area['name'],
            episodes=10,  # lowered for trial
            max_steps=2000,
            target_update_freq=50, save_freq=100, log_freq=1, auto_load=False
        )
