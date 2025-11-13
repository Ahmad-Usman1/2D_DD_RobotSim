"""
dqn_dd_robot.py

A single-file example that:
- Defines a simple continuous differential-drive env but with DISCRETE actions
  (so we can train a DQN).
- Implements a simple DQN (PyTorch), replay buffer, training loop, and test loop.
- Visualizes with matplotlib using a single persistent figure (no window spam).

Run:
    python dqn_dd_robot.py train   # to train
    python dqn_dd_robot.py test    # to test / visualize trained agent
"""

import argparse
import math
import random
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import os
import time


# -------------------------
# Environment
# -------------------------
class DifferentialDriveEnv(gym.Env):
    """
    Continuous state (x,y,theta) + 3 ray distances
    Discrete action: small set of wheel commands mapped to (v_l, v_r)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.map_size = 10.0
        self.dt = 0.1
        self.wheel_base = 0.5
        self.prev_dist_to_goal = None
        # Discrete actions: [forward, left_turn, right_turn, backward, stop]
        self.action_list = [
            (1.0, 1.0),  # forward
            (-0.2, 0.6),  # turn left (left wheel slightly back, right forward)
            (0.6, -0.2),  # turn right
            (-0.6, -0.6),  # backward
            (0.0, 0.0),  # stop
        ]
        self.action_space = spaces.Discrete(len(self.action_list))

        # Observations: x, y, sin(theta), cos(theta), dist_left, dist_front, dist_right, distance_to_goal
        obs_low = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array(
            [self.map_size, self.map_size, 1.0, 1.0, 5.0, 5.0, 5.0, 20.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Obstacles as rectangles: (x, y, w, h)
        self.obstacles = [(2, 2, 2, 0.5), (6, 4, 0.5, 3), (3, 7, 4, 0.5)]
        self.goal = np.array([9.0, 9.0])
        self.render_mode = render_mode
        self._max_episode_steps = 500
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # start near (1,1) with small noise
        self.state = np.array(
            [
                1.0 + self.np_random.uniform(-0.1, 0.1),
                1.0 + self.np_random.uniform(-0.1, 0.1),
                0.0 + self.np_random.uniform(-0.2, 0.2),
            ]
        )
        self.steps = 0
        obs = self._get_obs()
        x = obs[0]
        y = obs[1]
        self.prev_dist_to_goal = np.linalg.norm(self.goal - np.array([x, y]))
        return obs, {}

    def step(self, action):
        # action is discrete index -> map to wheel velocities
        v_l, v_r = self.action_list[int(action)]
        v = (v_l + v_r) / 2.0
        omega = (v_r - v_l) / self.wheel_base

        x, y, theta = self.state
        x += v * math.cos(theta) * self.dt
        y += v * math.sin(theta) * self.dt
        theta = (theta + omega * self.dt) % (2 * math.pi)
        self.state = np.array([x, y, theta])

        self.steps += 1

        obs = self._get_obs()
        dist_to_goal = np.linalg.norm(self.goal - np.array([x, y]))
        delta = self.prev_dist_to_goal - dist_to_goal

        # reward: dense reward pushing toward goal, big penalty for collision, small step penalty
        reward = delta * 2
        if delta < -0.10:  # only punish if moved away more than 5 cm
            reward += delta * 3.0  # small negative
        reward -= 1.00 + self.steps*0.01 # living penalty to encourage speed
        if self._check_collision():
            reward -= 10.0
            terminated = True
            truncated = False
        elif dist_to_goal < 0.5:
            reward += 200.0
            terminated = True
            truncated = False
        elif self.steps >= self._max_episode_steps:
            terminated = False
            truncated = True
        else:
            terminated = False
            truncated = False

        info = {"distance_to_goal": dist_to_goal}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        x, y, theta = self.state
        distances = self._ray_cast(x, y, theta)
        dist_to_goal = np.linalg.norm(self.goal - np.array([x, y]))
        obs = np.array(
            [
                x,
                y,
                math.sin(theta),
                math.cos(theta),
                distances[0],
                distances[1],
                distances[2],
                dist_to_goal,
            ],
            dtype=np.float32,
        )
        return obs

    def _ray_cast(self, x, y, theta):
        dirs = [-math.pi / 4, 0.0, math.pi / 4]
        distances = []
        for d in dirs:
            ray_angle = theta + d
            dist = self._distance_to_wall(x, y, ray_angle)
            distances.append(dist)
        return distances

    def _distance_to_wall(self, x, y, angle, max_dist=5.0):
        # sample along ray
        for r in np.linspace(0, max_dist, 150):
            px = x + r * math.cos(angle)
            py = y + r * math.sin(angle)
            if self._in_obstacle(px, py):
                return r
        return max_dist

    def _in_obstacle(self, x, y):
        # off-map is considered collision
        if x < 0 or y < 0 or x > self.map_size or y > self.map_size:
            return True
        for ox, oy, w, h in self.obstacles:
            if ox <= x <= ox + w and oy <= y <= oy + h:
                return True
        return False

    def _check_collision(self):
        x, y, _ = self.state
        return self._in_obstacle(x, y)

    def render(self):
        if self.render_mode != "human":
            return

        # create figure once
        if not hasattr(self, "fig"):
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.ion()
        self.ax.clear()
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect("equal")

        # obstacles
        for ox, oy, w, h in self.obstacles:
            self.ax.add_patch(plt.Rectangle((ox, oy), w, h, color="gray"))

        # goal
        self.ax.plot(self.goal[0], self.goal[1], "go", markersize=10)

        # robot
        x, y, theta = self.state
        self.ax.plot(x, y, "ro")
        self.ax.arrow(
            x, y, 0.4 * math.cos(theta), 0.4 * math.sin(theta), head_width=0.15
        )

        # sensor rays for visualization
        distances = self._ray_cast(x, y, theta)
        dirs = [-math.pi / 4, 0.0, math.pi / 4]
        for d, dist in zip(dirs, distances):
            ang = theta + d
            ex = x + dist * math.cos(ang)
            ey = y + dist * math.sin(ang)
            self.ax.plot([x, ex], [y, ey], linestyle="--")

        plt.title(f"Step {self.steps}")
        plt.pause(0.001)

    def close(self):
        if hasattr(self, "fig"):
            plt.close(self.fig)


# -------------------------
# Replay Buffer
# -------------------------
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # convert to tensors later
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# -------------------------
# Q-Network (simple MLP)
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Utilities: epsilon schedule etc.
# -------------------------
def epsilon_by_frame(frame_idx, eps_start=1.0, eps_final=0.02, eps_decay=20000):
    return eps_final + (eps_start - eps_final) * math.exp(-1.0 * frame_idx / eps_decay)


# -------------------------
# Training loop
# -------------------------
def train(
    env,
    device,
    num_frames=200000,
    replay_initial=1000,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    target_update=1000,
    save_path="dqn_robot.pth",
):
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = QNetwork(obs_dim, n_actions).to(device)
    target_net = QNetwork(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(100000)

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)
    frame_idx = 0
    episode_rewards = []
    episode_reward = 0.0

    while frame_idx < num_frames:
        eps = epsilon_by_frame(frame_idx)
        # select action
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                qvals = policy_net(state.unsqueeze(0))
                action = int(torch.argmax(qvals, dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)

        if frame_idx % 50 == 0:
            env.render()

        done = terminated or truncated
        next_state = torch.tensor(next_obs, dtype=torch.float32).to(device)
        replay.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)

        state = next_state
        episode_reward += reward
        frame_idx += 1

        # episode end handling
        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            s, _ = env.reset()
            state = torch.tensor(s, dtype=torch.float32).to(device)

        # learn
        if len(replay) > replay_initial:
            transitions = replay.sample(batch_size)
            states = torch.tensor(np.vstack(transitions.state), dtype=torch.float32).to(
                device
            )
            actions = (
                torch.tensor(transitions.action, dtype=torch.long)
                .unsqueeze(1)
                .to(device)
            )
            rewards = (
                torch.tensor(transitions.reward, dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )
            next_states = torch.tensor(
                np.vstack(transitions.next_state), dtype=torch.float32
            ).to(device)
            dones = (
                torch.tensor(transitions.done, dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )

            q_values = policy_net(states).gather(1, actions)
            with torch.no_grad():
                q_next = target_net(next_states).max(1)[0].unsqueeze(1)
                q_target = rewards + gamma * q_next * (1 - dones)

            loss = nn.functional.mse_loss(q_values, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # target network update
        if frame_idx % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # basic logging
        if frame_idx % 2000 == 0:
            mean_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
            print(
                f"Frame: {frame_idx}  eps: {eps:.3f}  mean_reward(last20 eps): {mean_reward:.3f}"
            )

    # save model
    torch.save(
        {
            "policy_state_dict": policy_net.state_dict(),
        },
        save_path,
    )
    print("Training finished. Model saved to", save_path)
    return policy_net


# -------------------------
# Test / Eval loop
# -------------------------
def evaluate(env, device, model_path="dqn_robot.pth", episodes=5, render=True):
    checkpoint = torch.load(model_path, map_location=device)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy_net = QNetwork(obs_dim, n_actions).to(device)
    policy_net.load_state_dict(checkpoint["policy_state_dict"])
    policy_net.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        total_rew = 0.0
        done = False
        steps = 0
        while not done and steps < 1000:
            if render:
                env.render()
            state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = int(torch.argmax(policy_net(state_t), dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_rew += reward
            steps += 1
        print(f"Episode {ep+1} reward: {total_rew:.2f} steps: {steps}")
    env.close()


# -------------------------
# Main / CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"], help="train or test")
    parser.add_argument(
        "--frames", type=int, default=120000, help="number of training frames"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dqn_robot.pth",
        help="path for saving/loading model",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="eval episodes (test mode)"
    )
    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = DifferentialDriveEnv(render_mode="human")

    if args.mode == "train":
        train(env, device, num_frames=args.frames, save_path=args.model)
    else:
        evaluate(
            env, device, model_path=args.model, episodes=args.episodes, render=True
        )


if __name__ == "__main__":
    main()
