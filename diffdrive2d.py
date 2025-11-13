import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class DifferentialDriveEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.map_size = 10.0  # meters
        self.dt = 0.1
        self.wheel_base = 0.5

        # Actions: left & right wheel velocities
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observations: (x, y, theta, dist_left, dist_front, dist_right)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Obstacles: list of rectangles [(x, y, w, h), ...]
        self.obstacles = [(2, 2, 2, 0.5), (6, 4, 0.5, 3), (3, 7, 4, 0.5)]

        self.goal = np.array([9.0, 9.0])
        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([1.0, 1.0, 0.0])  # x, y, theta
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        v_l, v_r = action
        v = (v_l + v_r) / 2.0
        omega = (v_r - v_l) / self.wheel_base

        x, y, theta = self.state
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += omega * self.dt
        self.state = np.array([x, y, theta])

        obs = self._get_obs()
        reward = -np.linalg.norm(self.goal - np.array([x, y]))
        terminated = (
            self._check_collision()
            or np.linalg.norm(self.goal - np.array([x, y])) < 0.5
        )
        info = {}
        return obs, reward, terminated, False, info

    def _get_obs(self):
        x, y, theta = self.state
        distances = self._ray_cast(x, y, theta)
        return np.array([x, y, theta] + distances)

    def _ray_cast(self, x, y, theta):
        # Very rough simulated distances to nearest obstacle
        dirs = [-np.pi / 4, 0, np.pi / 4]
        distances = []
        for d in dirs:
            ray_angle = theta + d
            dist = self._distance_to_wall(x, y, ray_angle)
            distances.append(dist)
        return distances

    def _distance_to_wall(self, x, y, angle, max_dist=5.0):
        # Sample points along the ray
        for r in np.linspace(0, max_dist, 100):
            px, py = x + r * np.cos(angle), y + r * np.sin(angle)
            if self._in_obstacle(px, py):
                return r
        return max_dist

    def _in_obstacle(self, x, y):
        for ox, oy, w, h in self.obstacles:
            if ox <= x <= ox + w and oy <= y <= oy + h:
                return True
        return x < 0 or y < 0 or x > self.map_size or y > self.map_size

    def _check_collision(self):
        x, y, _ = self.state
        return self._in_obstacle(x, y)

    def render(self):
        if self.render_mode != "human":
            return

        # Create the figure once
        if not hasattr(self, "fig"):
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(0, self.map_size)
            self.ax.set_ylim(0, self.map_size)

        self.ax.clear()
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)

        # Draw obstacles
        for ox, oy, w, h in self.obstacles:
            self.ax.add_patch(plt.Rectangle((ox, oy), w, h, color="gray"))

        # Draw goal and robot
        self.ax.plot(self.goal[0], self.goal[1], "go", markersize=10)
        x, y, theta = self.state
        self.ax.plot(x, y, "ro")
        self.ax.arrow(x, y, 0.5*np.cos(theta), 0.5*np.sin(theta), head_width=0.2)

        plt.pause(0.001)
        
env = DifferentialDriveEnv(render_mode="human")

obs, _ = env.reset()
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        break
plt.show()