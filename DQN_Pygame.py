import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import random
from collections import deque
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math

# Hyperparameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
MAP_WIDTH = 20
MAP_HEIGHT = 15
CELL_SIZE = 40
NUM_RAYS = 10
RAY_LENGTH = 500
ROBOT_RADIUS = 15

# DQN Hyperparameters
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

# Training parameters
NUM_EPISODES = 500
MAX_STEPS = 300
VISUALIZE_TRAINING = False  # Toggle visualization during training
VISUALIZE_INTERVAL = 50  # Visualize every N episodes when enabled


class Environment:
    def __init__(self, domain_randomization=True):
        self.width = MAP_WIDTH
        self.height = MAP_HEIGHT
        self.cell_size = CELL_SIZE
        self.domain_randomization = domain_randomization
        self.reset()
    
    def generate_map(self):
        """Generate a random map with obstacles"""
        self.obstacles = []
        num_obstacles = random.randint(5, 15)
        
        for _ in range(num_obstacles):
            w = random.randint(1, 4)
            h = random.randint(1, 4)
            x = random.randint(1, self.width - w - 1)
            y = random.randint(1, self.height - h - 1)
            self.obstacles.append((x, y, w, h))
        
        # Add border walls
        self.obstacles.append((0, 0, self.width, 1))  # Top
        self.obstacles.append((0, 0, 1, self.height))  # Left
        self.obstacles.append((0, self.height - 1, self.width, 1))  # Bottom
        self.obstacles.append((self.width - 1, 0, 1, self.height))  # Right
        
        # Domain randomization - visual parameters
        if self.domain_randomization:
            self.wall_color = (
                random.randint(50, 150),
                random.randint(50, 150),
                random.randint(50, 150)
            )
        else:
            self.wall_color = (100, 100, 100)
    
    def find_valid_position(self):
        """Find a random position not inside obstacles"""
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.uniform(2, (self.width - 2) * self.cell_size)
            y = random.uniform(2, (self.height - 2) * self.cell_size)
            
            if not self.check_collision(x, y):
                return x, y
        
        return self.width * self.cell_size / 2, self.height * self.cell_size / 2
    
    def reset(self):
        """Reset environment with new random map"""
        self.generate_map()
        
        # Initialize agent position
        self.agent_x, self.agent_y = self.find_valid_position()
        self.agent_angle = random.uniform(0, 2 * math.pi)
        
        # Initialize goal position (far from agent)
        while True:
            self.goal_x, self.goal_y = self.find_valid_position()
            dist = math.sqrt((self.goal_x - self.agent_x)**2 + (self.goal_y - self.agent_y)**2)
            if dist > 200:  # Ensure goal is far enough
                break
        
        self.steps = 0
        self.prev_distance = self.get_distance_to_goal()
        return self.get_observation()
    
    def get_distance_to_goal(self):
        """Calculate Euclidean distance to goal"""
        return math.sqrt((self.goal_x - self.agent_x)**2 + (self.goal_y - self.agent_y)**2)
    
    def check_collision(self, x, y, radius=ROBOT_RADIUS):
        """Check if position collides with any obstacle"""
        for ox, oy, ow, oh in self.obstacles:
            ox_px = ox * self.cell_size
            oy_px = oy * self.cell_size
            ow_px = ow * self.cell_size
            oh_px = oh * self.cell_size
            
            # Check if circle intersects rectangle
            closest_x = max(ox_px, min(x, ox_px + ow_px))
            closest_y = max(oy_px, min(y, oy_px + oh_px))
            
            dist = math.sqrt((x - closest_x)**2 + (y - closest_y)**2)
            if dist < radius:
                return True
        return False
    
    def cast_ray(self, angle):
        """Cast a single ray and return normalized distance to obstacle"""
        x, y = self.agent_x, self.agent_y
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        for step in range(1, RAY_LENGTH):
            x += dx
            y += dy
            
            if self.check_collision(x, y, radius=1):
                return step / RAY_LENGTH
        
        return 1.0
    
    def get_observation(self):
        """Get raycasting observation vector"""
        rays = []
        angle_range = math.pi / 2  # 90-degree forward arc
        start_angle = self.agent_angle - angle_range / 2
        
        for i in range(NUM_RAYS):
            angle = start_angle + (i / (NUM_RAYS - 1)) * angle_range
            distance = self.cast_ray(angle)
            rays.append(distance)
        
        return np.array(rays, dtype=np.float32)
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        self.steps += 1
        
        # Action space with velocity-based movements:
        # 0=forward, 1=backward, 2=slight left, 3=slight right, 4=hard left, 5=hard right
        collision = False
        
        if action == 0:  # Move forward
            new_x = self.agent_x + math.cos(self.agent_angle) * 15
            new_y = self.agent_y + math.sin(self.agent_angle) * 15
            
            if not self.check_collision(new_x, new_y):
                self.agent_x = new_x
                self.agent_y = new_y
            else:
                collision = True
        
        elif action == 1:  # Move backward
            new_x = self.agent_x - math.cos(self.agent_angle) * 10
            new_y = self.agent_y - math.sin(self.agent_angle) * 10
            
            if not self.check_collision(new_x, new_y):
                self.agent_x = new_x
                self.agent_y = new_y
            else:
                collision = True
        
        elif action == 2:  # Slight left (turn + move)
            self.agent_angle += 0.2
            new_x = self.agent_x + math.cos(self.agent_angle) * 12
            new_y = self.agent_y + math.sin(self.agent_angle) * 12
            
            if not self.check_collision(new_x, new_y):
                self.agent_x = new_x
                self.agent_y = new_y
            else:
                collision = True
        
        elif action == 3:  # Slight right (turn + move)
            self.agent_angle -= 0.2
            new_x = self.agent_x + math.cos(self.agent_angle) * 12
            new_y = self.agent_y + math.sin(self.agent_angle) * 12
            
            if not self.check_collision(new_x, new_y):
                self.agent_x = new_x
                self.agent_y = new_y
            else:
                collision = True
        
        elif action == 4:  # Hard left (sharp turn + move)
            self.agent_angle += 0.5
            new_x = self.agent_x + math.cos(self.agent_angle) * 8
            new_y = self.agent_y + math.sin(self.agent_angle) * 8
            
            if not self.check_collision(new_x, new_y):
                self.agent_x = new_x
                self.agent_y = new_y
            else:
                collision = True
        
        elif action == 5:  # Hard right (sharp turn + move)
            self.agent_angle -= 0.5
            new_x = self.agent_x + math.cos(self.agent_angle) * 8
            new_y = self.agent_y + math.sin(self.agent_angle) * 8
            
            if not self.check_collision(new_x, new_y):
                self.agent_x = new_x
                self.agent_y = new_y
            else:
                collision = True
        
        # Calculate reward
        current_distance = self.get_distance_to_goal()
        distance_reward = (self.prev_distance - current_distance) * 0.1
        self.prev_distance = current_distance
        
        reward = distance_reward
        reward -= 0.05  # Small step penalty to encourage efficiency
        
        # Collision penalty and episode termination
        done = False
        if collision:
            reward -= 50  # Large collision penalty
            done = True  # End episode on collision
        
        # Check if goal reached
        if current_distance < 30:
            reward += 100
            done = True
        
        # Timeout
        if self.steps >= MAX_STEPS:
            done = True
        
        obs = self.get_observation()
        
        return obs, reward, done
    
    def get_ray_endpoints(self):
        """Get ray endpoints for visualization"""
        endpoints = []
        angle_range = math.pi / 2
        start_angle = self.agent_angle - angle_range / 2
        
        for i in range(NUM_RAYS):
            angle = start_angle + (i / (NUM_RAYS - 1)) * angle_range
            distance = self.cast_ray(angle) * RAY_LENGTH
            
            end_x = self.agent_x + math.cos(angle) * distance
            end_y = self.agent_y + math.sin(angle) * distance
            endpoints.append((end_x, end_y))
        
        return endpoints


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.epsilon = EPSILON_START
        self.steps = 0
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < BATCH_SIZE:
            return 0
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy policy network weights to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


class Visualizer:
    def __init__(self, env):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("RL Navigation with Raycasting")
        self.clock = pygame.time.Clock()
        self.env = env
        self.font = pygame.font.Font(None, 24)
    
    def render(self, episode, reward, epsilon):
        """Render the environment"""
        self.screen.fill((255, 255, 255))
        
        # Draw obstacles
        for ox, oy, ow, oh in self.env.obstacles:
            rect = pygame.Rect(
                ox * self.env.cell_size,
                oy * self.env.cell_size,
                ow * self.env.cell_size,
                oh * self.env.cell_size
            )
            pygame.draw.rect(self.screen, self.env.wall_color, rect)
        
        # Draw rays
        ray_endpoints = self.env.get_ray_endpoints()
        for end_x, end_y in ray_endpoints:
            pygame.draw.line(
                self.screen,
                (200, 200, 255),
                (int(self.env.agent_x), int(self.env.agent_y)),
                (int(end_x), int(end_y)),
                1
            )
        
        # Draw goal
        pygame.draw.circle(
            self.screen,
            (0, 255, 0),
            (int(self.env.goal_x), int(self.env.goal_y)),
            20
        )
        
        # Draw agent
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            (int(self.env.agent_x), int(self.env.agent_y)),
            ROBOT_RADIUS
        )
        
        # Draw agent direction
        end_x = self.env.agent_x + math.cos(self.env.agent_angle) * 25
        end_y = self.env.agent_y + math.sin(self.env.agent_angle) * 25
        pygame.draw.line(
            self.screen,
            (255, 255, 0),
            (int(self.env.agent_x), int(self.env.agent_y)),
            (int(end_x), int(end_y)),
            3
        )
        
        # Draw info
        info_text = [
            f"Episode: {episode}",
            f"Reward: {reward:.2f}",
            f"Epsilon: {epsilon:.3f}",
            f"Steps: {self.env.steps}"
        ]
        
        for i, text in enumerate(info_text):
            surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(surface, (10, 10 + i * 25))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def check_quit(self):
        """Check if user wants to quit"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
    
    def close(self):
        pygame.quit()


def train():
    """Main training loop"""
    env = Environment(domain_randomization=True)
    agent = DQNAgent(state_size=NUM_RAYS, action_size=6)  # 6 actions now
    visualizer = None
    if VISUALIZE_TRAINING:
        visualizer = Visualizer(env)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    
    print("Starting training...")
    print(f"Visualization: {'ON' if VISUALIZE_TRAINING else 'OFF'}")
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        collision_episode = False
        
        for step in range(MAX_STEPS):
            # Check for quit only if visualizing
            if visualizer and visualizer.check_quit():
                print("Training interrupted by user")
                visualizer.close()
                return episode_rewards, episode_lengths
            
            # Select and execute action
            action = agent.select_action(state, training=True)
            next_state, reward, done = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            # Render if visualization is enabled and it's the right interval
            if visualizer and episode % VISUALIZE_INTERVAL == 0:
                visualizer.render(episode, episode_reward, agent.epsilon)
            
            if done:
                if env.get_distance_to_goal() < 30:
                    success_count += 1
                # Check if episode ended due to collision
                if reward <= -40:  # Collision causes large negative reward
                    collision_episode = True
                    collision_count += 1
                break
        
        # Update target network
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            success_rate = success_count / (episode + 1)
            collision_rate = collision_count / (episode + 1)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Success: {success_rate:.2%}, Collisions: {collision_rate:.2%}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    if visualizer:
        visualizer.close()
    return episode_rewards, episode_lengths


def plot_results(rewards, lengths):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    axes[0].plot(rewards, alpha=0.6, label='Episode Reward')
    
    # Moving average
    window = 20
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), moving_avg, 
                     'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Rewards Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    axes[1].plot(lengths, alpha=0.6, label='Episode Length')
    
    if len(lengths) >= window:
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(lengths)), moving_avg,
                     'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].set_title('Episode Length Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("Results saved to 'training_results.png'")
    plt.show()


def test_agent(agent, num_episodes=10, visualize=True):
    """Test the trained agent on new environments"""
    env = Environment(domain_randomization=True)
    visualizer = None
    if visualize:
        visualizer = Visualizer(env)
    
    test_rewards = []
    test_success = 0
    test_collisions = 0
    
    print(f"\nTesting agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            if visualizer and visualizer.check_quit():
                print("Testing interrupted by user")
                visualizer.close()
                return
            
            # Select action without exploration
            action = agent.select_action(state, training=False)
            next_state, reward, done = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if visualize:
                visualizer.render(episode, episode_reward, 0.0)
            
            if done:
                if env.get_distance_to_goal() < 30:
                    test_success += 1
                if reward <= -40:
                    test_collisions += 1
                break
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={step + 1}")
    
    if visualizer:
        visualizer.close()
    
    avg_reward = np.mean(test_rewards)
    success_rate = test_success / num_episodes
    collision_rate = test_collisions / num_episodes
    
    print(f"\nTest Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Collision Rate: {collision_rate:.2%}")
    
    return test_rewards


if __name__ == "__main__":
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available. Using CPU.")
    
    print()
    
    # Train the agent
    rewards, lengths = train()
    
    # Plot results
    plot_results(rewards, lengths)
    
    print("\nTraining complete!")
    
    # Ask user if they want to test
    print("\nStarting test phase with visualization...")
    
    # Create agent for testing (load the trained one)
    env = Environment(domain_randomization=True)
    agent = DQNAgent(state_size=NUM_RAYS, action_size=6)
    
    # In a real scenario, you'd load saved weights here
    # For now, we'll test with the trained agent from memory
    # agent.policy_net.load_state_dict(torch.load('model.pth'))
    
    test_rewards = test_agent(agent, num_episodes=5, visualize=True)
    
    print("\nAll done!")