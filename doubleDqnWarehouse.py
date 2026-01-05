import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# Grid setup (same as original)
num_rows, num_columns = 11, 11
actions = ['up', 'right', 'down', 'left']
penalties = np.full((num_rows, num_columns), -100.0)
penalties[0, 5] = 100.0  # Goal

open_areas = {
    1: list(range(1, 10)), 2: [1, 7, 9], 3: list(range(1, 8)) + [9],
    4: [3, 7], 5: list(range(num_columns)), 6: [5],
    7: list(range(1, 10)), 8: [3, 7], 9: list(range(num_columns))
}
for row in range(1, 10):
    for col in open_areas[row]:
        penalties[row, col] = -1.0


def check_terminal_state(row, col):
    return penalties[row, col] in [100.0, -100.0]


valid_positions = [
    (r, c) for r in range(num_rows)
    for c in range(num_columns)
    if not check_terminal_state(r, c)
]


def random_start():
    return valid_positions[np.random.randint(len(valid_positions))]


def calculate_distance_to_goal(row, col, goal_row=0, goal_col=5):
    return abs(row - goal_row) + abs(col - goal_col)


def farthest_start(goal_row=0, goal_col=5):
    distances = [
        (calculate_distance_to_goal(r, c, goal_row, goal_col), (r, c))
        for r, c in valid_positions
    ]
    distances.sort(reverse=True)
    return distances[0][1], distances[1][1], distances[2][1]


def calculate_next_position(row, col, action):
    if actions[action] == 'up' and row > 0:
        row -= 1
    elif actions[action] == 'right' and col < num_columns - 1:
        col += 1
    elif actions[action] == 'down' and row < num_rows - 1:
        row += 1
    elif actions[action] == 'left' and col > 0:
        col -= 1
    return row, col


# Neural Network for Q-function approximation
class DQN(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, output_size=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# Convert grid position to normalized state vector
def state_to_tensor(row, col):
    # Normalize to [0, 1] range
    return [row / num_rows, col / num_columns]


def select_action(state, epsilon, model, device):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = model(state_tensor)
        return q_values.argmax().item()


def train_double_dqn(episodes, epsilon=1.0, gamma=0.95, lr=0.001, 
                     batch_size=64, target_update=10, replay_size=10000,
                     min_replay_size=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize networks for robot
    robot_main = DQN().to(device)
    robot_target = DQN().to(device)
    robot_target.load_state_dict(robot_main.state_dict())
    robot_optimizer = optim.Adam(robot_main.parameters(), lr=lr)
    robot_replay = ReplayBuffer(replay_size)
    
    # Initialize networks for human
    human_main = DQN().to(device)
    human_target = DQN().to(device)
    human_target.load_state_dict(human_main.state_dict())
    human_optimizer = optim.Adam(human_main.parameters(), lr=lr)
    human_replay = ReplayBuffer(replay_size)
    
    total_rewards = []
    success_count = 0
    
    for ep in range(episodes):
        ep_reward = 0
        reached_goal = False
        
        # Robot training episode
        row, col = random_start()
        state = state_to_tensor(row, col)
        
        for step in range(1000):
            if check_terminal_state(row, col):
                break
            
            action = select_action(state, epsilon, robot_main, device)
            new_row, new_col = calculate_next_position(row, col, action)
            reward = penalties[new_row, new_col]
            next_state = state_to_tensor(new_row, new_col)
            done = check_terminal_state(new_row, new_col) or (new_row, new_col) == (0, 5)
            
            robot_replay.push(state, action, reward, next_state, done)
            ep_reward += reward
            
            row, col = new_row, new_col
            state = next_state
            
            if (row, col) == (0, 5):
                reached_goal = True
                break
        
        # Human training episode
        row, col = random_start()
        state = state_to_tensor(row, col)
        
        for step in range(1000):
            if check_terminal_state(row, col):
                break
            
            action = select_action(state, epsilon, human_main, device)
            new_row, new_col = calculate_next_position(row, col, action)
            reward = penalties[new_row, new_col]
            next_state = state_to_tensor(new_row, new_col)
            done = check_terminal_state(new_row, new_col) or (new_row, new_col) == (0, 5)
            
            human_replay.push(state, action, reward, next_state, done)
            ep_reward += reward
            
            row, col = new_row, new_col
            state = next_state
            
            if (row, col) == (0, 5):
                reached_goal = True
                break
        
        # Train robot network
        if len(robot_replay) >= min_replay_size:
            states, actions, rewards, next_states, dones = robot_replay.sample(batch_size)
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            
            # Double DQN: use main network for action selection, target for Q-value
            with torch.no_grad():
                next_actions = robot_main(next_states).argmax(1)
                next_q_values = robot_target(next_states).gather(1, next_actions.unsqueeze(1))
                target_q = rewards + (1 - dones) * gamma * next_q_values.squeeze()
            
            current_q = robot_main(states).gather(1, actions.unsqueeze(1)).squeeze()
            loss = F.mse_loss(current_q, target_q)
            
            robot_optimizer.zero_grad()
            loss.backward()
            robot_optimizer.step()
        
        # Train human network
        if len(human_replay) >= min_replay_size:
            states, actions, rewards, next_states, dones = human_replay.sample(batch_size)
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            
            # Double DQN: use main network for action selection, target for Q-value
            with torch.no_grad():
                next_actions = human_main(next_states).argmax(1)
                next_q_values = human_target(next_states).gather(1, next_actions.unsqueeze(1))
                target_q = rewards + (1 - dones) * gamma * next_q_values.squeeze()
            
            current_q = human_main(states).gather(1, actions.unsqueeze(1)).squeeze()
            loss = F.mse_loss(current_q, target_q)
            
            human_optimizer.zero_grad()
            loss.backward()
            human_optimizer.step()
        
        # Update target networks periodically
        if ep % target_update == 0:
            robot_target.load_state_dict(robot_main.state_dict())
            human_target.load_state_dict(human_main.state_dict())
        
        total_rewards.append(ep_reward)
        if reached_goal:
            success_count += 1
        
        epsilon = max(0.01, epsilon * 0.995)
        
        if ep % 100 == 0:
            print(f"Episode {ep}/{episodes}")
    
    print(f"Avg Reward: {np.mean(total_rewards):.2f}")
    print(f"Success Rate: {100 * success_count / episodes:.2f}%")
    
    return robot_main, human_main, total_rewards


def find_shortest_path(start_row, start_col, model, max_steps=200, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    path = [[start_row, start_col]]
    visited = set()
    row, col = start_row, start_col
    
    for _ in range(max_steps):
        if check_terminal_state(row, col) or (row, col) in visited:
            break
        visited.add((row, col))
        
        state = state_to_tensor(row, col)
        action = select_action(state, epsilon=0, model=model, device=device)
        row, col = calculate_next_position(row, col, action)
        path.append([row, col])
    
    return path, len(path)


def simulate_paths(robot_starts, human_starts, robot_model, human_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot_paths = [find_shortest_path(r[0], r[1], robot_model, device=device) 
                   for r in robot_starts]
    human_paths = [find_shortest_path(h[0], h[1], human_model, device=device) 
                   for h in human_starts]
    
    for i, path in enumerate(robot_paths):
        print(f"Robot {i+1}: {path[0]} | Steps: {path[1]}")
    for i, path in enumerate(human_paths):
        print(f"Human {i+1}: {path[0]} | Steps: {path[1]}")
    
    return robot_paths, human_paths


def create_simulation_gif(robot_paths, human_paths, output_file="warehouse_simulation.gif"):
    fig, ax = plt.subplots(figsize=(8, 8))
    grid_display = penalties.copy()
    offsets = [(-0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)]
    labels_added = False

    def plot_grid():
        ax.matshow(grid_display, cmap="coolwarm", vmin=-100, vmax=100)
        ax.set_xticks(np.arange(num_columns))
        ax.set_yticks(np.arange(num_rows))
        ax.grid(True)
        ax.set_title("Warehouse Simulation")

    def update(frame):
        nonlocal labels_added
        ax.clear()
        plot_grid()

        for i, path in enumerate(robot_paths):
            r, c = path[0][frame] if frame < len(path[0]) else path[0][-1]
            offset = offsets[i % len(offsets)]
            ax.plot(c + offset[1], r + offset[0], 'o', markersize=12,
                    label=f"Robot {i+1}" if frame == 0 else "")

        for i, path in enumerate(human_paths):
            r, c = path[0][frame] if frame < len(path[0]) else path[0][-1]
            offset = offsets[i % len(offsets)]
            ax.plot(c + offset[1], r + offset[0], 'x', markersize=12,
                    label=f"Human {i+1}" if frame == 0 else "")

        if not labels_added:
            ax.legend(loc="upper right")
            labels_added = True

    total_frames = max(len(p[0]) for p in robot_paths + human_paths)
    ani = FuncAnimation(fig, update, frames=total_frames, interval=500)
    ani.save(output_file, writer=PillowWriter(fps=5))
    plt.close()
    print(f"Simulation saved as {output_file}")


if __name__ == "__main__":
    robot_model, human_model, _ = train_double_dqn(
        episodes=5000, epsilon=1.0, gamma=0.95, lr=0.001,
        batch_size=64, target_update=10
    )

    robot_starts = [(9, 7), (7, 4), (7, 3)]
    human_starts = list(farthest_start())

    robot_paths, human_paths = simulate_paths(
        robot_starts, human_starts, robot_model, human_model
    )

    create_simulation_gif(robot_paths, human_paths)

