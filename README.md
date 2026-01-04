# Warehouse Robot Pathfinding with Reinforcement Learning

A reinforcement learning project that uses Q-Learning to train autonomous agents (robots and humans) to navigate through a warehouse environment with obstacles and find optimal paths to a goal location.

## Project Overview

This project implements a grid-based warehouse navigation system where agents learn to navigate from random starting positions to a goal location while avoiding obstacles. The implementation uses Q-Learning, a model-free reinforcement learning algorithm, to train two separate agents (robot and human) with independent Q-tables.

## Methods Used

### Tabular Q-Learning (Not DQN)

**Important Note**: This project uses **Tabular Q-Learning**, not Deep Q-Networks (DQN). 

- **Tabular Q-Learning**: Uses a Q-table (3D NumPy array) to store Q-values for each state-action pair directly. This is suitable for small, discrete state spaces (11×11 grid = 121 states).
- **DQN (Deep Q-Network)**: Would use a neural network to approximate Q-values, which is necessary for large or continuous state spaces but not needed here.

The implementation uses a Q-table of shape `(num_rows, num_columns, 4)` where each entry `Q[row, col, action]` stores the learned Q-value for that state-action pair.

### Q-Learning Algorithm

**Q-Learning** is a value-based, model-free reinforcement learning algorithm that learns the optimal action-selection policy by estimating the quality (Q-value) of state-action pairs. The algorithm uses the following update rule:

```
Q(s, a) ← Q(s, a) + α[r + γ * max(Q(s', a')) - Q(s, a)]
```

Where:
- **Q(s, a)**: Current Q-value for state `s` and action `a`
- **α (alpha)**: Learning rate (0.8-0.9) - controls how much new information overrides old information
- **r**: Immediate reward received
- **γ (gamma)**: Discount factor (0.95) - determines the importance of future rewards
- **s'**: Next state after taking action `a`

### Why Q-Learning?

1. **Model-Free**: Q-Learning doesn't require a model of the environment's dynamics, making it suitable for environments where transition probabilities are unknown.

2. **Off-Policy Learning**: The algorithm can learn the optimal policy while following an exploratory policy (ε-greedy), allowing for efficient exploration of the state space.

3. **Discrete State-Action Space**: Perfect for grid-based environments where states and actions are discrete and finite.

4. **Convergence Guarantees**: Under certain conditions, Q-Learning is guaranteed to converge to the optimal Q-function.

5. **Simplicity**: Easy to implement and understand, making it ideal for educational purposes and prototyping.

### Exploration Strategy: ε-Greedy

The implementation uses an **ε-greedy exploration strategy**:
- With probability **ε**: Select a random action (exploration)
- With probability **1-ε**: Select the action with the highest Q-value (exploitation)

The epsilon value decays over time (ε = max(0.01, ε × 0.995)), starting at 1.0 (pure exploration) and gradually decreasing to favor exploitation as the agent learns.

## Environment Setup

### Grid Configuration
- **Grid Size**: 11×11 cells
- **Goal Location**: (0, 5) with reward +100
- **Obstacles**: Cells with reward -100 (walls/barriers)
- **Open Areas**: Cells with reward -1 (navigable paths)

### Action Space
Four possible actions:
- `up`: Move one cell up
- `right`: Move one cell right
- `down`: Move one cell down
- `left`: Move one cell left

### Reward Structure

**Note**: The reward function is **not binary (0/1)**. It uses a three-tier reward system:

- **Goal State**: +100 (terminal state, episode ends successfully)
- **Obstacle/Wall**: -100 (terminal state, episode ends with failure)
- **Open Path**: -1 (small penalty per step to encourage finding shorter paths)

This reward structure provides:
- Strong positive signal when reaching the goal (+100)
- Strong negative signal when hitting obstacles (-100)
- Small step penalty (-1) to incentivize efficiency and shorter paths

## Current Results

### Training Performance
- **Training Episodes**: 5,000
- **Average Reward**: 135.60
- **Success Rate**: 93.28% (agents successfully reach the goal)

### Hyperparameters
- **Learning Rate (α)**: 0.8-0.9
- **Discount Factor (γ)**: 0.95
- **Initial Epsilon (ε)**: 1.0 (decays to 0.01)
- **Epsilon Decay**: 0.995 per episode

### Pathfinding Results

**Robot Agents** (3 different starting positions):
- Robot 1: 16 steps from (9, 7) to (0, 5)
- Robot 2: 13 steps from (7, 4) to (0, 5)
- Robot 3: 14 steps from (7, 3) to (0, 5)

**Human Agents** (3 farthest starting positions):
- Human 1: 19 steps from (9, 10) to (0, 5)
- Human 2: 19 steps from (9, 0) to (0, 5)
- Human 3: 18 steps from (9, 9) to (0, 5)

### Visualization
The project generates an animated GIF (`warehouse_simulation.gif`) showing the simultaneous movement of all agents through the warehouse environment.

## Project Structure

```
Reinforcement-Learning-Project/
├── WarehouseRobotPath.ipynb    # Main implementation notebook
└── README.md                    # This file
```

## Dependencies

- `numpy`: Numerical computations and array operations
- `matplotlib`: Plotting and visualization
- `IPython`: Display functionality for Jupyter notebooks

Install dependencies:
```bash
pip install numpy matplotlib ipython
```

## How to Run

1. Open `WarehouseRobotPath.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells to:
   - Train the Q-Learning agents (robot and human)
   - Generate optimal paths from various starting positions
   - Create an animated visualization of the paths
3. The simulation GIF will be saved as `warehouse_simulation.gif`

### Code Cells Overview

- **Cell 0**: Main implementation with training, pathfinding, and visualization
- **Cell 1**: Alternative implementation with different hyperparameters (ε=0.1, α=0.5) and reward plotting
- **Cell 2**: Another variant with α=0.65 and reward visualization

## Key Features

1. **Dual Agent System**: Separate Q-tables for robot and human agents, allowing independent learning
2. **Flexible Starting Positions**: Supports random starts and farthest-point selection for testing
3. **Path Visualization**: Animated GIF showing agent movements through the warehouse
4. **Performance Metrics**: Tracks success rate, average rewards, and path lengths

## Future Improvements

- Implement Deep Q-Networks (DQN) for larger state spaces
- Add multi-agent coordination and collision avoidance
- Experiment with different reward structures
- Implement policy gradient methods (e.g., REINFORCE, Actor-Critic)
- Add real-time visualization during training

## License

This project is open source and available for educational purposes.

