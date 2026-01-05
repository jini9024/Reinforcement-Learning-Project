from doubleDqnWarehouse import train_double_dqn, simulate_paths, farthest_start
import numpy as np
import time

print("="*70)
print("Testing Updated Double DQN Performance")
print("Changes: Training every 4 steps, min_replay_size=500")
print("="*70)

# Run training
start_time = time.time()
robot_model, human_model, rewards = train_double_dqn(
    episodes=3000, 
    epsilon=1.0, 
    gamma=0.95, 
    lr=0.001, 
    batch_size=64, 
    target_update=10
)
elapsed_time = time.time() - start_time

print("\n" + "="*70)
print("PERFORMANCE ANALYSIS")
print("="*70)

# Overall statistics
print(f"\nOverall Statistics (3000 episodes):")
print(f"  Average Reward: {np.mean(rewards):.2f}")
print(f"  Success Rate: {sum(1 for r in rewards if r > 50) / len(rewards) * 100:.2f}%")
print(f"  Training Time: {elapsed_time:.2f} seconds")

# Learning progression
first_500 = rewards[:500]
last_500 = rewards[-500:]
middle_500 = rewards[1000:1500]

print(f"\nLearning Progression:")
print(f"  First 500 episodes - Avg Reward: {np.mean(first_500):.2f}")
print(f"  Middle 500 episodes (1000-1500) - Avg Reward: {np.mean(middle_500):.2f}")
print(f"  Last 500 episodes - Avg Reward: {np.mean(last_500):.2f}")
print(f"  Improvement: {np.mean(last_500) - np.mean(first_500):.2f}")

# Success rate progression
first_success = sum(1 for r in first_500 if r > 50) / len(first_500) * 100
last_success = sum(1 for r in last_500 if r > 50) / len(last_500) * 100
print(f"\nSuccess Rate Progression:")
print(f"  First 500 episodes: {first_success:.2f}%")
print(f"  Last 500 episodes: {last_success:.2f}%")
print(f"  Improvement: {last_success - first_success:.2f}%")

# Test pathfinding
print(f"\n" + "="*70)
print("PATHFINDING TEST")
print("="*70)

robot_starts = [(9, 7), (7, 4), (7, 3)]
human_starts = list(farthest_start())

robot_paths, human_paths = simulate_paths(
    robot_starts, human_starts, robot_model, human_model
)

# Analyze path quality
print(f"\nPath Quality Analysis:")
robot_success = sum(1 for p in robot_paths if p[0][-1] == [0, 5])
human_success = sum(1 for p in human_paths if p[0][-1] == [0, 5])
print(f"  Robots reaching goal: {robot_success}/3")
print(f"  Humans reaching goal: {human_success}/3")
print(f"  Total success: {robot_success + human_success}/6")

avg_robot_steps = np.mean([p[1] for p in robot_paths])
avg_human_steps = np.mean([p[1] for p in human_paths])
print(f"  Average robot path length: {avg_robot_steps:.1f} steps")
print(f"  Average human path length: {avg_human_steps:.1f} steps")

print("\n" + "="*70)
print("TEST COMPLETED")
print("="*70)

