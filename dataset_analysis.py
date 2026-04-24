import pickle
import numpy as np

dataset_path = f'data/chart.pkl'

scale = 1

average_step_reward = []

with open(dataset_path, 'rb') as f:
    trajectories = pickle.load(f)

states, traj_lens, returns = [], [], []
for path in trajectories:

    #clip all rewards to be in [-1, 1]
    for i in range(len(path['rewards'])):
        if path['rewards'][i]*scale > 1.0:
            path['rewards'][i] = 1.0
        elif path['rewards'][i]*scale < -1.0:
            path['rewards'][i] = -1.0
    states.append(path['observations'])
    traj_lens.append(len(path['observations']))
    returns.append(path['rewards'].sum())
    average_step_reward.extend(path['rewards'])

traj_lens, returns = np.array(traj_lens), np.array(returns)  

print(f'Number of trajectories: {len(traj_lens)}')
print(f'Average return trajectory: {np.mean(returns)}, std: {np.std(returns)}, max: {np.max(returns)}, min: {np.min(returns)}')
print(f'Average trajectory length: {np.mean(traj_lens)}, std: {np.std(traj_lens)}, max: {np.max(traj_lens)}, min: {np.min(traj_lens)}')
print(f'Average step reward: {np.mean(average_step_reward)}, std: {np.std(average_step_reward)}, max: {np.max(average_step_reward)}, min: {np.min(average_step_reward)}')


#Number of trajectories where returns between -1 and 1, and between 1 and greater than 5
num_traj_between_neg1_and_1 = np.sum((returns >= -1) & (returns <= 1))
num_traj_between_1_and_5 = np.sum((returns > 1) & (returns <= 5))
num_traj_greater_than_5 = np.sum(returns > 5)

print(f'Number of trajectories with returns between -1 and 1: {num_traj_between_neg1_and_1}')
print(f'Number of trajectories with returns between 1 and 5: {num_traj_between_1_and_5}')
print(f'Number of trajectories with returns greater than 5: {num_traj_greater_than_5}')

target_return = 30.0

for coef in [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]:
    num_traj_above_target = np.sum(returns >= target_return * coef)
    print(f'Number of trajectories with returns above {target_return * coef}: {num_traj_above_target}')

#Print first observation, reward and action of the first trajectory
print(f'First observation of the first trajectory: {states[0][0]}')
print(f'First reward of the first trajectory: {returns[0]}')
print(f'First action of the first trajectory: {trajectories[879]["actions"][11]}')
