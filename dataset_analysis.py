import pickle
import numpy as np

dataset_path = f'data/chart.pkl'

scale = 1000

with open(dataset_path, 'rb') as f:
    trajectories = pickle.load(f)

states, traj_lens, returns = [], [], []
for path in trajectories:
    states.append(path['observations'])
    traj_lens.append(len(path['observations']))
    returns.append(path['rewards'].sum() * scale)

traj_lens, returns = np.array(traj_lens), np.array(returns)  

print(f'Number of trajectories: {len(traj_lens)}')
print(f'Average return: {np.mean(returns)}, std: {np.std(returns)}, max: {np.max(returns)}, min: {np.min(returns)}')
print(f'Average trajectory length: {np.mean(traj_lens)}, std: {np.std(traj_lens)}, max: {np.max(traj_lens)}, min: {np.min(traj_lens)}')


