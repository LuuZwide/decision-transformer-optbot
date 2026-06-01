import torch
import numpy as np
import pickle
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.Colab import ChartEnv, build
import matplotlib.pyplot as plt
import seaborn as sns

def get_attention_weights(model, state, actions, rewards, target_return, timesteps):
    """
    Extracts attention weights from the model to see if it is 
    attending to the target_return token.
    """
    model.eval()
    
    # 1. Ensure the model is configured to return attention weights
    # Most implementations require setting output_attentions=True in the transformer config
    # If your specific model doesn't support this, you'll need to use a hook (see below).
    
    with torch.no_grad():
        # Ensure the inputs have batch and sequence dimensions
        state = state.reshape(1, -1, model.state_dim)
        actions = actions.reshape(1, -1, model.act_dim)
        rewards = rewards.reshape(1, -1)
        target_return = target_return.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # Forward pass: Assuming model.forward() returns (state_preds, action_preds, return_preds, attention_weights)
        _, _, _, attention_weights = model(
            states=state,
            actions=actions,
            rewards=rewards,
            returns_to_go=target_return,
            timesteps=timesteps,
            output_attentions=True # This flag is key
        )
    
    # 2. Analyze weights
    # attention_weights usually has shape (num_layers, batch_size, num_heads, sequence_length, sequence_length)
    # For a Decision Transformer, your sequence length is (K * 3) if you pack (R, S, A)
    
    # Check the last layer's attention (usually the most refined)
    last_layer_attn = attention_weights[-1] # [batch, heads, seq, seq]
    
    # Average over heads
    avg_attn = last_layer_attn.mean(dim=1).squeeze(0) # [seq, seq]
    
    # In a typical DT (R, S, A) sequence:
    # The 'return' tokens are usually at indices [0, 3, 6...] 
    # If the model is conditioned, these indices should show high attention 
    # towards the recent 'state' and 'action' tokens.
    
    return avg_attn

# --- Alternative: Using a Hook if model.forward() doesn't return weights ---
def get_attn_via_hook(model):
    attn_storage = {}
    
    def hook_fn(module, input, output):
        # Captures the attention output of the specific attention layer
        attn_storage['weights'] = output[1] 

    model.transformer.h[-1].attn.register_forward_hook(hook_fn)
    return attn_storage 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path ="/opt/decision-transformer-optbot/data/chart.pkl"
with open(dataset_path, 'rb') as f:
    trajectories = pickle.load(f)


states, traj_lens, returns = [], [], []
for traj in trajectories:
    states.append(traj['observations'])
    traj_lens.append(len(traj['observations']))
    returns.append(traj['rewards'].sum())

# Concatenate all observations into a single array for proper normalization
all_observations = np.concatenate(states, axis=0)
state_mean = torch.from_numpy(np.mean(all_observations, axis=0)).to(device=device, dtype=torch.float32)
state_std = torch.from_numpy(np.std(all_observations, axis=0) + 1e-6).to(device=device, dtype=torch.float32)
state_dim = 27
act_dim = 5
max_ep_len = 1440
scale = 1

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    max_length=20,
    max_ep_len=1440,
    hidden_size=128,
    n_layer=3,
    n_head=4,
    activation_function='relu',
    n_positions=1024,
    n_inner=4*128,
    resid_pdrop=0.1,
    attn_pdrop=0.1
).to(device=device)

max_length = 20

model.eval()
model.load_state_dict(torch.load('/opt/decision-transformer-optbot/saved_models/DT_chart-ASR/DT', map_location=device, weights_only=True))
model.to(device=device)

initial_target_return = 25.0
env_charts, env_close_prices, env_test_charts, env_close_test_prices = build.build_charts()

curr_value_sum = 0
port_value_sum = 0
# run for 10 episodes
action_array = []
avg_returns = []
for episode in range(20):
    env = ChartEnv.ChartEnv(chart = env_test_charts, close_prices= env_close_test_prices , symbols = ['EURUSD', 'GBPUSD','USDJPY','USDCHF','AUDUSD'],timesteps = 1, episode_length = 1440, recurrent= False, random_start=True)

    state = env.reset()
    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    ep_return = initial_target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    sim_states = []
    episode_return, episode_length, predicted_return = 0, 0, 0
    for t in range(10):
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        

        # Trim to max_length for model input
        states_input = states[-max_length:] if len(states) > max_length else states
        actions_input = actions[-max_length:] if len(actions) > max_length else actions
        rewards_input = rewards[-max_length:] if len(rewards) > max_length else rewards
        target_return_input = target_return[:, -max_length:] if target_return.shape[1] > max_length else target_return
        timesteps_input = timesteps[:, -max_length:] if timesteps.shape[1] > max_length else timesteps

        action, returns_predictions = model.get_action(
            (states_input.to(dtype=torch.float32) - state_mean) / state_std,
            actions_input.to(dtype=torch.float32),
            rewards_input.to(dtype=torch.float32),
            target_return_input.to(dtype=torch.float32),
            timesteps_input.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()
        action_array.append(action)

        state, reward, done, info = env.step(action)
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
    
        pred_return = target_return[0,-1] - (reward/scale)
        #print("Predicted Return: ",returns_predictions.detach().cpu().numpy()[0], "Reward: ", reward, "Updated Predicted Return: ", pred_return.detach().cpu().numpy()[0])
        predicted_return += returns_predictions.detach().cpu().numpy()[0]

        # if t < 500 :
        #    avg_attn = get_attention_weights(model, states_input, actions_input, rewards_input, target_return_input, timesteps_input)
        # else:
        #    done = True
        if episode_return < -5:
            done = True

        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)
        episode_return += reward
        #print("Predicted Return: ",returns_to_go.detach().cpu().numpy()[0], "Episode Return: ", target_return[0,-1].detach().cpu().numpy())
        episode_length += 1
        if done:
            break   
    
    avg_returns.append(episode_return)
    curr_value_sum += (info['current_value'] -1 )
    port_value_sum += (info['port_value'] -1 )
    #average actions
    avg_actions, std_actions = np.round(np.mean(action_array, axis=0),2), np.round(np.std(action_array, axis=0),2)
    print(f"Episode {episode}: Step {t}, Final Episode Return: {returns_predictions.detach().cpu().numpy()[0]}, Predicted Return: {predicted_return}, current_value: {info['current_value']}, port_value: {info['port_value']}, avg_action: {avg_actions}, std_action: {std_actions}")

print(f"Average Return over 10 episodes: {np.mean(avg_returns)}")
print("Current Value sum : ", curr_value_sum)
print("Port Value sum : ", port_value_sum)
#plt.figure(figsize=(10, 8))
#sns.heatmap(avg_attn.cpu().numpy(), cmap='viridis')
#plt.title("Attention Heatmap")
#plt.show()