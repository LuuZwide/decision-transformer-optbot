import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import os
import optuna

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.training.Colab import ChartEnv, build, utils

os.environ["WANDB_MODE"] = "offline"
env_charts, env_close_prices, env_test_charts, env_close_test_prices = build.build_charts()

max_ep_len = 1000
scale = 1.0
env_targets = [30.0]

"""
    tags used to define and separate different experiments
    Affect Project_name and Experiment_name    
        - baseline
        - HPS
"""

global_step = 0

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def save_model(model, path):
    #if path does not exist create it
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")

def experiment(
        project_name,
        variant
    ):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    project_name = f'{project_name}-{variant["tag"]}'
    env_name = variant['env']
    model_type = variant['model_type']
    experiment_name = f'DT_{env_name}-{variant["loss_outputs"]}'

    state_dim = 32
    act_dim = 5

    # load dataset
    dataset_path = f'data/chart.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations']) #append all the observations
        traj_lens.append(len(path['observations']))# count of all the observations in the trajectory
        returns.append(path['rewards'].sum()) # Sum of all rewards in the trajectory
    traj_lens, returns = np.array(traj_lens), np.array(returns)


    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens) # Number of timesteps in the entire dataset

    print('=' * 50)
    print(f'Starting new experiment: {env_name}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1) # si - starting index

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim)) # max_len is context length K
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1) #Pad if traj less than K
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):

        return_dict = dict()
        def fn(model):
            returns, lengths, current_values = [], [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length, current_value = evaluate_episode_rtg(
                            ChartEnv.ChartEnv(chart = env_test_charts, close_prices= env_close_test_prices , symbols = ['EURUSD', 'GBPUSD','USDJPY','USDCHF','AUDUSD'],timesteps = 1, episode_length = 1440, recurrent= False, random_start=True) ,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            ChartEnv.ChartEnv(chart = env_test_charts, close_prices= env_close_test_prices , symbols = ['EURUSD', 'GBPUSD','USDJPY','USDCHF','AUDUSD'],timesteps = 1, episode_length = 1440, recurrent= False, random_start=True) ,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
                if model_type == 'dt':
                    current_values.append(current_value)
            
            return_dict['return_mean_gm'] = np.mean(returns)
            return_dict['return_std_gm'] = np.std(returns)
            return_dict['normalised_return'] = utils.normalize_score(np.mean(returns))
            return_dict['length_mean_gm'] = np.mean(lengths)
            return_dict['length_std_gm'] = np.std(lengths)
            return_dict['current_value_mean_gm'] = np.mean(current_values)

            #Evaluate Conditional performance
            rcsl_table = wandb.Table(columns=["Target Performance", "Actual Performance"], allow_mixed_types=True) #Normalised scores 
            rcsl_error_table = wandb.Table(columns=["Target Return", "MSE"], allow_mixed_types=True) #Mean Squared Error(MSE) L2
            rcsl_std_table = wandb.Table(columns=["Target Return", "STD"], allow_mixed_types=True) #Standard Deviation(STD)
            rcsl_norm_score = wandb.Table(columns=["Target Norm Return", "Actual Return"], allow_mixed_types=True)
            rcsl_mean_length = wandb.Table(columns=["Target Return", "Mean Length"], allow_mixed_types=True) #Mean Length of episodes
            rcsl_std_length = wandb.Table(columns=["Target Return", "STD Length"], allow_mixed_types=True) #STD of Length of episodes
            rcsl_current_value_table = wandb.Table(columns=["Target Return", "Mean Current Value"], allow_mixed_types=True) #Mean Current Value of episodes

            rc_loss = 0
            
            for eval_rtg_coef in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                eval_rtg = target_rew * eval_rtg_coef 
                scores = []
                current_values = []
                for i in range(num_eval_episodes):
                    ret, length, current_value = evaluate_episode_rtg(
                        ChartEnv.ChartEnv(chart = env_test_charts, close_prices= env_close_test_prices , symbols = ['EURUSD', 'GBPUSD','USDJPY','USDCHF','AUDUSD'],timesteps = 1, episode_length = 1440, recurrent= False, random_start=True),
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=eval_rtg/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        )
                    scores.append(ret)
                    lengths.append(length)
                    if model_type == 'dt':
                        current_values.append(current_value)
                
                mean_scores,std_scores = np.mean(scores),np.std(scores)
                mean_lengths, std_lengths = np.mean(lengths), np.std(lengths)
                mean_current_values, std_current_values = np.mean(current_values), np.std(current_values)
                rcsl_current_value_table.add_data(eval_rtg, mean_current_values)

                #RCSL Table
                rcsl_table.add_data(eval_rtg, mean_scores)
                norm_target = utils.normalize_score(eval_rtg)
                norm_actual = utils.normalize_score(mean_scores)


                rcsl_norm_score.add_data(norm_target,norm_actual)

                #RCSL Error Table
                rcsl_error = (eval_rtg - mean_scores)**2
                rcsl_error_table.add_data(eval_rtg, rcsl_error)

                #RCSL std Table 
                rcsl_std_table.add_data(eval_rtg, std_scores)

                #RCSL Length Tables
                rcsl_mean_length.add_data(eval_rtg, mean_lengths)
                rcsl_std_length.add_data(eval_rtg, std_lengths)

                rc_loss += rcsl_error

            return_dict[f'rcsl_evaluation/RCSL Table'] = rcsl_table # Target vs Mean achieved 
            return_dict[f'rcsl_evaluation/RCSL Error Table'] = rcsl_error_table # Target vs (MSE Error for that target return)
            return_dict[f'rcsl_evaluation/RCSL std Table'] = rcsl_std_table # Target vs (STD of achieved)
            return_dict["rcsl_evaluation/RCSL total loss"] = rc_loss # Total RCSL Loss
            return_dict["rcsl_evaluation/RCSL mean length"] = rcsl_mean_length # Target vs Mean Length
            return_dict["rcsl_evaluation/RCSL std length"] = rcsl_std_length
            return_dict["rcsl_evaluation/RCSL mean current value"] = mean_current_values
            return_dict["rcsl_evaluation/RCSL std current value"] = std_current_values
            return_dict["rcsl_evaluation/RCSL current value table"] = rcsl_current_value_table
            return_dict["rcsl_evaluation/RCSL norm score table"] = rcsl_norm_score
            return return_dict
        
        return fn

    
    def get_loss_fn(s_hat, a_hat, r_hat, s, a, r, loss_outputs):
        
        #Build loss function based on outputs and return built function

        if loss_outputs == 'A':
            return torch.mean((a_hat - a)**2)
        elif loss_outputs == 'AS':
            return torch.mean((a_hat - a)**2) + torch.mean((s_hat - s)**2)
        elif loss_outputs == 'AR': 
            return torch.mean((a_hat - a)**2) + torch.mean((r_hat - r)**2)
        else :
            return torch.mean((a_hat - a)**2) + torch.mean((s_hat - s)**2) + torch.mean((r_hat - r)**2)
        
    #Perform hyper parameter search ....
    def objective(trial):

        #Phase 1 of HPS - Stabilize training

        #Phase 2 Architecture related hyperparameters
            # n_layer 
            # embed_dim
            # n_head
        
        #Phase 3 Regularization related hyperparameters
            # dropout

        learning_rate = trial.suggest_loguniform("learning_rate",1e-5, 1e-2)
        #n_layer = trial.suggest_categorical("n_layer", [2, 4, 6])
        #n_head = trial.suggest_categorical("n_head", [2, 4, 8])
        #dropout = trial.suggest_uniform("dropout",0.1,0.3)
        embed_dim = trial.suggest_categorical("embed_dim",[128,256])
        context_k = trial.suggest_categorical("context_k",[10,20,50])
        batch_size = trial.suggest_categorical("batch_size", [32, 64])

        variant['learning_rate'] = learning_rate
        #variant['n_layer'] = n_layer
        #variant['n_head'] = n_head
        #variant['dropout'] = dropout
        variant['embed_dim'] = embed_dim
        variant['batch_size'] = batch_size
        
        dict_values = {}
        dict_values['learning_rate'] = learning_rate
        #dict_values['n_layer'] = n_layer    
        #dict_values['n_head'] = n_head
        #dict_values['dropout'] = dropout
        dict_values['embed_dim'] = embed_dim
        dict_values['batch_size'] = batch_size

        if model_type == 'dt':
            model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=context_k,
                max_ep_len=max_ep_len,
                hidden_size=variant['embed_dim'],
                n_layer=variant['n_layer'],
                n_head=variant['n_head'],
                n_inner=4*variant['embed_dim'],
                activation_function=variant['activation_function'],
                n_positions=1024,
                resid_pdrop=variant['dropout'],
                attn_pdrop=variant['dropout'],
            )
        else :
            model = MLPBCModel(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=context_k,
                hidden_size=variant['embed_dim'],
                n_layer=variant['n_layer'],
            )
        
        model = model.to(device=device)

        warmup_steps = variant['warmup_steps']
        optimizer = torch.optim.AdamW( # type: ignore
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
        )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
            )

        if model_type == 'dt':
            trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn= lambda s_hat, a_hat, r_hat, s, a, r: get_loss_fn(s_hat, a_hat, r_hat, s, a, r, variant['loss_outputs']),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
        else :
            trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn= lambda s_hat, a_hat, r_hat, s, a, r: get_loss_fn(s_hat, a_hat, r_hat, s, a, r, variant['loss_outputs']),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
        
        for iter in range(variant['max_hp_iters']):
            outputs, rcsl_outputs = trainer.train_iteration(num_steps=variant['num_hp_steps_per_iter'], iter_num=iter+1, print_logs=True)
            current_return = outputs["evaluation/return_mean_gm"]
            trial.report(current_return, step=iter)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return outputs["evaluation/return_mean_gm"]

    if variant['do_search'] != 0: 
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(5, variant['num_trials'] // 5),
            n_warmup_steps=max(1, variant['max_hp_iters'] // 3),
            interval_steps=1,
        )

        study = optuna.create_study(direction='maximize',
                                    storage="sqlite:///dt_" + variant['env'] + "-" + variant['loss_outputs'] + ".sqlite3",
                                    study_name=variant['env']+"-Hyperparam-search-"+variant['loss_outputs'],
                                    pruner=pruner)

        study.optimize(
            objective,
            n_trials=variant['num_trials']

        )
        #change experiment name ...-Best
        out = variant['loss_outputs']
        experiment_name = f'DT_{env_name}-{out}-Best'

        #Best params
        variant['learning_rate'] = study.best_params['learning_rate']
        #variant['n_layer'] = study.best_params['n_layer']
        #variant['n_head'] = study.best_params['n_head']
        #variant['dropout'] = study.best_params['dropout']   
        variant['embed_dim'] = study.best_params['embed_dim']
        variant['K'] = study.best_params['context_k']
        variant['batch_size'] = study.best_params['batch_size']
        print("HPS Complete - Best hyperparameters : ", study.best_params)


    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=variant['K'] ,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=variant['K'] ,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW( # type: ignore
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=variant['batch_size'],
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn= lambda s_hat, a_hat, r_hat, s, a, r: get_loss_fn(s_hat, a_hat, r_hat, s, a, r, variant['loss_outputs']), #log a and r loss
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=variant['batch_size'],
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn= lambda s_hat, a_hat, r_hat, s, a, r: get_loss_fn(s_hat, a_hat, r_hat, s, a, r, variant['loss_outputs']),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
        
    wandb.init(
            project= project_name,
            name= experiment_name,
            config=variant
        )

    for iter in range(variant['max_iters']):
        outputs, rcsl_outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)
            wandb.log(rcsl_outputs)
        save_model(trainer.model, f"saved_models/{experiment_name}/iter_{iter+1}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='chart')
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000) #30 minutes each 
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)

    #Hyperparameter search iters
    parser.add_argument('--do_search', type=int, default=0) #1 True / 0 False
    parser.add_argument('--num_trials', type=int, default=10) 
    parser.add_argument('--max_hp_iters', type=int, default=10) 
    parser.add_argument('--num_hp_steps_per_iter', type=int, default=10) #30 minutes each 
    parser.add_argument('--tag', type=str, default='baseline') #HPS / baseline

    #Outputs
    parser.add_argument('--loss_outputs', type=str, default='A') #Can be A, AS, or ASR
    
    args = parser.parse_args()
    
    project_name = 'decision-transformer-opt-experiments'
    
    experiment(project_name=project_name, variant=vars(args))

    if args.log_to_wandb:
        wandb.finish()
