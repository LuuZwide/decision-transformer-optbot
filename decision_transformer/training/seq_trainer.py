import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        loss_dict = {}

        #action loss
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        #state loss
        state_dim = state_preds.shape[2]
        state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        state_target = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]

        #reward loss
        reward_preds = reward_preds.reshape(-1)[attention_mask.reshape(-1) > 0]
        reward_target = rewards.reshape(-1)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target, action_target, reward_target,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            self.diagnostics['training/state_error'] = torch.mean((state_preds-state_target)**2).detach().cpu().item()
            self.diagnostics['training/reward_error'] = torch.mean((reward_preds-reward_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
