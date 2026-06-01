import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision_transformer.models.decision_transformer import DecisionTransformer


class TestDecisionTransformer(unittest.TestCase):
    def test_forward_returns_reward_head(self):
        model = DecisionTransformer(
            state_dim=27,
            act_dim=5,
            hidden_size=32,
            max_length=4,
            max_ep_len=1440,
            n_layer=2,
            n_head=2,
            n_inner=128,
            activation_function='relu',
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )

        states = torch.zeros((2, 4, 27), dtype=torch.float32)
        actions = torch.zeros((2, 4, 5), dtype=torch.float32)
        rewards = torch.zeros((2, 4, 1), dtype=torch.float32)
        rtg = torch.zeros((2, 5, 1), dtype=torch.float32)
        timesteps = torch.zeros((2, 4), dtype=torch.long)
        attention_mask = torch.ones((2, 4), dtype=torch.long)

        state_preds, action_preds, reward_preds = model(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        self.assertTrue(hasattr(model, 'predict_reward'))
        self.assertEqual(reward_preds.shape, rewards.shape)
        self.assertEqual(state_preds.shape, states.shape)
        self.assertEqual(action_preds.shape, actions.shape)


if __name__ == '__main__':
    unittest.main()
