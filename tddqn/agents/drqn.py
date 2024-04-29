from typing import Callable, Union

import torch
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
from tddqn.agents.dqn import DqnAgent
from context.context import Context

class DrqnAgent(DqnAgent):
    # noinspection PyTypeChecker
    def __init__(
        self,
        network_factory: Callable[[], Module],
        buffer_size: int,
        device: torch.device,
        env_obs_length: int,
        max_env_steps: int,
        obs_mask: Union[int, float],
        num_actions: int,
        learning_rate: float = 0.0003,
        batch_size: int = 32,
        context_len: int = 50,
        gamma: float = 0.99,
        grad_norm_clip: float = 1.0,
        target_update_frequency: int = 10_000,
        embed_size: int = 64,
        history: int = 50,
        **kwargs,
    ):
        super().__init__(
            network_factory,
            buffer_size,
            device,
            env_obs_length,
            max_env_steps,
            obs_mask,
            num_actions,
            learning_rate,
            batch_size,
            context_len,
            gamma,
            grad_norm_clip,
            target_update_frequency,
            **kwargs,
        )
        self.history = history
        self.zeros_hidden = torch.zeros(
            1,
            1,
            embed_size,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        hidden_states = (self.zeros_hidden, self.zeros_hidden)

        self.train_context = Context(
            context_len,
            obs_mask,
            num_actions,
            env_obs_length,
            init_hidden=hidden_states,
        )
        self.eval_context = Context(
            context_len,
            obs_mask,
            num_actions,
            env_obs_length,
            init_hidden=hidden_states,
        )

        self.training = True

    def observe(self, obs, action, reward, done) -> None:
        self.context.add_transition(obs, action, reward, done)
        if self.training:
            self.replay_buffer.store(obs, action, reward, done, self.context.timestep)

    def get_action(self, epsilon: float = 0.0) -> int:
        with torch.no_grad():
            observation_tensor = (
                torch.as_tensor(
                    self.context.obs[min(self.context.timestep, self.context_len - 1)],
                    dtype=self.obs_tensor_type,
                    device=self.device,).unsqueeze(0).unsqueeze(0))
            action_tensor = (
                torch.as_tensor(
                self.context.action[min(self.context.timestep, self.context_len - 1)],
                dtype=torch.long,
                device=self.device,).unsqueeze(0).unsqueeze(0))
            q_values, self.context.hidden = self.policy_network(observation_tensor, action_tensor, hidden_states = self.context.hidden)
            rng = np.random.default_rng()
            if rng.random() < epsilon:
                return rng.integers(self.num_actions)
            else:
                return torch.argmax(q_values).item()

    def train(self) -> None:
        if not self.replay_buffer.can_sample(self.batch_size):
            return

        self.eval_off()
        (
            obss,
            actions,
            rewards,
            next_obss,
            next_actions,
            dones,
            episode_lengths,
        ) = self.replay_buffer.sample(self.batch_size)

        obss = torch.as_tensor(obss, dtype=self.obs_tensor_type, device=self.device)
        next_obss = torch.as_tensor(next_obss, dtype=self.obs_tensor_type, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        next_actions = torch.as_tensor(next_actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.long, device=self.device)
        episode_lengths = torch.as_tensor(episode_lengths, dtype=torch.long, device=torch.device("cpu")).squeeze()

        q_values, _ = self.policy_network(obss, actions, episode_lengths = episode_lengths)
        q_values = q_values.gather(2, actions).squeeze()

        with torch.no_grad():
            argmax = torch.argmax(self.policy_network(next_obss)[0],dim=2,).unsqueeze(-1)
            next_obs_q_values = (self.target_network(next_obss)[0].gather(2, argmax).squeeze())
            targets = rewards.squeeze() + (1 - dones.squeeze()) * (next_obs_q_values * self.gamma)

        q_values = q_values[:, -self.history :]
        targets = targets[:, -self.history :]
        loss = F.mse_loss(q_values, targets)
        self.td_errors.add(loss.item())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.num_train_steps += 1

        if self.num_train_steps % self.target_update_frequency == 0:
            self.target_update()
