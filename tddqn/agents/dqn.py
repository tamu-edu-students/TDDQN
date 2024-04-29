from typing import Callable, Union
import torch
from torch.nn import Module
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from tddqn.buffers.replay_buffer import ReplayBuffer
from context.context import Context

class DqnAgent:
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
        context_len: int = 1,
        gamma: float = 0.99,
        grad_norm_clip: float = 1.0,
        target_update_frequency: int = 10_000,
        **kwargs,
    ):
        self.context_len = context_len
        self.env_obs_length = env_obs_length
        self.policy_network = network_factory()
        self.target_network = network_factory()
        self.target_update()
        self.target_network.eval()
        self.obs_context_type = np.int_
        self.obs_tensor_type = torch.long
        self.device = device
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(
            buffer_size,
            env_obs_length=env_obs_length,
            obs_mask=obs_mask,
            max_episode_steps=max_env_steps,
            context_len=context_len,
        )

        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_norm_clip = grad_norm_clip
        self.target_update_frequency = target_update_frequency
        self.num_train_steps = 0
        self.num_actions = num_actions
        self.training = True
        self.obs_mask = obs_mask

        self.train_context = Context(context_len, obs_mask, self.num_actions, env_obs_length)
        self.eval_context = Context(context_len, obs_mask, self.num_actions, env_obs_length)
        self.context = self.train_context

    def eval_on(self) -> None:
        self.training = False
        self.context = self.eval_context
        self.policy_network.eval()

    def eval_off(self) -> None:
        self.training = True
        self.context = self.train_context
        self.policy_network.train()

    def get_action(self, epsilon=0.0) -> int:
        with torch.no_grad():
            if np.random.default_rng().random() < epsilon:
                return np.random.default_rng().integers(self.num_actions)
            q_values = self.policy_network(
                torch.as_tensor(
                    self.context.obs[min(self.context.timestep, self.context_len - 1)],
                    dtype=self.obs_tensor_type,
                    device=self.device,
                ).unsqueeze(0).unsqueeze(0))
            return torch.argmax(q_values).item()

    def observe(self, obs, action, reward, done) -> None:
        if self.training:
            self.replay_buffer.store(obs, action, reward, done)

    def context_reset(self, obs: np.ndarray) -> None:
        self.context.reset(obs)
        if self.training:
            self.replay_buffer.store_obs(obs)

    def train(self) -> None:
        if not self.replay_buffer.can_sample(self.batch_size):
            return

        self.eval_off()
        obss, actions, rewards, next_obss, _, dones, _ = self.replay_buffer.sample(self.batch_size)
        obss = torch.as_tensor(obss, dtype=self.obs_tensor_type, device=self.device)
        next_obss = torch.as_tensor(next_obss, dtype=self.obs_tensor_type, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).squeeze()
        dones = torch.as_tensor(dones, dtype=torch.long, device=self.device).squeeze()

        q_values = self.policy_network(obss)
        q_values = q_values.gather(2, actions).squeeze()

        with torch.no_grad():
            next_obs_qs = self.policy_network(next_obss)
            argmax = torch.argmax(next_obs_qs, axis=-1).unsqueeze(-1)
            next_obs_q_values = (self.target_network(next_obss).gather(2, argmax).squeeze())
            targets = rewards + (1 - dones) * (next_obs_q_values * self.gamma)

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.num_train_steps += 1

        if self.num_train_steps % self.target_update_frequency == 0:
            self.target_update()

    def target_update(self) -> None:
        self.target_network.load_state_dict(self.policy_network.state_dict())
