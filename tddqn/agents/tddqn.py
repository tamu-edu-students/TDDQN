from typing import Callable, Union
import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F
from tddqn.agents.drqn import DqnAgent

class TddqnAgent(DqnAgent):
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
            target_update_frequency
        )
        self.history = history

    def get_action(self, epsilon: float = 0.0) -> int:
        with torch.no_grad():
            if np.random.default_rng().random() < epsilon:
                return np.random.default_rng().integers(self.num_actions)
            context_obs_tensor = torch.as_tensor(
                self.context.obs[: min(self.context.max_length, self.context.timestep + 1)],
                dtype=self.obs_tensor_type,
                device=self.device).unsqueeze(0)
            
            q_values = self.policy_network(context_obs_tensor)
            return torch.argmax(q_values[:, -1, :]).item()

    def context_reset(self, obs: np.ndarray) -> None:
        self.context.reset(obs)
        if self.training:
            self.replay_buffer.store_obs(obs)

    def observe(self, obs: np.ndarray, action: int, reward: float, done: bool) -> None:
        self.context.add_transition(obs, action, reward, done)
        if self.training:
            self.replay_buffer.store(obs, action, reward, done, self.context.timestep)

    def train(self) -> None:
        if not self.replay_buffer.can_sample(self.batch_size):
            return
        self.eval_off()
        
        (obss, actions, rewards, next_obss, next_actions, dones, episode_lengths) = self.replay_buffer.sample(self.batch_size)

        obss = torch.as_tensor(obss, dtype=self.obs_tensor_type, device=self.device)
        next_obss = torch.as_tensor(next_obss, dtype=self.obs_tensor_type, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        next_actions = torch.as_tensor(next_actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.long, device=self.device)
        q_values = self.policy_network(obss)
        q_values = q_values.gather(2, actions).squeeze()

        with torch.no_grad():
            if self.history:
                argmax = torch.argmax(self.policy_network(next_obss), dim=2,).unsqueeze(-1)
                next_obs_q_values = self.target_network(next_obss)
                next_obs_q_values = next_obs_q_values.gather(2, argmax).squeeze()

            targets = rewards.squeeze() + (1 - dones.squeeze()) * (
                next_obs_q_values * self.gamma
            )

        q_values = q_values[:, -self.history :]
        targets = targets[:, -self.history :]
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.num_train_steps += 1

        if self.num_train_steps % self.target_update_frequency == 0:
            self.target_update()
