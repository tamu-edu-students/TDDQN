import numpy as np
import random
from typing import Optional, Tuple, Union


class ReplayBuffer:

    def __init__(
        self,
        buffer_size: int,
        env_obs_length: Union[int, Tuple],
        obs_mask: int,
        max_episode_steps: int,
        context_len: Optional[int] = 1,
    ):
        self.max_size = buffer_size // max_episode_steps
        self.context_len = context_len
        self.env_obs_length = env_obs_length
        self.max_episode_steps = max_episode_steps
        self.obs_mask = obs_mask
        self.pos = [0, 0]

        self.obss = np.full(
            [
                self.max_size,
                max_episode_steps + 1,  # Keeps first and last obs together for +1
                env_obs_length,
            ],
            obs_mask,
            dtype=np.float32,
        )

        # Need the +1 so we have space to roll for the first observation
        self.actions = np.zeros(
            [self.max_size, max_episode_steps + 1, 1],
            dtype=np.uint8,
        )
        self.rewards = np.zeros(
            [self.max_size, max_episode_steps, 1],
            dtype=np.float32,
        )
        self.dones = np.ones(
            [self.max_size, max_episode_steps, 1],
            dtype=np.bool_,
        )
        self.episode_lengths = np.zeros([self.max_size], dtype=np.uint8)

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        episode_length: Optional[int] = 0,
    ) -> None:
        episode_idx = self.pos[0] % self.max_size
        obs_idx = self.pos[1]
        self.obss[episode_idx, obs_idx + 1] = obs
        self.actions[episode_idx, obs_idx] = action
        self.rewards[episode_idx, obs_idx] = reward
        self.dones[episode_idx, obs_idx] = done
        self.episode_lengths[episode_idx] = episode_length
        self.pos = [self.pos[0], self.pos[1] + 1]

    def store_obs(self, obs: np.ndarray) -> None:
        """Use this at the beginning of the episode to store the first obs"""
        episode_idx = self.pos[0] % self.max_size
        self.cleanse_episode(episode_idx)
        self.obss[episode_idx, 0] = obs

    def can_sample(self, batch_size: int) -> bool:
        return batch_size < self.pos[0]

    def flush(self):
        self.pos = [self.pos[0] + 1, 0]

    def cleanse_episode(self, episode_idx: int) -> None:
        self.obss[episode_idx] = np.full(
            [
                self.max_episode_steps
                + 1,  # Keeps first and last obs together for +1
                self.env_obs_length,
            ],
            self.obs_mask,
            dtype=np.float32,
        )
        self.actions[episode_idx] = np.zeros(
            [self.max_episode_steps + 1, 1],
            dtype=np.uint8,
        )
        self.rewards[episode_idx] = np.zeros(
            [self.max_episode_steps, 1],
            dtype=np.float32,
        )
        self.dones[episode_idx] = np.ones(
            [self.max_episode_steps, 1],
            dtype=np.bool_,
        )
        self.episode_lengths[episode_idx] = 0

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Exclude the current episode we're in
        valid_episodes = [
            i
            for i in range(min(self.pos[0], self.max_size))
            if i != self.pos[0] % self.max_size
        ]
        episode_idxes = np.array(
            [[random.choice(valid_episodes)] for _ in range(batch_size)]
        )
        transition_starts = np.array(
            [
                random.randint(
                    0, max(0, self.episode_lengths[idx[0]] - self.context_len)
                )
                for idx in episode_idxes
            ]
        )
        transitions = np.array(
            [range(start, start + self.context_len) for start in transition_starts]
        )
        return (
            self.obss[episode_idxes, transitions],
            self.actions[episode_idxes, transitions],
            self.rewards[episode_idxes, transitions],
            self.obss[episode_idxes, 1 + transitions],
            self.actions[episode_idxes, 1 + transitions],
            self.dones[episode_idxes, transitions],
            np.clip(self.episode_lengths[episode_idxes], 0, self.context_len),
        )
