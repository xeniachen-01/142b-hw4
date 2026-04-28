from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim
import time 

from .env import Connect4Config, Connect4Env
from .opponents import Agent
from .evaluate import evaluate_agent_pair
import matplotlib.pyplot as plt
from collections import deque
from IPython.display import clear_output
import gymnasium as gym

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")
print("Torch:", torch.__version__, "| Gymnasium:", gym.__version__)



@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 128
    replay_capacity: int = 50000 #M for replay buffer capacity
    min_replay_size: int = 1000 #minimum replay buffer size before starting optimization
    target_sync_interval: int = 250 #C for target network sync
    train_interval: int = 1     #N for number of environment steps between optimization updates for replay ratio
    hidden_dim: int = 128  #set based on google search
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000
    max_episodes: int = 500
    eval_interval: int = 50
    eval_games: int = 40
    gradient_clip_norm: float = 5 # reduce to 1 if unstable
    seed: int = 42


    # win rate 92.5%
    # gamma: float = 0.99
    # learning_rate: float = 3e-4
    # batch_size: int = 128
    # replay_capacity: int = 10000 #M for replay buffer capacity
    # min_replay_size: int = 1000 #minimum replay buffer size before starting optimization
    # target_sync_interval: int = 100 #C for target network sync
    # train_interval: int = 4     #N for number of environment steps between optimization updates for replay ratio
    # hidden_dim: int = 128  #set based on google search
    # epsilon_start: float = 1.0
    # epsilon_end: float = 0.05
    # epsilon_decay_steps: int = 10000
    # max_episodes: int = 50
    # eval_interval: int = 50
    # eval_games: int = 40
    # gradient_clip_norm: float = 5 # reduce to 1 if unstable
    # seed: int = 42

    def epsilon_at_step(self, step: int) -> float:
        if self.epsilon_decay_steps <= 0:
            return self.epsilon_end
        mix = min(max(step, 0) / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + mix * (self.epsilon_end - self.epsilon_start)


class ReplayBuffer:
    """
    Store DQN transitions and support random minibatch sampling.

    Inputs to `push` should be one transition:
    observation, action, reward, next observation, next legal-action mask, and done flag.
    `sample(batch_size)` should return batched NumPy arrays suitable for training.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buf: deque[tuple[np.ndarray, int, float, np.ndarray, np.ndarray, bool]] = deque(maxlen=capacity)
        # raise NotImplementedError("TODO: implement replay buffer storage")

    def __len__(self) -> int:
        return len(self.buf)
        # raise NotImplementedError("TODO: return replay buffer size")

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_legal_mask: np.ndarray,
        done: bool,
    ):
        self.buf.append((
            obs.astype(np.float32, copy=True),
            int(action), 
            float(reward), 
            next_obs.astype(np.float32, copy=True),
            next_legal_mask.astype(bool, copy=True),
            bool(done))
        )
        # raise NotImplementedError("TODO: append one transition to the replay buffer")

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buf, batch_size)
        obs, action, reward, next_obs, next_legal_mask, done = zip(*batch) # from disc 13
        return (
            np.stack(obs),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.stack(next_obs),
            np.stack(next_legal_mask),
            np.asarray(done, dtype=np.bool_),
        )
        # raise NotImplementedError("TODO: sample a minibatch of transitions")


class Connect4QNetwork(nn.Module):
    """
    Map an observation tensor to one Q-value per action.

    The forward pass should accept a tensor with shape `(batch_size, 2, rows, cols)`
    and return a tensor with shape `(batch_size, action_size)`.
    """

    def __init__(self, observation_shape: tuple[int, int, int], action_size: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        input_dim = int(np.prod(observation_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
    

       
        # raise NotImplementedError("TODO: define your Q-network layers")

        # raise NotImplementedError("TODO: return a tensor of shape (batch_size, action_size)")


def _masked_argmax(q_values: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Mask illegal actions before taking argmax."""
    masked_q = q_values.masked_fill(~legal_mask, -1e9)
    return masked_q.argmax(dim=1)


def _select_action(
    model: Connect4QNetwork,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    epsilon: float,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    """TODO: implement epsilon-greedy action selection with legal-action masking."""
    legal_actions = np.flatnonzero(legal_mask)
    if legal_actions.size == 0:
        raise RuntimeError("No legal actions available")
    if rng.random() < epsilon:
        return int(rng.choice(legal_actions))
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
    legal_tensor = torch.from_numpy(legal_mask).unsqueeze(0).to(device=device, dtype=torch.bool)
    with torch.no_grad():
        q_values = model(obs_tensor)
        action = _masked_argmax(q_values, legal_tensor)
    return int(action.item())
     # raise NotImplementedError("TODO: implement epsilon-greedy action selection")


def _apply_opponent_turn(env: Connect4Env, opponent: Agent) -> tuple[float, bool]:
    """Advance the environment through one scripted opponent move."""
    if env.done:
        raise RuntimeError("Cannot apply opponent turn to a finished game")
    action = opponent.select_action(env)
    _, _, done, info = env.step(action)
    if not done:
        return 0.0, False
    return float(info["opponent_reward"]), True


def _optimize_model(
    online_model: Connect4QNetwork,
    target_model: Connect4QNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    config: DQNConfig,
    device: torch.device,
) -> float:
    """TODO: run one DQN optimization step and return the scalar loss."""
    obs, actions, rewards, next_obs, next_masks, dones = replay_buffer.sample(config.batch_size)
    obs_tensor = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
    actions_tensor = torch.from_numpy(actions).to(device=device, dtype=torch.int64)
    rewards_tensor = torch.from_numpy(rewards).to(device=device, dtype=torch.float32)
    next_obs_tensor = torch.from_numpy(next_obs).to(device=device, dtype=torch.float32)
    next_mask_tensor = torch.from_numpy(next_masks).to(device=device, dtype=torch.bool)
    done_tensor = torch.from_numpy(dones).to(device=device, dtype=torch.bool)

    q_values = online_model(obs_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_model(next_obs_tensor)
        next_state_values = next_q_values.masked_fill(~next_mask_tensor, -1e9).max(dim=1).values
        next_state_values = torch.where(done_tensor, torch.zeros_like(next_state_values), next_state_values)
        targets = rewards_tensor + config.gamma * next_state_values

    loss = F.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_model.parameters(), max_norm=config.gradient_clip_norm)
    optimizer.step()
    return float(loss.item())
    #raise NotImplementedError("TODO: implement one DQN update step")


class DQNPolicyAgent:
    """Minimal agent wrapper expected by evaluate.py and play_connect4.py."""

    def __init__(
        self,
        model: Connect4QNetwork,
        config: Connect4Config,
        device: torch.device | str | None = None,
        name: str = "dqn",
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.name = name
        self.model.to(self.device)
        self.model.eval()

    def select_action(self, env: Connect4Env) -> int:
        if env.config != self.config:
            raise ValueError("Environment config does not match the loaded DQN checkpoint")
        obs = env.get_observation()
        legal_mask = env.legal_actions_mask()
        return _select_action(
            model=self.model,
            obs=obs,
            legal_mask=legal_mask,
            epsilon=0.0,
            device=self.device,
            rng=np.random.default_rng(0),
        )
        #raise NotImplementedError("TODO: choose an action for evaluation using your trained model")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: torch.device | str | None = None) -> "DQNPolicyAgent":
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        env_config = Connect4Config(**checkpoint["env_config"])
        hidden_dim = int(checkpoint["training_config"]["hidden_dim"])
        model = Connect4QNetwork(
            observation_shape=env_config.observation_shape,
            action_size=env_config.action_size,
            hidden_dim=hidden_dim,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, config=env_config, device=device)


def save_dqn_checkpoint(
    checkpoint_path: str | Path,
    model: Connect4QNetwork,
    env_config: Connect4Config,
    training_config: DQNConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_config": asdict(env_config),
        "training_config": asdict(training_config),
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def train_dqn(
    env_config: Connect4Config,
    training_config: DQNConfig,
    opponent: Agent,
    eval_opponents: dict[str, Agent] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(training_config.seed)

    env = Connect4Env(env_config)
    online_model = Connect4QNetwork(
        observation_shape=env_config.observation_shape,
        action_size=env_config.action_size,
        hidden_dim=training_config.hidden_dim,
    ).to(resolved_device)
    target_model = Connect4QNetwork(
        observation_shape=env_config.observation_shape,
        action_size=env_config.action_size,
        hidden_dim=training_config.hidden_dim,
    ).to(resolved_device)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(online_model.parameters(), lr=training_config.learning_rate)
    replay_buffer = ReplayBuffer(training_config.replay_capacity)
    history: list[dict[str, Any]] = []
    global_step = 0

    training_start_time = time.time()

    for episode in range(1, training_config.max_episodes + 1):
        episode_start_time = time.time()
        start_player = 1 if rng.random() < 0.5 else -1
        env.reset(start_player=start_player)
        episode_reward = 0.0
        episode_loss = 0.0
        updates = 0

        if env.current_player == -1:
            reward_from_opponent, done = _apply_opponent_turn(env, opponent)
            episode_reward += reward_from_opponent
            if done:
                history.append({
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "epsilon": training_config.epsilon_at_step(global_step),
                    "winner": env.winner,
                    "updates": updates,
                })
                continue

        while not env.done:
            obs = env.get_observation()
            legal_mask = env.legal_actions_mask()
            epsilon = training_config.epsilon_at_step(global_step)
            action = _select_action(
                model=online_model,
                obs=obs,
                legal_mask=legal_mask,
                epsilon=epsilon,
                device=resolved_device,
                rng=rng,
            )
            _, reward, done, _ = env.step(action)
            global_step += 1

            if done:
                next_obs = env.get_observation()
                next_legal_mask = env.legal_actions_mask()
            else:
                reward_from_opponent, done = _apply_opponent_turn(env, opponent)
                reward += reward_from_opponent
                next_obs = env.get_observation()
                next_legal_mask = env.legal_actions_mask()

            replay_buffer.push(obs, action, reward, next_obs, next_legal_mask, done)
            episode_reward += reward

            if len(replay_buffer) >= training_config.min_replay_size and global_step % training_config.train_interval == 0:
                loss_value = _optimize_model(
                    online_model=online_model,
                    target_model=target_model,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    config=training_config,
                    device=resolved_device,
                )
                episode_loss += loss_value
                updates += 1

            if global_step % training_config.target_sync_interval == 0:
                target_model.load_state_dict(online_model.state_dict())

        record: dict[str, Any] = {
            "episode": episode,
            "episode_reward": episode_reward,
            "epsilon": training_config.epsilon_at_step(global_step),
            "winner": env.winner,
            "updates": updates,
        }
        if updates > 0:
            record["mean_loss"] = episode_loss / updates

        if eval_opponents and episode % training_config.eval_interval == 0:
            policy_agent = DQNPolicyAgent(model=online_model, config=env_config, device=resolved_device)
            record["evaluation"] = {
                name: evaluate_agent_pair(
                    player_one=policy_agent,
                    player_two=eval_opponent,
                    games=training_config.eval_games,
                    config=env_config,
                )
                for name, eval_opponent in eval_opponents.items()
            }

        history.append(record)

    target_model.load_state_dict(online_model.state_dict())
    return {
        "model": online_model,
        "target_model": target_model,
        "history": history,
        "device": str(resolved_device),
    }
    #raise NotImplementedError("TODO: implement the DQN training loop")