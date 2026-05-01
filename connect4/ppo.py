from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

from .env import Connect4Config, Connect4Env
from .opponents import Agent
from .evaluate import evaluate_agent_pair


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    #gae_lambda: float = 0.95 -> value from first and second run
    #gae_lambda: float = 0.8 -> value from third and fourth run
    #gae_lambda: float = 0.2 -> value from fifth run
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    #rollout_steps: int = 512 -> value from first three runs
    rollout_steps: int = 256
    #update_epochs: int = 4
    update_epochs: int = 3
    #minibatch_size: int = 128 -> value from first three runs
    minibatch_size: int = 64
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    #hidden_dim: int = 128 -> value from first five runs
    hidden_dim: int = 32
    #max_updates: int = 400 -> value from first run
    max_updates: int = 200
    eval_interval: int = 20
    eval_games: int = 40
    seed: int = 42


class Connect4ActorCritic(nn.Module):
    """
    Map an observation tensor to policy logits and a state-value estimate.

    The forward pass should accept a tensor with shape `(batch_size, 2, rows, cols)`.
    It should return a pair `(policy_logits, state_values)`, where the shapes are
    `(batch_size, action_size)` and `(batch_size,)` respectively.
    """

    def __init__(self, observation_shape: tuple[int, int, int], action_size: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.observation_shape = observation_shape
        input_dim = int(np.prod(observation_shape))
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_size)
        self.value_head = nn.Linear(hidden_dim, 1)
        # raise NotImplementedError("TODO: define your PPO policy/value network layers")

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # raise NotImplementedError("TODO: return (policy_logits, state_values)")
        features = self.backbone(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


def _masked_distribution(logits: torch.Tensor, legal_mask: torch.Tensor) -> Categorical:
    """Mask illegal actions before building a categorical policy."""
    masked_logits = logits.masked_fill(~legal_mask, -1e9)
    return Categorical(logits=masked_logits)


def _apply_opponent_turn(env: Connect4Env, opponent: Agent) -> tuple[float, bool]:
    """Advance the environment through one scripted opponent move."""
    if env.done:
        raise RuntimeError("Cannot apply opponent turn to a finished game")
    action = opponent.select_action(env)
    _, _, done, info = env.step(action)
    if not done:
        return 0.0, False
    return float(info["opponent_reward"]), True


def _advance_to_agent_turn(env: Connect4Env, opponent: Agent) -> bool:
    """TODO: keep stepping until it is the learning agent's turn again."""
    # raise NotImplementedError("TODO: advance past the opponent turn when needed")
    if env.done:
        return True
    if env.current_player == -1:
        _, done = _apply_opponent_turn(env, opponent)
        return done
    return False


def _sample_action(
    model: Connect4ActorCritic,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> tuple[int, float, float]:
    """TODO: sample an action and return action, log-probability, and value."""
    # raise NotImplementedError("TODO: implement PPO action sampling")
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
    legal_tensor = torch.from_numpy(legal_mask).unsqueeze(0).to(device=device, dtype=torch.bool)
    with torch.no_grad():
        logits, value = model(obs_tensor)
        dist = _masked_distribution(logits, legal_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    return action.item(), float(log_prob.item()), float(value.item())


def _greedy_action(
    model: Connect4ActorCritic,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> int:
    """TODO: choose the evaluation-time action from the policy."""
    # raise NotImplementedError("TODO: implement greedy evaluation action selection")
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
    legal_tensor = torch.from_numpy(legal_mask).unsqueeze(0).to(device=device, dtype=torch.bool)
    with torch.no_grad():
        logits, _ = model(obs_tensor)
        # this use "dist" instead of "masked_logits"
        #dist = _masked_distribution(logits, legal_tensor)
        masked_logits = logits.masked_fill(~legal_tensor, -1e9)
        # action = dist.mode()
        action = masked_logits.argmax(dim=1)
    return int(action.item())


def _bootstrap_value(
    model: Connect4ActorCritic,
    obs: np.ndarray,
    device: torch.device,
) -> float:
    """TODO: estimate the value of the final state in a rollout."""
    # raise NotImplementedError("TODO: implement bootstrap value computation")
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        _, value = model(obs_tensor)
    return float(value.item())


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    config: PPOConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """TODO: compute generalized advantage estimates and returns."""
    # raise NotImplementedError("TODO: implement GAE")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_advantage = 0.0
    next_value = last_value
    for step in range(len(rewards) - 1, -1, -1):
        nonterminal = 1.0 - float(dones[step])
        delta = rewards[step] + config.gamma * next_value * nonterminal - values[step]
        next_advantage = delta + config.gamma * config.gae_lambda * nonterminal * next_advantage
        advantages[step] = next_advantage
        next_value = values[step]
    returns = advantages + values
    return advantages, returns


class PPOPolicyAgent:
    """Minimal agent wrapper expected by evaluate.py and play_connect4.py."""

    def __init__(
        self,
        model: Connect4ActorCritic,
        config: Connect4Config,
        device: torch.device | str | None = None,
        name: str = "ppo",
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.name = name
        self.model.to(self.device)
        self.model.eval()

    def select_action(self, env: Connect4Env) -> int:
        # raise NotImplementedError("TODO: choose an action for evaluation using your trained model")
        obs = env.get_observation()
        legal_mask = env.legal_actions_mask()
        return _greedy_action(self.model, obs, legal_mask, self.device)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: torch.device | str | None = None) -> "PPOPolicyAgent":
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        env_config = Connect4Config(**checkpoint["env_config"])
        hidden_dim = int(checkpoint["training_config"]["hidden_dim"])
        model = Connect4ActorCritic(
            observation_shape=env_config.observation_shape,
            action_size=env_config.action_size,
            hidden_dim=hidden_dim,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, config=env_config, device=device)


def save_ppo_checkpoint(
    checkpoint_path: str | Path,
    model: Connect4ActorCritic,
    env_config: Connect4Config,
    training_config: PPOConfig,
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


def train_ppo(
    env_config: Connect4Config,
    training_config: PPOConfig,
    opponent: Agent,
    eval_opponents: dict[str, Agent] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    # raise NotImplementedError("TODO: implement the PPO training loop")
    np.random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(training_config.seed)
    env = Connect4Env(env_config)
    model = Connect4ActorCritic(
        observation_shape=env_config.observation_shape,
        action_size=env_config.action_size,
        hidden_dim=training_config.hidden_dim,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    history: list[dict[str, Any]] = []
    env.reset(start_player=1 if rng.random() < 0.5 else -1)
    while _advance_to_agent_turn(env, opponent):
        env.reset(start_player=1 if rng.random() < 0.5 else -1)

    for update in range(1, training_config.max_updates + 1):
        obs_buffer = np.zeros((training_config.rollout_steps, *env_config.observation_shape), dtype=np.float32)
        legal_buffer = np.zeros((training_config.rollout_steps, env_config.action_size), dtype=bool)
        action_buffer = np.zeros(training_config.rollout_steps, dtype=np.int64)
        logprob_buffer = np.zeros(training_config.rollout_steps, dtype=np.float32)
        reward_buffer = np.zeros(training_config.rollout_steps, dtype=np.float32)
        done_buffer = np.zeros(training_config.rollout_steps, dtype=np.bool_)
        value_buffer = np.zeros(training_config.rollout_steps, dtype=np.float32)

        episode_rewards: list[float] = []
        episode_reward = 0.0

        for step in range(training_config.rollout_steps):
            obs = env.get_observation()
            legal_mask = env.legal_actions_mask()
            action, log_prob, value = _sample_action(model, obs, legal_mask, resolved_device)
         
            obs_buffer[step] = obs
            legal_buffer[step] = legal_mask
            action_buffer[step] = action
            logprob_buffer[step] = log_prob
            value_buffer[step] = value

            _, reward, done, _ = env.step(action)
            if not done:
                reward_from_opponent, done = _apply_opponent_turn(env, opponent)
                reward += reward_from_opponent

            reward_buffer[step] = reward
            done_buffer[step] = done
            episode_reward += reward
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                env.reset(start_player=1 if rng.random() < 0.5 else -1)
                while _advance_to_agent_turn(env, opponent):
                    episode_rewards.append(episode_reward)
                    episode_reward = 0.0
                    env.reset(start_player=1 if rng.random() < 0.5 else -1)

        if env.done:
            last_value = 0.0
        else:
            last_value = _bootstrap_value(model, env.get_observation(), resolved_device)

        advantages, returns = _compute_gae(
            rewards=reward_buffer,
            values=value_buffer,
            dones=done_buffer,
            last_value=last_value,
            config=training_config,
        )

        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_tensor = torch.from_numpy(obs_buffer).to(device=resolved_device, dtype=torch.float32)
        legal_tensor = torch.from_numpy(legal_buffer).to(device=resolved_device, dtype=torch.bool)
        action_tensor = torch.from_numpy(action_buffer).to(device=resolved_device, dtype=torch.int64)
        old_logprob_tensor = torch.from_numpy(logprob_buffer).to(device=resolved_device, dtype=torch.float32)
        advantage_tensor = torch.from_numpy(normalized_advantages).to(device=resolved_device, dtype=torch.float32)
        return_tensor = torch.from_numpy(returns).to(device=resolved_device, dtype=torch.float32)

        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        minibatch_count = 0

        batch_indices = np.arange(training_config.rollout_steps)
        for _ in range(training_config.update_epochs):
            rng.shuffle(batch_indices)
            for start in range(0, training_config.rollout_steps, training_config.minibatch_size):
                end = start + training_config.minibatch_size
                mb_idx = batch_indices[start:end]

                logits, values = model(obs_tensor[mb_idx])
                dist = _masked_distribution(logits, legal_tensor[mb_idx])
                new_log_prob = dist.log_prob(action_tensor[mb_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_prob - old_logprob_tensor[mb_idx])
                unclipped = ratio * advantage_tensor[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - training_config.clip_coef, 1.0 + training_config.clip_coef) * advantage_tensor[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, return_tensor[mb_idx])
                loss = policy_loss + training_config.value_coef * value_loss - training_config.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_config.max_grad_norm)
                optimizer.step()

                policy_loss_total += float(policy_loss.item())
                value_loss_total += float(value_loss.item())
                entropy_total += float(entropy.item())
                minibatch_count += 1

        record: dict[str, Any] = {
            "update": update,
            "mean_reward": float(reward_buffer.mean()),
            "episodes_completed": len(episode_rewards),
            "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "policy_loss": policy_loss_total / max(minibatch_count, 1),
            "value_loss": value_loss_total / max(minibatch_count, 1),
            "entropy": entropy_total / max(minibatch_count, 1),
        }

        if eval_opponents and update % training_config.eval_interval == 0:
            policy_agent = PPOPolicyAgent(model=model, config=env_config, device=resolved_device)
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

    return {
        "model": model,
        "history": history,
        "device": str(resolved_device),
    }