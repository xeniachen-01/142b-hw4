from __future__ import annotations

import argparse
from pathlib import Path
import torch

from connect4.dqn import DQNConfig, save_dqn_checkpoint, train_dqn, DQNPolicyAgent
from connect4.env import Connect4Config
from connect4.evaluate import evaluate_agent_pair
from connect4.opponents import build_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on a configurable Connect-style game.")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--connect-n", type=int, default=4)
    parser.add_argument("--opponent", choices=["random", "heuristic"], default="random")
    parser.add_argument("--eval-opponent", choices=["random", "heuristic"], default="random")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--min-replay-size", type=int, default=1000)
    parser.add_argument("--target-sync-interval", type=int, default=250)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", default="checkpoints/connect4_dqn.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = Connect4Config(rows=args.rows, cols=args.cols, connect_n=args.connect_n)
    training_config = DQNConfig(
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay_size,
        target_sync_interval=args.target_sync_interval,
        hidden_dim=args.hidden_dim,
        epsilon_decay_steps=args.epsilon_decay_steps,
        max_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        seed=args.seed,
    )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    opponent = build_agent(args.opponent, seed=args.seed)
    eval_opponent = build_agent(args.eval_opponent, seed=args.seed + 1)
    result = train_dqn(
        env_config=env_config,
        training_config=training_config,
        opponent=opponent,
        eval_opponents={args.eval_opponent: eval_opponent},
        device=device,
    )

    checkpoint_path = Path(args.checkpoint_path)
    save_dqn_checkpoint(
        checkpoint_path=checkpoint_path,
        model=result["model"],
        env_config=env_config,
        training_config=training_config,
        metadata={
            "train_opponent": args.opponent,
            "eval_opponent": args.eval_opponent,
            "device": result["device"],
        },
    )

    policy_agent = DQNPolicyAgent.from_checkpoint(checkpoint_path, device=device)
    final_eval = evaluate_agent_pair(
        player_one=policy_agent,
        player_two=eval_opponent,
        games=args.eval_games,
        config=env_config,
    )

    print(f"Training device: {result['device']}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    if result["history"]:
        last_record = result["history"][-1]
        print(f"Final episode: {last_record['episode']}")
        print(f"Final episode reward: {last_record['episode_reward']:.3f}")
        if "mean_loss" in last_record:
            print(f"Final mean loss: {last_record['mean_loss']:.5f}")
    print(
        f"Evaluation vs {args.eval_opponent}: "
        f"wins={final_eval['player_one_wins']} "
        f"losses={final_eval['player_two_wins']} "
        f"draws={final_eval['draws']} "
        f"win_rate={final_eval['player_one_win_rate']:.3f}"
    )


if __name__ == "__main__":
    main()