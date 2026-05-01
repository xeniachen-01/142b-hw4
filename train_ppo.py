from __future__ import annotations

import argparse
from pathlib import Path

import torch

from connect4.env import Connect4Config
from connect4.evaluate import evaluate_agent_pair
from connect4.opponents import build_agent
from connect4.ppo import PPOConfig, PPOPolicyAgent, save_ppo_checkpoint, train_ppo

import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on a configurable Connect-style game.")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--connect-n", type=int, default=4)
    parser.add_argument("--opponent", choices=["random", "heuristic"], default="random")
    parser.add_argument("--eval-opponent", choices=["random", "heuristic"], default="random")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", default="checkpoints/connect4_ppo.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = Connect4Config(rows=args.rows, cols=args.cols, connect_n=args.connect_n)
    training_config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        max_updates=args.updates,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        seed=args.seed,
    )
    device = torch.device("cpu")


    opponent = build_agent(args.opponent, seed=args.seed)
    eval_opponent = build_agent(args.eval_opponent, seed=args.seed + 1)

    # before train_dqn call:
    t0 = time.time()

    result = train_ppo(
        env_config=env_config,
        training_config=training_config,
        opponent=opponent,
        eval_opponents={args.eval_opponent: eval_opponent},
        device=device,
    )
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.0f}s  ({elapsed/60:.1f} min)")

    checkpoint_path = Path(args.checkpoint_path)
    save_ppo_checkpoint(
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

    policy_agent = PPOPolicyAgent.from_checkpoint(checkpoint_path, device=device)
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
        print(f"Final update: {last_record['update']}")
        print(f"Mean rollout reward: {last_record['mean_reward']:.3f}")
        print(f"Mean episode reward: {last_record['mean_episode_reward']:.3f}")
        print(f"Policy loss: {last_record['policy_loss']:.5f}")
        print(f"Value loss: {last_record['value_loss']:.5f}")
        print(f"Entropy: {last_record['entropy']:.5f}")
    print(
        f"Evaluation vs {args.eval_opponent}: "
        f"wins={final_eval['player_one_wins']} "
        f"losses={final_eval['player_two_wins']} "
        f"draws={final_eval['draws']} "
        f"win_rate={final_eval['player_one_win_rate']:.3f}"
    )


if __name__ == "__main__":
    main()