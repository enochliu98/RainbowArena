''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse

import torch
import numpy as np

import rainbowarena
from rainbowarena.agents import RandomAgent
from rainbowarena.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    reorganize_ppo,
    Logger,
    plot_curve,
)
from tqdm import tqdm

def train(args):
    # Check whether gpu is available
    device = torch.device("cpu")

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rainbowarena.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        from rainbowarena.agents import DQNAgent
        if args.load_checkpoint_path != "":
            agent = DQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = DQNAgent(
                learning_rate=6e-6,
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[128, 64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )

    elif args.algorithm == 'nfsp':
        from rainbowarena.agents import NFSPAgent
        if args.load_checkpoint_path != "":
            agent = NFSPAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                hidden_layers_sizes=[64, 64],
                q_mlp_layers=[64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
    elif args.algorithm == 'duelingdqn':
        from rainbowarena.agents import DuelingDQNAgent
        if args.load_checkpoint_path != "":
            agent = DuelingDQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = DuelingDQNAgent(
                learning_rate=6e-6,
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[128, 64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
    elif args.algorithm == 'doubledqn':
        from rainbowarena.agents import DoubleDQNAgent
        if args.load_checkpoint_path != "":
            agent = DoubleDQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = DoubleDQNAgent(
                learning_rate=6e-6,
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[128, 64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )

    elif args.algorithm == 'noisydqn':
        from rainbowarena.agents import NoisyDQNAgent
        if args.load_checkpoint_path != "":
            agent = NoisyDQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = NoisyDQNAgent(
                learning_rate=6e-6,
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[128, 64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
    elif args.algorithm == 'ppo':
        from rainbowarena.agents import PPOAgent
        agent = PPOAgent(
            state_dim=np.prod(env.state_shape[0]),
            action_dim=env.num_actions,
        )
    elif args.algorithm == 'om':
        from rainbowarena.agents import OMAgent
        agent = OMAgent(
                learning_rate=6e-6,
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )

    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in tqdm(range(args.num_episodes)):

            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            if args.algorithm == 'ppo':
                trajectories = reorganize_ppo(trajectories, agent.buffer, payoffs)
                if episode % 5 == 0:
                    agent.update()
            else:
                trajectories = reorganize(trajectories, payoffs)
                for ts in trajectories[0]:
                    agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        args.num_eval_games,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    # plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='gongzhu_v2',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=50000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=500,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='RL/gongzhu_v2/dqn',
    )

    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default='',
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=10000)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)