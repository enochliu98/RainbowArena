
import os
import argparse

import rainbowarena
from rainbowarena.agents import (
    DQNAgent,
    RandomAgent,

)
import torch
import numpy as np
from rainbowarena.utils import (
    get_device,
    set_seed,
    tournament,
)


# def load_model(model_path, env=None, position=None, device=None):
#     if os.path.isfile(model_path):  # Torch model
#         import torch
#         agent = torch.load(model_path, map_location=device)
#         agent.set_device(device)
#     elif os.path.isdir(model_path):  # CFR model
#         from rlcard.agents import CFRAgent
#         agent = CFRAgent(env, model_path)
#         agent.load()
#     elif model_path == 'random':  # Random model
#         from rlcard.agents import RandomAgent
#         agent = RandomAgent(num_actions=env.num_actions)
#     else:  # A model in the model zoo
#         from rlcard import models
#         agent = models.load(model_path).agents[position]
#
#     return agent

def load_model(model_path, env=None, device=None, name=None):
    if name == 'dqn':
        from rainbowarena.agents import DQNAgent
        if model_path != "":
            agent = DQNAgent.from_checkpoint(checkpoint=torch.load(model_path))
        else:
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64, 64],
                device=device,
                save_path='',
                save_every=100
            )

    elif name == 'nfsp':
        from rainbowarena.agents import NFSPAgent
        if model_path != "":
            agent = NFSPAgent.from_checkpoint(checkpoint=torch.load(model_path))
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
    elif name == 'duelingdqn':
        from rainbowarena.agents import DuelingDQNAgent
        if model_path != "":
            agent = DuelingDQNAgent.from_checkpoint(checkpoint=torch.load(model_path))
        else:
            agent = DuelingDQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
    elif name == 'doubledqn':
        from rainbowarena.agents import DoubleDQNAgent
        if model_path != "":
            agent = DoubleDQNAgent.from_checkpoint(checkpoint=torch.load(model_path))
        else:
            agent = DoubleDQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )

    elif name == 'noisydqn':
        from rainbowarena.agents import NoisyDQNAgent
        if model_path != "":
            agent = NoisyDQNAgent.from_checkpoint(checkpoint=torch.load(model_path))
        else:
            agent = NoisyDQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
    elif name == 'ppo':
        from rainbowarena.agents import PPOAgent
        agent = PPOAgent(
            state_dim=np.prod(env.state_shape[0]),
            action_dim=env.num_actions,
        )
    else:
        agent = RandomAgent(num_actions=env.num_actions)

    return agent


def evaluate(args):
    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rainbowarena.make(args.env, config={'seed': args.seed})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, device, args.names[position]))
    env.set_agents(agents)

    print(agents[0].epsilons)
    print(agents[1].epsilons)
    print(agents[2].epsilons)
    print(agents[3].epsilons)



    # Evaluate
    rewards = tournament(env, args.num_games)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
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
        ],
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=[

            'SP/gongzhu_sp_new_3/10.pt',
            'SP/gongzhu_fp_new_3/10.pt',
            'SP/gongzhu_psro/10.pt',
            'SP/gongzhu_sp_new_3/10.pt',
        ],
    )

    parser.add_argument(
        '--names',
        nargs='*',
        default=[
            'dqn',
            'dqn',
            'dqn',
            'dqn'
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
        default=39,
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)

'''
--> Running on the CPU
0  0.005686195337470251
1 SP/gongzhu_sp_2/3.pt 0.12102965350234351
2 SP/gongzhu_sp_2/1.pt 0.07626848296712391
3 SP/gongzhu_sp_2/2.pt -0.2029843318069377
'''