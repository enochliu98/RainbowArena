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

num_players = 2
num_policies = 6
seed = 9
num_games = 200
set_seed(seed)
env = rainbowarena.make('splendor_v2', config={'seed': seed})

# policies = ['random',
#             'SP/gongzhu_sp/1.pt', 'SP/gongzhu_sp/2.pt', 'SP/gongzhu_sp/3.pt', 'SP/gongzhu_sp/4.pt',
#             'SP/gongzhu_sp/5.pt',
#             'SP/gongzhu_sp/6.pt', 'SP/gongzhu_sp/7.pt', 'SP/gongzhu_sp/8.pt', 'SP/gongzhu_sp/9.pt',
#             'SP/gongzhu_sp/10.pt',
#             'SP/gongzhu_fp/1.pt', 'SP/gongzhu_fp/2.pt', 'SP/gongzhu_fp/3.pt', 'SP/gongzhu_fp/4.pt',
#             'SP/gongzhu_fp/5.pt',
#             'SP/gongzhu_fp/6.pt', 'SP/gongzhu_fp/7.pt', 'SP/gongzhu_fp/8.pt', 'SP/gongzhu_fp/9.pt',
#             'SP/gongzhu_fp/10.pt',
#             'SP/gongzhu_psro/1.pt', 'SP/gongzhu_psro/2.pt', 'SP/gongzhu_psro/3.pt', 'SP/gongzhu_psro/4.pt',
#             'SP/gongzhu_psro/5.pt',
#             'SP/gongzhu_psro/6.pt', 'SP/gongzhu_psro/7.pt', 'SP/gongzhu_psro/8.pt', 'SP/gongzhu_psro/9.pt',
#             'SP/gongzhu_psro/10.pt',
#             ]

policies = ['random',
            'RL/splendor/doubledqn/checkpoint_dqn.pt',
            'RL/splendor/dqn/checkpoint_dqn.pt',
            'RL/splendor/duelingdqn/checkpoint_dqn.pt',
            'RL/splendor/noisydqn/checkpoint_dqn.pt',
            'RL/splendor/ppo_tmp/model.pth']
algorithms = ['random',
              'doubledqn',
              'dqn',
              'duelingdqn',
              'noisydqn',
              'ppo']


class EloRatingSystem:
    def __init__(self, K=30):
        self.K = K  # K因子
        self.ratings = {}  # 保存玩家评分的字典

    def add_player(self, player_name, initial_rating=1500):
        """添加新玩家，初始评分为1500"""
        self.ratings[player_name] = initial_rating

    def get_expected_score(self, player_rating, opponent_rating):
        """计算预期得分"""
        return 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))

    def update_ratings(self, player1, player2, result):
        """
        更新评分
        player1, player2: 玩家名称
        result: 1表示player1胜，0表示player2胜，0.5表示平局
        """
        R1 = self.ratings[player1]
        R2 = self.ratings[player2]

        E1 = self.get_expected_score(R1, R2)
        E2 = self.get_expected_score(R2, R1)

        self.ratings[player1] = R1 + self.K * (result - E1)
        self.ratings[player2] = R2 + self.K * ((1 - result) - E2)

    def get_rating(self, player_name):
        """获取玩家评分"""
        return self.ratings.get(player_name, None)


def load_model(model_path, name=None):
    if name == 'random':
        from rainbowarena.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    if name == 'dqn':
        from rainbowarena.agents import DQNAgent
        agent = DQNAgent.from_checkpoint(checkpoint=torch.load(model_path))
    elif name == 'duelingdqn':
        from rainbowarena.agents import DuelingDQNAgent
        agent = DuelingDQNAgent.from_checkpoint(checkpoint=torch.load(model_path))
    elif name == 'doubledqn':
        from rainbowarena.agents import DoubleDQNAgent
        agent = DoubleDQNAgent.from_checkpoint(checkpoint=torch.load(model_path))
    elif name == 'noisydqn':
        from rainbowarena.agents import NoisyDQNAgent
        agent = NoisyDQNAgent.from_checkpoint(checkpoint=torch.load(model_path))
    elif name == 'ppo':
        agent = torch.load(model_path)
    return agent


def evaluate(player_1, player_2):
    agent_player_1 = load_model(policies[player_1], algorithms[player_1])
    agent_player_2 = load_model(policies[player_2], algorithms[player_2])

    agents = [agent_player_2,
              agent_player_1]
    env.set_agents(agents)
    rewards_1 = tournament(env, num_games)

    agents = [agent_player_1,
              agent_player_2]
    env.set_agents(agents)
    rewards_2 = tournament(env, num_games)

    result = rewards_1[1] + rewards_2[0]  # 第一次，玩家1在位置0；第二次，玩家1在位置1，两次进行相加

    return result


def main():
    elo_system = EloRatingSystem(K=30)

    # 示例玩家列表
    players = list(range(num_policies))
    for player in players:
        elo_system.add_player(player)

    # 进行比赛并更新ELO分数
    # 进行多轮对抗，确保一对玩家都对抗过相同次数，且确保有玩家长时间没有比赛可打的情况
    for round in range(10):
        for player_1 in players:
            for player_2 in players:
                if player_1 != player_2:
                    # 评估若干局，但是更新elo时只当做一局进行更新，以减少对应的随机性，另外，在评估的时候，需要不断更换位置进行更新
                    result = evaluate(player_1, player_2)
                    if result > 0:
                        elo_system.update_ratings(player_1, player_2, 1)
                    elif result < 0:
                        elo_system.update_ratings(player_1, player_2, 0)
                    else:
                        elo_system.update_ratings(player_1, player_2, 0.5)
    # 打印更新后的ELO分数
    for player in players:
        print(f"{algorithms[player]}的最终评分: {elo_system.get_rating(player):.2f}")


if __name__ == "__main__":
    main()
