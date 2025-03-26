import itertools
import multiprocessing
from collections import deque
import numpy as np
import ray
import torch
import tqdm

from rainbowarena.utils import (
    tournament
)
from rainbowarena.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, NoisyDQNAgent, RandomAgent
import pygambit as gbt
import rainbowarena
import time
import multiprocessing as mp

num_processes = 5
num_policies = 4  # 策略数目
num_players = 4  # 玩家数目
env = 'gongzhu'  # 游戏类型
algorithm = 'dqn'  # 算法类型
n_eval_games = 100  # 模拟游戏局数
seed = 9  # 随机种子
np.random.seed(seed)  # np随机种子

# 策略组合
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

policies = ['random', 'random', 'random', 'random']

# 环境
eval_env = rainbowarena.make(
    env,
    config={
        'seed': seed,
    }
)


def all_elements_same_v1(tup):
    return len(set(tup)) == 1


def estimate_policies(estimated_policies):
    # (1) 加载两个智能体对应的模型
    agents = [None for _ in range(num_players)]

    for i in range(num_players):
        if estimated_policies[i] == 'random':
            agents[i] = RandomAgent(num_actions=eval_env.num_actions)
        else:
            if algorithm == 'dqn':
                agents[i] = DQNAgent.from_checkpoint(checkpoint=torch.load(estimated_policies[i]))
            elif algorithm == 'duelingdqn':
                agents[i] = DuelingDQNAgent.from_checkpoint(checkpoint=torch.load(estimated_policies[i]))
            elif algorithm == 'doubledqn':
                agents[i] = DoubleDQNAgent.from_checkpoint(checkpoint=torch.load(estimated_policies[i]))
            elif algorithm == 'noisydqn':
                agents[i] = NoisyDQNAgent.from_checkpoint(checkpoint=torch.load(estimated_policies[i]))

    # (2) 在评估环境中设置智能体
    eval_env.set_agents(agents)

    # (3) 执行评估，返回agent_1获得的奖励
    result = tournament(eval_env, n_eval_games)

    return result


def evalute_payoff(estimated_policies_index_list):
    # 初始化meta_payoff
    meta_payoff = []
    for _ in range(num_players):
        player_meta_payoff = np.zeros([num_policies] * num_players)
        meta_payoff.append(player_meta_payoff)

    # 填充payoff
    for estimated_policies_index in estimated_policies_index_list:  # 遍历所有索引
        if all_elements_same_v1(estimated_policies_index):
            winning_rates = [0.0] * num_players
        else:
            estimated_policies = []
            for indice in estimated_policies_index:  # 遍历每个索引
                estimated_policies.append(policies[indice])
            winning_rates = estimate_policies(estimated_policies)

        current_index = deque(estimated_policies_index)
        current_index_tmp = tuple(current_index)

        for _ in range(num_players - 1):  # 移动index
            for player_id in range(num_players):
                meta_payoff[player_id][current_index_tmp] = winning_rates[player_id]

            current_index.rotate(1)
            current_index_tmp = tuple(current_index)

        for player_id in range(num_players):
            meta_payoff[player_id][current_index_tmp] = winning_rates[player_id]

    return meta_payoff


def get_all_index():
    index_list = []
    ranges = [range(num_policies)] * num_players

    for index in itertools.product(*ranges):
        current_index = deque(index)
        current_index_tmp = tuple(current_index)

        flag = False

        for _ in range(num_players):
            if current_index_tmp in index_list:
                flag = True
                break
            current_index.rotate(1)
            current_index_tmp = tuple(current_index)

        if flag is True:
            continue

        index_list.append(index)

    return index_list


def split_list(lst, n):
    n = min(n, len(lst))  # 确保 n 不超过列表长度

    # 计算每个子列表的大小
    quotient, remainder = divmod(len(lst), n)
    iterator = iter(lst)

    # 使用 islice 切分列表
    return [list(itertools.islice(iterator, quotient + (1 if i < remainder else 0))) for i in range(n)]


if __name__ == '__main__':
    estimated_policies_index_list_all = get_all_index()

    s_time = time.time()

    meta_payoff = evalute_payoff(estimated_policies_index_list_all)

    e_time = time.time()

    print("success")

    print(meta_payoff)
    print(e_time-s_time)


