import itertools
from collections import deque
import numpy as np
import torch

from rainbowarena.utils import (
    tournament,
)
from rainbowarena.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, NoisyDQNAgent, RandomAgent, LLMAgent
import pygambit as gbt
import rainbowarena

n_plc = 31  # 策略数目
n_p = 3  # 玩家数目
env = 'lama'  # 游戏类型
algorithm = 'dqn'  # 算法类型
n_eval_games = 100  # 模拟游戏局数
seed = 9  # 随机种子
np.random.seed(seed)  # np随机种子

# 初始化meta_payoff
meta_payoff = []
for _ in range(n_p):
    player_meta_payoff = np.zeros([n_plc] * n_p)
    meta_payoff.append(player_meta_payoff)

# 策略组合
policies = ['random',
            'SP/lama_sp/1.pt', 'SP/lama_sp/2.pt', 'SP/lama_sp/3.pt', 'SP/lama_sp/4.pt', 'SP/lama_sp/5.pt'
            'SP/lama_sp/6.pt', 'SP/lama_sp/7.pt', 'SP/lama_sp/8.pt', 'SP/lama_sp/9.pt', 'SP/lama_sp/10.pt',
            'SP/lama_fp/1.pt', 'SP/lama_fp/2.pt', 'SP/lama_fp/3.pt', 'SP/lama_fp/4.pt', 'SP/lama_fp/5.pt'
            'SP/lama_fp/6.pt', 'SP/lama_fp/7.pt', 'SP/lama_fp/8.pt', 'SP/lama_fp/9.pt', 'SP/lama_fp/10.pt',
            'SP/lama_psro/1.pt', 'SP/lama_psro/2.pt', 'SP/lama_psro/3.pt', 'SP/lama_psro/4.pt', 'SP/lama_psro/5.pt'
            'SP/lama_psro/6.pt', 'SP/lama_psro/7.pt', 'SP/lama_psro/8.pt', 'SP/lama_psro/9.pt', 'SP/lama_psro/10.pt',
            ]

# 所有indices
ranges = [range(n_plc)] * n_p

# indices的map
ranges_map = {}

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
    agents = [None for _ in range(n_p)]

    for i in range(n_p):
        if estimated_policies[i] == 'random':
            agents[i] = RandomAgent(num_actions=eval_env.num_actions)
        elif estimated_policies[i] == 'llm':
            agents[i] = LLMAgent()
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


for indices in itertools.product(*ranges):
    # 开始填充
    pre_indices = indices
    indices = deque(indices)
    indices_tmp = tuple(indices)

    # 填充1：原本就有的
    flag = False  # 判断是否找到

    for idx in range(n_p):
        if indices_tmp in ranges_map:
            flag = True  # 找到了
            payoffs = ranges_map[indices_tmp]
            for player_id in range(n_p):
                meta_payoff[player_id][pre_indices] = payoffs[(player_id + idx) % n_p]
            break
        indices.rotate(1)
        indices_tmp = tuple(indices)

    if flag is True:
        continue

    # 填充2：原本没有，需要模拟
    if all_elements_same_v1(pre_indices):
        winning_rates = [0.0] * n_p
    else:
        estimated_policies = []
        for indice in indices:
            estimated_policies.append(policies[indice])
        winning_rates = estimate_policies(estimated_policies)

    for player_id in range(n_p):
        meta_payoff[player_id][pre_indices] = winning_rates[player_id]
    # 更新map
    ranges_map[pre_indices] = tuple(winning_rates)


g = gbt.Game.from_arrays(*meta_payoff)
result = gbt.nash.logit_solve(g)
result = result.equilibria[0][g.players[0]]
p0_sol = [item[1] for item in result]
print(p0_sol)