import numpy as np
import os
import argparse
import rainbowarena
from rainbowarena.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
)
from rainbowarena.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, NoisyDQNAgent, RandomAgent, OMAgent
from tqdm import tqdm
import copy
import pygambit as gbt
import itertools
import math
from collections import deque

k = 4


def empty_list_generator(num_dimensions):
    result = []
    for _ in range(num_dimensions - 1):
        result = [result]
    return result


def all_elements_same_v1(tup):
    return len(set(tup)) == 1


class PSRO:
    def __init__(self):
        self.eval_env = None
        self.env = None
        self.Agent = None
        self._meta_strategy = None
        self._meta_payoff = None
        self._policies = None
        self._new_policies = None
        self._iterations = None
        self._window_payoff = None
        self._num_players = 4  # 玩家数目
        self.device = get_device()
        set_seed(args.seed)

    def estimate_policies(self, estimated_policies):
        # (1) 加载两个智能体对应的模型
        agents = [None for _ in range(self._num_players)]
        if args.algorithm == 'dqn':
            for i in range(self._num_players):
                agents[i] = DQNAgent.from_checkpoint(checkpoint=estimated_policies[i])
        elif args.algorithm == 'duelingdqn':
            for i in range(self._num_players):
                agents[i] = DuelingDQNAgent.from_checkpoint(checkpoint=estimated_policies[i])
        elif args.algorithm == 'doubledqn':
            for i in range(self._num_players):
                agents[i] = DoubleDQNAgent.from_checkpoint(checkpoint=estimated_policies[i])
        elif args.algorithm == 'noisydqn':
            for i in range(self._num_players):
                agents[i] = NoisyDQNAgent.from_checkpoint(checkpoint=estimated_policies[i])
        elif args.algorithm == 'om':
            for i in range(self._num_players):
                agents[i] = OMAgent.from_checkpoint(checkpoint=estimated_policies[i])

        # (2) 在评估环境中设置智能体
        self.eval_env.set_agents(agents)

        # (3) 执行评估，返回agent_1获得的奖励
        result = tournament(self.eval_env, args.num_eval_games)

        return result

    def train_oracle(self):
        # 初始化一个包含所有对手数据的map
        opponent_info_map = {}

        if len(self._window_payoff) < k:
            self._window_payoff.append([0.0, 0, len(self._policies)])
        else:
            self._window_payoff = sorted(self._window_payoff, key=lambda x: x[0], reverse=True)
            self._window_payoff.pop(-1)
            self._window_payoff.append([0.0, 0, len(self._policies)])

        print(self._window_payoff)

        if len(self._window_payoff) == k:
            print()

        current_window_policies = []
        current_window_policies_map = {}
        for idx, policy in enumerate(self._window_payoff):
            self._window_payoff[idx][0] = 0.0
            self._window_payoff[idx][1] = 0
            current_window_policies.append(policy[2])  # 策略编号加入
            current_window_policies_map[policy[2]] = idx  # map加入

        if len(self._policies) + 1 < k:
            ranges = [range(len(self._policies) + 1)] * (self._num_players - 1)
        else:
            ranges = [range(k)] * (self._num_players - 1)

        for index in itertools.product(*ranges):
            index = [current_window_policies[idx] for idx in index]
            index = tuple(index)
            opponent_info_map[index] = [[0.0 for _ in range(self._num_players)], 0]  # 胜率，次数

        opponents_list = list(opponent_info_map.keys())
        meta_strategy = [1 / len(opponents_list)] * len(opponents_list)

        # 训练一定数目的episode
        for episode in tqdm(range(args.num_episodes)):
            # (1) 采样并加载对手
            opponent_agents = [None for _ in range(self._num_players - 1)]  # 这里需要注意的是，为了进行完整填充，除了已有策略，还需要把最新策略加入其中
            sample_prob = np.abs(np.array(list(meta_strategy))) / np.sum(np.abs(np.array(list(meta_strategy))))
            sample_policy_idxs = opponents_list[np.random.choice(np.arange(len(meta_strategy)), p=sample_prob)]

            for i in range(self._num_players - 1):
                if sample_policy_idxs[i] == len(self._policies):
                    opponent_agents[i] = self.Agent
                    # print(opponent_agents[i].total_t)
                else:
                    opponent_agents[i] = DQNAgent.from_checkpoint(checkpoint=self._policies[sample_policy_idxs[i]])
                    # print(opponent_agents[i].total_t)

            agents = [self.Agent] + opponent_agents

            # (2) 在训练环境中设置智能体
            self.env.set_agents(agents)

            # (3) 采样一个episode的数据并重组数据
            trajectories, payoffs = self.env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)
            _, payoffs = self.env.run(is_training=False)

            current_policy_idxs = tuple(sample_policy_idxs)
            prior_info = opponent_info_map[current_policy_idxs]

            prior_payoff = prior_info[0]
            prior_count = prior_info[1]
            current_payoff = [(prior_payoff[i] * prior_count + payoffs[i]) / (prior_count + 1)
                              for i in range(self._num_players)]
            current_count = prior_count + 1
            opponent_info_map[current_policy_idxs][0] = current_payoff
            opponent_info_map[current_policy_idxs][1] = current_count

            for i in range(1, self._num_players):
                prior_payoff = self._window_payoff[current_window_policies_map[current_policy_idxs[i - 1]]][0]
                prior_count = self._window_payoff[current_window_policies_map[current_policy_idxs[i - 1]]][1]
                current_payoff = (prior_payoff * prior_count + payoffs[i]) / (prior_count + 1)
                current_count = prior_count + 1
                self._window_payoff[current_window_policies_map[current_policy_idxs[i - 1]]][0] = current_payoff
                self._window_payoff[current_window_policies_map[current_policy_idxs[i - 1]]][1] = current_count

            # (4) 训练
            for ts in trajectories[0]:
                self.Agent.feed(ts)

            # if episode % 100 == 0 and episode != 0:
            #     test_r_1 = self._window_payoff
            #     test_r_2 = opponent_info_map

            if episode % args.update_every == 0 and episode != 0:
                print(self._window_payoff)
                print(opponent_info_map)
                update_meta_strategy = []
                for key in opponent_info_map:
                    update_meta_strategy.append(math.exp(-opponent_info_map[key][0][0]))
                    opponent_info_map[key] = [[0.0 for _ in range(self._num_players)], 0]  # 重置
                update_meta_strategy = np.array(update_meta_strategy)
                meta_strategy = meta_strategy * update_meta_strategy
                meta_strategy = meta_strategy / np.sum(meta_strategy)

                print(meta_strategy)

    def init(self):
        # (1) 创建训练环境 (env) 和评估环境 (eval_env)
        self.env = rainbowarena.make(args.env, config={'seed': args.seed, })
        self.eval_env = rainbowarena.make(args.env, config={'seed': args.seed, })

        # (2) 创建智能体
        if args.algorithm == 'dqn':
            self.Agent = DQNAgent(
                learning_rate=6e-6,
                num_actions=self.env.num_actions,
                state_shape=self.env.state_shape[0],
                mlp_layers=[128, 64, 64],
                device=self.device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
        elif args.algorithm == 'duelingdqn':
            self.Agent = DuelingDQNAgent(
                num_actions=self.env.num_actions,
                state_shape=self.env.state_shape[0],
                mlp_layers=[64, 64],
                device=self.device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
        elif args.algorithm == 'doubledqn':
            self.Agent = DoubleDQNAgent(
                num_actions=self.env.num_actions,
                state_shape=self.env.state_shape[0],
                mlp_layers=[64, 64],
                device=self.device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
        elif args.algorithm == 'noisydqn':
            self.Agent = NoisyDQNAgent(
                num_actions=self.env.num_actions,
                state_shape=self.env.state_shape[0],
                mlp_layers=[64, 64],
                device=self.device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
        elif args.algorithm == 'om':
            self.Agent = OMAgent(
                num_actions=self.env.num_actions,
                state_shape=self.env.state_shape[0],
                mlp_layers=[64, 64],
                device=self.device,
                save_path=args.log_dir,
                save_every=args.save_every
            )

        # (3) 初始化其他参数
        self._iterations = 0  # 迭代次数
        self._initialize_policy()  # 初始化策略
        self._initialize_game_state()  # 初始化游戏状态
        self.update_meta_strategies()  # 获得meta-strategy
        self._window_payoff = [[0.0, 0, 0]]

    def _initialize_policy(self):
        """
            初始化策略集合
        """
        self._policies = []  # 原策略集合
        self._new_policies = [copy.deepcopy(self.Agent.checkpoint_attributes())]  # 新增策略集合

    def _initialize_game_state(self):
        """
            初始化meta_payoff并合并策略集合
        """
        self._meta_payoff = [np.full(tuple([len(self._policies) + 1 for _ in range(self._num_players)]), 0.0)
                             for _ in range(self._num_players)]
        self.update_empirical_gamestate()

    def update_empirical_gamestate(self):
        """
            增加新智能体，并更新游戏矩阵
        """
        # (1) 策略合并
        updated_policies = self._policies + self._new_policies
        total_number_policies = len(updated_policies)
        number_older_policies = len(self._policies)

        if args.sp_type == 'psro':
            # (2) 创建新的meta-payoff，并将原meta-payoff填充进去
            meta_payoff = [np.full(tuple([total_number_policies for _ in range(self._num_players)]), np.nan)
                           for _ in range(self._num_players)]
            older_policies_slice = tuple([slice(len(self._policies)) for _ in range(self._num_players)])
            for idx in range(self._num_players):
                meta_payoff[idx][older_policies_slice] = self._meta_payoff[idx]

            # (3) 填充其他元素
            import itertools
            from collections import deque
            ranges = [range(total_number_policies)] * (self._num_players)

            ranges_map = {}

            for indices in itertools.product(*ranges):
                # 判断是否需要填充
                flag = False
                for indice in indices:
                    if indice == number_older_policies:
                        flag = True
                        break
                if flag is False:
                    continue

                # 开始填充
                pre_indices = indices
                indices = deque(indices)
                indices_tmp = tuple(indices)

                # 填充1：原本就有的
                flag = False  # 判断是否找到

                for idx in range(self._num_players):
                    if indices_tmp in ranges_map:
                        flag = True  # 找到了
                        payoffs = ranges_map[indices_tmp]
                        for player_id in range(self._num_players):
                            meta_payoff[player_id][pre_indices] = payoffs[(player_id + idx) % self._num_players]
                        break
                    indices.rotate(1)
                    indices_tmp = tuple(indices)

                if flag is True:
                    continue

                # 填充2：原本没有，需要模拟
                if all_elements_same_v1(pre_indices):
                    winning_rates = [0.0] * self._num_players
                else:
                    estimated_policies = []
                    for indice in indices:
                        estimated_policies.append(updated_policies[indice])
                    winning_rates = self.estimate_policies(estimated_policies)

                for player_id in range(self._num_players):
                    meta_payoff[player_id][pre_indices] = winning_rates[player_id]
                # 更新map
                ranges_map[pre_indices] = tuple(winning_rates)

        # (4) 更新
        if args.sp_type == 'psro':
            self._meta_payoff = meta_payoff
        self._policies = updated_policies

        return self._meta_payoff

    def update_agents(self):
        """
            训练新智能体
        """
        self.train_oracle()  # 智能体训练
        self._new_policies = [copy.deepcopy(self.Agent.checkpoint_attributes())]

    def update_meta_strategies(self, ):
        """
            更新元博弈策略
        """
        if args.sp_type == 'psro':
            g = gbt.Game.from_arrays(*self._meta_payoff)
            result = gbt.nash.logit_solve(g)
            result = result.equilibria[0][g.players[0]]
            p0_sol = [item[1] for item in result]
            self._meta_strategy = p0_sol
        elif args.sp_type == 'self_play':
            self._meta_strategy = [0.0] * (len(self._policies) - 1) + [1.0]
        elif args.sp_type == 'fictious_play':
            values = [1.0] * len(self._policies)
            total_sum = sum(values)
            self._meta_strategy = [value / total_sum for value in values]
        elif args.sp_type == 'super_self_play':
            self._meta_strategy = None
        else:
            self._meta_strategy = None

    def iteration(self):
        """
            PSRO迭代
        """
        self._iterations += 1  # 迭代次数加一

        self.update_agents()  # (1) BR求解

        self.update_empirical_gamestate()  # (2) 更新游戏矩阵
        print(self._meta_payoff)

        self.update_meta_strategies()  # (3) 求解纳什均衡
        print(self._meta_strategy)

        self.Agent.save_checkpoint(path=args.log_dir, filename=str(self._iterations) + '.pt')  # (4) 保存模型

        # (5) 重新加载模型
        if args.sp_type == 'super_self_play':
            if len(self._policies) > k:
                new_k = len(self._policies)
                weights = [1 / new_k] * new_k
                self.Agent.from_checkpoints(self._policies, weights)
            else:
                weights = [1 / k] * k
                self.Agent.from_checkpoints(self._policies[-k:], weights)

            self.Agent.total_t = 0

    def train_psro(self, psro_loop):
        for i in range(psro_loop):
            print('')
            print('----------------------------------------')
            print('  PSRO_LOOP      |  ' + str(i))
            print('----------------------------------------')
            self.iteration()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PSRO example in RLCard")
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
        default=200,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--update_every',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='SP/gongzhu_ssp_5/',
    )
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default='',
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10000
    )
    parser.add_argument(
        '--sp_type',
        type=str,
        default='super_self_play',
        choices=[
            'self_play',
            'fictious_play',
            'psro'
        ],
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    psro = PSRO()
    psro.init()
    psro.train_psro(50)
