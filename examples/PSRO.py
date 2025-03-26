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
from rainbowarena.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, NoisyDQNAgent, RandomAgent
from tqdm import tqdm
import copy
import pygambit as gbt
import nashpy

def empty_list_generator(num_dimensions):
    result = []
    for _ in range(num_dimensions - 1):
        result = [result]
    return result


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
        self._num_players = 2  # 玩家数目
        self.device = get_device()
        set_seed(args.seed)

    def estimate_policies(self, estimated_policies):
        # (1) 加载两个智能体对应的模型
        agent_1, agent_2 = None, None
        if args.algorithm == 'dqn':
            agent_1 = DQNAgent.from_checkpoint(checkpoint=estimated_policies[0])
            agent_2 = DQNAgent.from_checkpoint(checkpoint=estimated_policies[1])
        elif args.algorithm == 'duelingdqn':
            agent_1 = DuelingDQNAgent.from_checkpoint(checkpoint=estimated_policies[0])
            agent_2 = DuelingDQNAgent.from_checkpoint(checkpoint=estimated_policies[1])
        elif args.algorithm == 'doubledqn':
            agent_1 = DoubleDQNAgent.from_checkpoint(checkpoint=estimated_policies[0])
            agent_2 = DoubleDQNAgent.from_checkpoint(checkpoint=estimated_policies[1])
        elif args.algorithm == 'noisydqn':
            agent_1 = NoisyDQNAgent.from_checkpoint(checkpoint=estimated_policies[0])
            agent_2 = NoisyDQNAgent.from_checkpoint(checkpoint=estimated_policies[1])

        # (2) 在评估环境中设置智能体
        self.eval_env.set_agents([agent_1, agent_2])

        # (3) 执行评估，返回agent_1获得的奖励
        result = tournament(self.eval_env, args.num_eval_games)[0]

        return result

    def train_oracle(self, meta_strategy):
        # 训练一定数目的episode
        for episode in tqdm(range(args.num_episodes)):
            # (1) 采样并加载对手
            opponent_agent = None

            sample_prob = np.abs(np.array(list(meta_strategy))) / np.sum(np.abs(np.array(list(meta_strategy))))

            sample_policy_idx = np.random.choice(np.arange(len(meta_strategy)), p=sample_prob)

            if args.algorithm == 'dqn':
                opponent_agent = DQNAgent.from_checkpoint(checkpoint=self._policies[sample_policy_idx])
            elif args.algorithm == 'duelingdqn':
                opponent_agent = DuelingDQNAgent.from_checkpoint(checkpoint=self._policies[sample_policy_idx])
            elif args.algorithm == 'doubledqn':
                opponent_agent = DoubleDQNAgent.from_checkpoint(checkpoint=self._policies[sample_policy_idx])
            elif args.algorithm == 'noisydqn':
                opponent_agent = NoisyDQNAgent.from_checkpoint(checkpoint=self._policies[sample_policy_idx])
            else:
                opponent_agent = RandomAgent(num_actions=self.env.num_actions)
            # print('\n  opponent_agent      |  ' + str(sample_policy_idx))

            # (2) 在训练环境中设置智能体
            self.env.set_agents([self.Agent, opponent_agent])

            # (3) 采样一个episode的数据并重组数据
            trajectories, payoffs = self.env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)

            # (4) 训练
            for ts in trajectories[0]:
                self.Agent.feed(ts)

            # (5) 评估
            # if episode % args.evaluate_every == 0:
            #     result_tmp = tournament(self.env, 3000)[0]
            #     print('\n  evaluation      |  ' + str(result_tmp))


    def init(self):
        # (1) 创建训练环境 (env) 和评估环境 (eval_env)
        self.env = rainbowarena.make(
            args.env,
            config={
                'seed': args.seed,
            }
        )

        self.eval_env = rainbowarena.make(
            args.env,
            config={
                'seed': args.seed,
            }
        )

        # (2) 创建智能体
        if args.algorithm == 'dqn':
            self.Agent = DQNAgent(
                num_actions=self.env.num_actions,
                state_shape=self.env.state_shape[0],
                mlp_layers=[64, 64],
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

        # (3) 初始化其他参数
        self._iterations = 0  # 迭代次数
        self._initialize_policy()  # 初始化策略
        self._initialize_game_state()  # 初始化游戏状态
        self.update_meta_strategies()  # 获得meta-strategy

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
        self._meta_payoff = np.array(empty_list_generator(self._num_players))
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
            meta_payoff = np.full(tuple([total_number_policies, total_number_policies]), np.nan)
            older_policies_slice = tuple([slice(len(self._policies)) for _ in range(self._num_players)])
            meta_payoff[older_policies_slice] = self._meta_payoff

            # (3) 填充其他元素
            for current_index in range(total_number_policies):
                index_tuple_1 = (number_older_policies, current_index)
                index_tuple_2 = (current_index, number_older_policies)

                if index_tuple_1[0] == index_tuple_1[1]:
                    winning_rate = 0
                else:
                    estimated_policies = [updated_policies[index_tuple_1[0]],
                                          updated_policies[index_tuple_1[1]]]
                    winning_rate = self.estimate_policies(estimated_policies)

                meta_payoff[index_tuple_1] = winning_rate
                meta_payoff[index_tuple_2] = -winning_rate

        # (4) 更新
        if args.sp_type == 'psro':
            self._meta_payoff = meta_payoff
        self._policies = updated_policies

        return self._meta_payoff

    def update_agents(self):
        """
            训练新智能体
        """
        meta_strategy = self._meta_strategy  # 元策略
        self.train_oracle(meta_strategy)  # 智能体训练
        self._new_policies = [copy.deepcopy(self.Agent.checkpoint_attributes())]

    def update_meta_strategies(self, ):
        """
            更新元博弈策略
        """
        if args.sp_type == 'psro':
            # print(self._meta_payoff)
            # p0_sol, _, _, _ = (
            #     self._meta_solver.solve_zero_sum_matrix_game(
            #         pyspiel.create_matrix_game(
            #             self._meta_payoff,
            #             -self._meta_payoff)))
            # # zero_sum_matrix = nashpy.Game(self._meta_payoff)
            # # p0_sol, p1_sol = zero_sum_matrix.lemke_howson(initial_dropped_label=0)
            # self._meta_strategy = list(p0_sol)
            # print(p0_sol)
            g = gbt.Game.from_arrays(self._meta_payoff)
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

        self.Agent.save_checkpoint(path=args.log_dir, filename=str(self._iterations)+'.pt')


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
        default='ticket2ride_small_v2',
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
        default=1000,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='SP/ticket2ride_sp/',
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
        default='self_play',
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
    psro.train_psro(10)
