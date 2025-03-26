import os
from colorama import init, Fore
import time
import ray
import numpy as np

print(os.listdir())

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

import json
import rainbowarena
from rainbowarena.agents.llm_agents_tmp.ticket2ride_small_v2.ticket2ride_small_agent import TicketAgent
from rainbowarena.agents.llm_agents_tmp.uno.uno_agent import UnoAgent
from rainbowarena.agents.llm_agents_tmp.mahjong.mahjong_agent import MahjongAgent
from rainbowarena.agents.llm_agents_tmp.splendor_v2.splendor_agent import SplendorAgent
from rainbowarena.agents.llm_agents_tmp.gongzhu.gongzhu_agent import GongzhuAgent
from rainbowarena.agents.llm_agents_tmp.lama.lama_agent import LamaAgent
from rainbowarena.agents.llm_agents_tmp.papayoo.papayoo_agent import PapayooAgent
from rainbowarena.agents.llm_agents_tmp.wizard.wizard_agent import WizardAgent
from rainbowarena.agents.llm_agents_tmp.random_agent import RandomAgent
import argparse

parser = argparse.ArgumentParser("Arena")

parser.add_argument(
    '--env',
    type=str,
    # default='ticket2ride_small_v2',
    # default='uno',
    # default='mahjong',
    # default= 'splendor_v2'
    # default='gongzhu'
    # default='lama'
    default= 'papayoo'
    # default='wizard'

)
parser.add_argument(
    '--seed',
    type=int,
    default=42,
)
parser.add_argument(
    '--model',
    type=str,
    default='LLM',
    choices=[
        'RL',
        'LLM',
        'Random'
    ],
)
parser.add_argument(
    '--num_process',
    type=int,
    default=8,
)

args = parser.parse_args()


@ray.remote
def ray_simulation(ray_id):
    # 初始化环境
    env = rainbowarena.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    # 创建游戏名称到Agent类的映射
    agent_class_mapping = {
        'uno': UnoAgent,
        'ticket2ride_small_v2': TicketAgent,
        'splendor_v2': SplendorAgent,
        'lama': LamaAgent,
        'mahjong': MahjongAgent,
        'gongzhu': GongzhuAgent,
        'papayoo': PapayooAgent,
        'wizard': WizardAgent
    }
    # 规定玩家
    # two players: uno, ticket2ride,splendor; three players: lama; four players: mahjong, gongzhu, papayoo, wizard
    player_count_dict = {
        'uno': 2,
        'ticket2ride_small_v2': 2,
        'splendor_v2': 2,
        'lama': 3,
        'mahjong': 4,
        'gongzhu': 4,
        'papayoo': 4,
        'wizard': 4
    }
    player_count = player_count_dict.get(args.env.lower(), 0)  # 获取游戏对应的玩家数量，默认为0
    DICT_PLAYER = {
        0: 'LLM',
        1: 'LLM',
        2: 'Random',
        3: 'Random',
    }
    if len(DICT_PLAYER) != player_count:
        raise ValueError(f"Error: The length of players ({len(DICT_PLAYER)}) "
                         f"does not match the expected number of players ({player_count}) for game '{args.env}'.")

    # 多个玩家的模型名称列表
    model_names = [
        'gpt-4o-mini',
        'glm-4-flash',
        'random',
        'random',

    ]
    if len(model_names) != player_count:
        raise ValueError(f"Error: The length of model_names ({len(model_names)}) "
                         f"does not match the expected number of players ({player_count}) for game '{args.env}'.")

    # llm配置文件
    file_path = os.path.join("../rainbowarena", "agents", "llm_agents_tmp", args.env, "info.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as file:
        info = json.load(file)

    # 配置文件实例化
    opponents = []
    for player_id, model_name in enumerate(model_names):
        if DICT_PLAYER[player_id] == 'LLM':
            info_copy = info.copy()
            info_copy['model_name'] = model_name
            agent_class = agent_class_mapping.get(args.env.lower())
            if agent_class is None:
                raise ValueError(f"No agent class found for the game '{args.env}'")
            opponent = agent_class(llm_info=info_copy)
            opponents.append(opponent)
        if DICT_PLAYER[player_id] == 'Random':
            opponent = RandomAgent()
            opponents.append(opponent)
    # 设置环境
    for opponent in opponents:
        opponent.set_env(env)

    # 游戏对战循环
    init()
    num_games = 5  # 游戏对战局数
    total_start_time = time.time()
    # 初始化胜利次数
    victory_counts = [0] * len(opponents)  # 跟踪每个模型的胜利次数
    # 初始化总分数
    total_scores = [0] * len(opponents)  # 跟踪每个模型的总分数

    for game_num in range(num_games):
        game_start_time = time.time()
        state, player_id = env.reset()  # 初始化
        round_number = game_num + 1
        act_number = 1
        act_counts = [0] * int(len(opponents))
        while not env.is_over():  # 判断结束

            if DICT_PLAYER[player_id] == 'LLM':
                action = opponents[player_id].act(player_id, state, info)
            elif DICT_PLAYER[player_id] == 'Random':
                action = opponents[player_id].step(state)

            act_counts[player_id] += 1
            act_number += 1
            next_state, next_player_id = env.step(action, False)  # 环境更新
            state = next_state
            player_id = next_player_id

        game_end_time = time.time()
        game_duration = game_end_time - game_start_time

        payoffs = env.get_payoffs()  # 结果统计

        if args.env in ['uno', 'mahjong', 'ticket2ride_small_v2', 'splendor_v2']:  # 统计胜利次数
            for player_id, payoff in enumerate(payoffs):
                if payoff == 1:
                    victory_counts[player_id] += 1
        else:  # 累加每个玩家的得分
            for player_id, payoff in enumerate(payoffs):
                total_scores[player_id] += payoff

        # 指标统计：retry中illegal_action表示幻觉；其余为两种匹配失败，表示指令遵循。
        results_file_path = os.path.join("", "results", args.env, f"ray_{ray_id}_" + "game_results.txt")
        with open(results_file_path, "a") as results_file:
            results_file.write(f"\nGame {round_number} results:\n")
            for player_id in range(len(model_names)):
                results_file.write(
                    f"Model: {model_names[player_id]}, retry counts: {opponents[player_id].retry_counts}, match_one_counts: {opponents[player_id].match_one}, match_two_counts: {opponents[player_id].match_two}, illegal_counts: {opponents[player_id].illegal_counts}, random_counts: {opponents[player_id].random_counts}, act_counts: {act_counts[player_id]}  \n")
            results_file.write(f"Payoffs: {payoffs}\n")
            results_file.write(f"Duration: {game_duration:.2f} seconds\n")
        # 重置计数器
        for opponent in opponents:
            opponent._reset_count()

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # 计算并输出胜率
    with open(results_file_path, "a") as results_file:
        for player_id in range(len(model_names)):
            if args.env in ['uno', 'mahjong', 'ticket2ride_small_v2', 'splendor_v2']:
                win_rate = victory_counts[player_id] / num_games
                results_file.write(f"Model: {model_names[player_id]} Win Rate: {win_rate:.2f}\n")
            else:
                results_file.write(f"Model: {model_names[player_id]} Total scores: {total_scores[player_id]}\n")
        results_file.write(f"\nTotal Duration for {num_games} games: {total_duration:.2f} seconds\n")

    win_rates = []
    for player_id in range(len(model_names)):
        if args.env in ['uno', 'mahjong', 'ticket2ride_small_v2', 'splendor_v2']:
            win_rate = victory_counts[player_id] / num_games
        else:
            win_rate = total_scores[player_id] / num_games
        win_rates.append(win_rate)
    win_rates = np.array(win_rates)

    return win_rates


if __name__ == '__main__':
    futures = [ray_simulation.remote(ray_id) for ray_id in range(args.num_process)]
    results = ray.get(futures)
    stacked_results = np.stack(results)
    total_results = np.mean(stacked_results, axis=0)
    print(total_results)
