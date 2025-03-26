# -*- coding: utf-8 -*-
import functools
from heapq import merge
import numpy as np

from rainbowarena.games.century import Player
from rainbowarena.games.century import Round
from rainbowarena.games.century import Judger


class CenturyGame:

    def __init__(self, num_players=2):
        self.winner_id = None
        self.players = None
        self.round = None
        self.round_over_player = None
        self.state = None
        self.judger = None
        self.history_action = None
        self.np_random = np.random.RandomState()
        self.num_players = num_players
        self.payoffs = [0 for _ in range(self.num_players)]
        self.Num_upgrade_card = 2  # 可选的只有2和3升级，但是2卡有多张
        self.Num_trade_card = 34
        self.Num_token_card = 13
        self.trade_num = 3  # 交易次数

    def init_game(self):
        """
        信息初始化：
        ① 玩家
        ② 回合
        ③ 裁判
        ④ 当前玩家
        ⑤ 其他信息

        Returns:
            dict: first state in one game
            int: current player's id
        """
        # 【① 初始化玩家】
        self.players = [Player(player_id) for player_id in range(self.num_players)]

        # 【② 初始化回合】
        self.round = Round(self.np_random, self.num_players, self.Num_upgrade_card, self.Num_trade_card,
                           self.Num_token_card, self.trade_num, self.players)
        self.round_over_player = (self.round.current_player + self.num_players - 1) % self.num_players

        # 【③ 初始化裁判】
        self.judger = Judger()

        # 【④ 初始化当前玩家信息】
        player_id = self.round.current_player
        state = self.get_state(player_id)
        self.state = state

        # 【⑤ 初始化其他信息】
        self.history_action = []
        self.winner_id = None  # 胜者
        self.payoffs = [0 for _ in range(self.num_players)]

        return state, player_id

    def step(self, action):
        """
            执行一次行动
            ① 当前玩家
            ② 记录历史动作信息
            ③ 执行动作
            ④ 判断回合结束
            ⑤ 获取下一个玩家信息
            ⑥ 判断下一个玩家是否有可行动作
        """

        # 【① 获取当前玩家】
        player = self.players[self.round.current_player]

        # 【② 记录历史动作信息】
        self.history_action.append((self.round.current_player, self.extract_action_state_baseline(action, player)))

        # 【③ 执行动作】
        self.round.proceed_round(player, action)

        # 【④ 判断回合结束】
        if self.round.current_player == self.round_over_player:  # 当前回合结束
            winner_id = self.judger.judge_winner(self.players, self.round_over_player)
            if winner_id != -1:  # 不等于-1， 有胜者
                self.winner_id = winner_id

        # 【⑤ 获取下一个玩家信息】
        next_id = (player.player_id + 1) % len(self.players)  # 下一个玩家
        self.round.current_player = next_id  # 下一个玩家
        state = self.get_state(next_id)  # 获取新状态
        self.state = state  # 获取新状态

        # 【⑥ 判断下一个玩家是否有可行动作】
        tmp_legal_actions = self.get_legal_actions()  # 当无行动可执行时，终止
        if len(tmp_legal_actions) == 0:
            self.winner_id = (next_id + 1) % self.num_players
            # print(tmp_legal_actions)
            # print('bad')

        return state, next_id

    def get_state(self, player_id):
        """
            获取玩家状态
        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        """
        state = {}

        # 个人信息
        state["self_function_cards"] = self.players[player_id].function_cards
        state["self_played_function_cards"] = self.players[player_id].played_function_cards
        state["self_map_function_cards"] = self.players[player_id].map_function_cards
        state["self_point_cards"] = self.players[player_id].point_cards
        state["self_tokens"] = self.players[player_id].tokens
        state["self_point"] = self.players[player_id].point
        # 其他玩家信息
        state["other_function_cards"] = self.players[1 - player_id].function_cards
        state["other_played_function_cards"] = self.players[1 - player_id].played_function_cards
        state["other_map_function_cards"] = self.players[1 - player_id].map_function_cards
        state["other_point_cards"] = self.players[1 - player_id].point_cards
        state["other_tokens"] = self.players[1 - player_id].tokens
        state["other_point"] = self.players[1 - player_id].point
        # 牌桌信息
        state["deck_function_cards"] = self.round.dealer.deck[0]
        state["deck_function_cards_token"] = self.round.dealer.deck[1]
        state["deck_point_cards"] = self.round.dealer.deck[2]
        state["deck_point_cards_bonus"] = self.round.dealer.deck[3]
        # 区分玩家的信息
        state["self_player"] = player_id
        # 历史动作信息
        state["history_action"] = self.history_action
        # 隐藏信息
        state["invisible_info_function_cards"] = self.round.dealer.function_cards  # 牌是看不到的
        state["invisible_info_point_cards"] = self.round.dealer.point_cards
        # 其他信息
        state["all_function_cards"] = self.round.dealer.all_function_cards

        return state

    def get_legal_actions(self):
        """
            获取可行动作
        """
        player = self.players[self.round.current_player]  # 获取当前玩家
        return self.round.get_legal_actions(player)

    def get_payoffs(self):
        """
            获取结果
        Returns:
            (list): Each entry corresponds to the payoff of one player
        """
        winner = self.winner_id
        if winner is not None:
            for i in range(self.num_players):
                self.payoffs[i] = 1 if i == winner else -1
        return self.payoffs

    def get_num_actions(self):
        """
            动作数目
        Returns:
            int: the total number of abstract actions of doudizhu
        """
        return self.Num_upgrade_card * 20 + self.Num_trade_card * self.trade_num + self.Num_token_card + 6 + 1 + 5

    def get_player_id(self):
        """
            获取玩家id
        Returns:
            int: current player's id
        """
        return self.round.current_player

    def get_num_players(self):
        """
            玩家数目
        Returns:
            int: the number of players in doudizhu
        """
        return self.num_players

    def is_over(self):
        """
            游戏是否结束
        Returns:
            Bool: True(over) / False(not over)
        """
        if self.winner_id is None:
            return False
        return True

    def extract_action_state_baseline(self, action, player):
        """
        提取历史动作信息
        """
        info_category = []
        info_action = [action]

        if action < self.Num_upgrade_card * 20:
            info_category.append(0)
        elif action < self.Num_upgrade_card * 20 + self.Num_trade_card * 2:
            info_category.append(1)
        elif action < self.Num_upgrade_card * 20 + self.Num_trade_card * 2 + self.Num_token_card:
            info_category.append(2)
        elif action < self.Num_upgrade_card * 20 + self.Num_trade_card * 2 + self.Num_token_card + 6:
            info_category.append(3)
        elif action < action < self.Num_upgrade_card * 20 + self.Num_trade_card * 2 + self.Num_token_card + 6 + 1:
            info_category.append(4)
        else:
            info_category.append(5)

        info = info_category + info_action  # 2

        return info


if __name__ == "__main__":
    Game = CenturyGame()
    import time
    import random

    start_t = time.time()
    j = 0
    for i in range(100):
        print(i)
        Game.init_game()
        while Game.is_over() is False:
            legal_actions = Game.get_legal_actions()
            action = random.sample(legal_actions, 1)[0]
            state, _ = Game.step(action)
            # print(state["deck_function_cards_token"])
            j += 1
        # print(Game.get_payoffs(), Game.round_over_player)
    end_t = time.time()


    print(j)
    print(end_t - start_t)
