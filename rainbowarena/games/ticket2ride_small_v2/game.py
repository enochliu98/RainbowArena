# -*- coding: utf-8 -*-

import functools
import random
from heapq import merge
import numpy as np

from rainbowarena.games.ticket2ride_small_v2 import Player
from rainbowarena.games.ticket2ride_small_v2 import Round
from rainbowarena.games.ticket2ride_small_v2 import Judger


class Ticket2RideGame:

    def __init__(self, num_players=2):
        self.history_action = None
        self.state = None
        self.judger = None
        self.round_over_player = None
        self.round = None
        self.players = None
        self.last_round_flag = None
        self.winner_id = None
        self.np_random = np.random.RandomState()
        self.num_players = num_players
        self.payoffs = [0 for _ in range(self.num_players)]

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

        # 【初始化玩家】
        self.players = [Player(num) for num in range(self.num_players)]

        # 【初始化回合】
        self.round = Round(self.players, self.np_random, self.num_players)
        self.round_over_player = (self.round.current_player + self.num_players - 1) % self.num_players

        # 【初始化裁判】
        self.judger = Judger(self.np_random, self.num_players)

        # 【第一个玩家的信息】
        player_id = self.round.current_player
        state = self.get_state(player_id)
        self.state = state

        # 【其他信息】
        self.winner_id = None  # 胜者
        self.last_round_flag = False  # 最后一个回合的标志
        self.payoffs = [0 for _ in range(self.num_players)]
        self.history_action = []

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

        # 【获取当前玩家】
        player = self.players[self.round.current_player]

        # 【记录历史动作信息】
        self.history_action.append((self.round.current_player, action))

        # 【执行动作】
        self.round.proceed_round(player, action)

        if player.action_flag != -1:  # 两步操作的动作还没执行完
            next_id = player.player_id
        else:
            next_id = (player.player_id + 1) % len(self.players)  # 正常情况下下一位

        # 【判断回合结束】
        if next_id != player.player_id and self.round.current_player == self.round_over_player:
            if self.last_round_flag is True:  # 上回合中，是否满足有玩家剩下牌的数目为小于等于2,一定出胜负
                self.winner_id = self.judger.judge_winner(self.players)
            for player in self.players:
                if player.train_cars <= 2:  # 当可用火车车厢数为0,1,2时，下一轮结束
                    self.last_round_flag = True
                    break

        # 【下一个玩家信息】
        self.round.current_player = next_id
        state = self.get_state(next_id)
        self.state = state

        # 【判断下一个玩家是否有可行动作】
        legal_actions = self.get_legal_actions()
        if len(legal_actions) == 0:
            self.winner_id = self.num_players - 1 - next_id

        return state, next_id

    def get_state(self, player_id):
        """
            获取玩家状态

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        """
        state = {"self_train_cars": self.players[player_id].train_cars,
                 "self_player": player_id,
                 "self_train_car_cards": self.players[player_id].train_car_cards,
                 "self_ticket_cards": self.players[player_id].ticket_cards,
                 "self_line_map_valid": self.players[player_id].line_map_valid,
                 "self_board_map_valid": self.players[player_id].board_map_valid,
                 "self_point": self.players[player_id].point,
                 "self_action_flag": self.players[player_id].action_flag,
                 "other_train_cars": self.players[1 - player_id].train_cars,
                 "other_line_map_valid": self.players[1 - player_id].line_map_valid,
                 "other_board_map_valid": self.players[1 - player_id].board_map_valid,
                 "other_point": self.players[1 - player_id].point,
                 "deck_train_car_cards": self.round.dealer.deck[0],
                 "lines": self.round.dealer.lines,
                 "history_action": self.history_action,
                 "invisible_other_train_car_cards": self.players[1 - player_id].train_car_cards,
                 "invisible_other_ticket_car_cards": self.players[1 - player_id].ticket_cards,
                 "invisible_deck_ticket_cards": self.round.dealer.deck[1]
        }

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
        if self.winner_id is not None:
            for i in range(self.num_players):
                self.payoffs[i] = 1 if i == self.winner_id else -1
        return self.payoffs

    def get_num_actions(self):
        """
            动作数目
        Returns:
            int: the total number of abstract actions of doudizhu
        """
        return 42

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


if __name__ == "__main__":

    import time

    start_t = time.time()
    j = 0
    payoff = 0
    m_start = 0
    for i in range(10000):
        print(i)
        Game = Ticket2RideGame()
        _, player_id = Game.init_game()
        if player_id == 0:
            m_start += 1
        else:
            m_start -= 1
        while Game.is_over() is False:
            legal_actions = Game.get_legal_actions()
            action = random.sample(legal_actions, 1)[0]
            Game.step(action)
            j += 1
        payoff += Game.get_payoffs()[0]
    end_t = time.time()
    print(j)
    print(end_t - start_t)
    print(payoff)
    print(m_start)
