# -*- coding: utf-8 -*-
''' Implement Doudizhu Game class
'''
import functools
from heapq import merge
import numpy as np

from rainbowarena.games.ticket2ride_small import Player
from rainbowarena.games.ticket2ride_small import Round
from rainbowarena.games.ticket2ride_small import Judger


class Ticket2RideGame:
    """ Provide game APIs for env to run splendor and get corresponding state
    information.
    """

    def __init__(self, num_players):
        self.np_random = np.random.RandomState()
        self.num_players = num_players
        self.payoffs = [0 for _ in range(self.num_players)]

    def init_game(self):
        """ Initialize players and state.

        Returns:
            dict: first state in one game
            int: current player's id
        """
        # 初始化公共信息
        self.winner_id = None  # 胜者
        self.last_round_flag = False  # 最后一个回合的标志
        self.payoffs = [0 for _ in range(self.num_players)]

        # 初始化玩家
        self.players = [Player(num) for num in range(self.num_players)]

        # 初始化一个回合
        self.round = Round(self.np_random, self.num_players, self.players)
        self.round_over_player = (self.round.current_player + self.num_players - 1) % self.num_players

        # 初始化裁判
        self.judger = Judger()

        # 获取第一个玩家信息
        player_id = self.round.current_player
        state = self.get_state(player_id)
        self.state = state

        # 获取历史动作信息
        self.history_action = []

        return state, player_id

    def step(self, action):
        ''' Perform one draw of the game

        Args:
            action (str): specific action of doudizhu. Eg: '33344'

        Returns:
            dict: next player's state
            int: next player's id
        '''
        # perfrom action
        player = self.players[self.round.current_player]  # 获取当前玩家

        self.history_action.append((self.round.current_player, action))

        self.round.proceed_round(player, action)  # 当前玩家执行对应的动作

        if self.round.current_player == self.round_over_player:  # 当前回合结束

            if self.last_round_flag is True:
                winner_id = self.judger.judge_winner(self.players, self.round.current_player, self.round_over_player)

            if winner_id != -1:  # 不等于-1， 有胜者
                self.winner_id = winner_id

            for player in self.players:
                if player.train_cars <= 2:  # 当可用火车车厢数为0,1,2时，下一轮结束
                    self.last_round_flag = True
                    break

        next_id = (player.player_id + 1) % len(self.players)  # 下一个玩家
        self.round.current_player = next_id  # 下一个玩家

        tmp_legal_actions = self.get_legal_actions()  # 当无行动可执行时，终止
        if len(tmp_legal_actions) == 0:
            self.winner_id = self.num_players - 1 - next_id

        # get next state
        state = self.get_state(next_id)  # 获取新状态
        self.state = state  # 获取新状态

        return state, next_id

    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        state = {}

        # 个人信息
        state["self_train_cars"] = self.players[player_id].train_cars
        state["self_train_car_cards"] = self.players[player_id].train_car_cards
        state["self_ticket_cards"] = self.players[player_id].ticket_cards
        state["self_line_map_valid"] = self.players[player_id].line_map_valid
        state["self_longest_line"] = self.players[player_id].longest_line
        state["self_board_map_valid"] = self.players[player_id].board_map_valid
        state["self_point"] = self.players[player_id].point

        # 其他玩家信息
        state["other_train_cars"] = self.players[1 - player_id].train_cars
        state["other_line_map_valid"] = self.players[1 - player_id].line_map_valid
        state["other_longest_line"] = self.players[1 - player_id].longest_line
        state["other_board_map_valid"] = self.players[1 - player_id].board_map_valid
        state["other_point"] = self.players[1 - player_id].point

        # 牌桌信息
        state["deck_train_car_cards"] = self.round.dealer.deck[0]
        state["deck_ticket_cards"] = self.round.dealer.deck[1]
        state["lines"] = self.round.dealer.lines

        # 区分玩家的信息
        state["self_player"] = player_id

        # 历史动作信息
        state["history_action"] = self.history_action

        # 隐藏信息
        state["invisible_other_train_car_cards"] = self.players[1 - player_id].train_car_cards
        state["invisible_other_ticket_car_cards"] = self.players[1 - player_id].ticket_cards

        return state

    def get_legal_actions(self):
        player = self.players[self.round.current_player]  # 获取当前玩家
        return self.round.get_legal_actions(player)

    def get_payoffs(self):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        winner = self.winner_id
        if winner is not None:
            for i in range(self.num_players):
                self.payoffs[i] = 1 if i == winner else -1
        return self.payoffs

    def get_num_actions(self):
        ''' Return the total number of abstract acitons

        Returns:
            int: the total number of abstract actions of doudizhu
        '''
        return 54

    def get_player_id(self):
        ''' Return current player's id

        Returns:
            int: current player's id
        '''
        return self.round.current_player

    def get_num_players(self):
        ''' Return the number of players in doudizhu

        Returns:
            int: the number of players in doudizhu
        '''
        return self.num_players

    def is_over(self):
        ''' Judge whether a game is over

        Returns:
            Bool: True(over) / False(not over)
        '''
        if self.winner_id is None:
            return False
        return True
