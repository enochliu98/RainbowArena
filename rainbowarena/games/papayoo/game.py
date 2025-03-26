import functools
from heapq import merge
import numpy as np

from rainbowarena.games.papayoo import Player
from rainbowarena.games.papayoo import Round
from rainbowarena.games.papayoo import Judger
from rainbowarena.games.papayoo.rule import PLAYER_NUM

def transform_scores(scores):
    n = len(scores)
    mean_score = np.mean(scores)
    std_dev = np.std(scores)

    if std_dev == 0:
        # All scores are the same
        transformed_scores = np.zeros(n)
    else:
        # Standardize scores
        z_scores = (scores - mean_score) / std_dev

        # Find the scaling factor
        k = 1 / np.max(np.abs(z_scores))

        # Scale the standardized scores
        transformed_scores = k * z_scores

    return transformed_scores


class PapayooGame:
    def __init__(self):
        self.history_action = None
        self.state = None
        self.judger = None
        self.round_over_player = None
        self.round = None
        self.players = None
        self.winner_id = None
        self.np_random = np.random.RandomState()
        self.num_players = PLAYER_NUM
        self.payoffs = [0 for _ in range(self.num_players)]

    def init_game(self):
        """
        ① 初始化公共信息
        ② 初始化玩家
        ③ 初始化回合
        ④ 初始化裁判
        ⑤ 初始化第一个玩家信息
        ⑥ 初始化历史动作信息
        """
        # 初始化公共信息
        self.winner_id = None  # 胜者
        self.payoffs = [0 for _ in range(self.num_players)]  # 收益
        # 初始化玩家
        self.players = [Player(player_id) for player_id in range(self.num_players)]
        # 初始化一个回合
        self.round = Round(self.np_random, self.num_players, self.players)
        # 初始化裁判
        self.judger = Judger()
        # 初始化第一个玩家信息
        player_id = self.round.current_player
        state = self.get_state(player_id)
        self.state = state
        # 初始化历史动作信息
        self.history_action = []

        return state, player_id

    def step(self, action):
        """
        ① 获取当前玩家
        ② 存储历史动作信息
        ③ 执行动作
        ④ 判断结束
        ⑤ 更新下一个玩家
        ⑥ 更新下一个状态
        """
        # ① 获取当前玩家
        player = self.players[self.round.current_player]
        # ② 存储历史动作信息
        self.history_action.append((self.round.current_player, action))
        # ③ 当前玩家执行对应的动作
        self.round.proceed_round(player, action)
        # ④ 判断结束
        is_dun_over = False
        is_round_over = False
        # （1）判断当前墩是否结束 判断方法：下的牌数等于玩家数
        if len(self.round.dun_cards) == self.num_players:
            # 判断胜者
            dun_winner_id = self.judger.judge_dun_winner(self.round.dun_cards,
                                                         self.round.dun_color)
            # 更新信息
            for dun_card in self.round.dun_cards:
                if dun_card[1] == self.round.dealer.papayoo_card:  # papayoo卡
                    self.players[dun_winner_id].point -= 40
                elif dun_card[1] >= 40:  # payo卡
                    self.players[dun_winner_id].point -= dun_card[1] - 39
            self.round.dun_num += 1
            self.round.dun_cards = []
            self.round.dun_color = -1
            is_dun_over = True
            # (2) 判断当前轮是否结束 判断方法：已完成的墩数等于60 // self.num_players
            if self.round.dun_num == 60 // self.num_players:
                # 更新信息
                for player_tmp in self.players:
                    player_tmp.init_game()
                self.round.dun_num = 0
                self.round.round_num += 1
                self.round.round_player = (self.round.round_player + 1) % self.num_players
                self.round.dealer.init_game(self.players)
                is_round_over = True
                # (3) 判断当前局是否结束 判断方法：已完成轮数等于总轮数
                if self.round.round_num == self.num_players:
                    self.winner_id = self.judger.judge_winner(self.players)
        # ⑤ 更新下一个玩家
        if is_round_over:
            next_id = self.round.round_player
        elif is_dun_over:
            next_id = dun_winner_id
        else:
            next_id = (self.round.current_player + 1) % len(self.players)
        self.round.current_player = next_id  # 下一个玩家
        # ⑥ 更新下一个状态
        state = self.get_state(next_id)  # 获取新状态
        self.state = state  # 获取新状态

        return state, next_id

    def get_state(self, player_id):
        """
        获取状态
        """
        state = {}

        # 个人信息
        state["self_cards"] = self.players[player_id].cards
        state["self_point"] = self.players[player_id].point
        # 其他玩家信息
        state["opponent_point"] = [self.players[(player_id + idx) % self.num_players].point
                                   for idx in range(1, self.num_players)]
        # 牌桌信息
        state["deck_dun_color"] = self.round.dun_color
        state["deck_dun_cards"] = self.round.dun_cards
        state["deck_dun_num"] = self.round.dun_num
        state["deck_round_num"] = self.round.round_num
        state["deck_papayoo_card"] = self.round.dealer.papayoo_card

        # 区分玩家的信息
        state["self_player"] = player_id
        # 历史动作信息
        state["history_action"] = self.history_action

        return state

    def get_legal_actions(self):
        """
        获取可用动作
        """
        player = self.players[self.round.current_player]  # 获取当前玩家
        return self.round.get_legal_actions(player)

    def get_payoffs(self):
        """
        获取收益
        """
        winner = self.winner_id
        if winner is not None:
            points = [player.point for player in self.players]
            self.payoffs = list(transform_scores(points))

        return self.payoffs

    def get_num_actions(self):
        """
        动作数目
        """
        return 60

    def get_player_id(self):
        """
        玩家id
        """
        return self.round.current_player

    def get_num_players(self):
        """
        玩家数目
        """
        return self.num_players

    def is_over(self):
        """
        是否结束
        """
        if self.winner_id is None:
            return False
        return True


if __name__ == "__main__":
    Game = PapayooGame()
    import time
    import random

    start_t = time.time()
    j = 0
    for i in range(10000):
        # print(i)
        Game.init_game()
        count = 0
        while Game.is_over() is False:
            legal_actions = Game.get_legal_actions()
            action = random.sample(legal_actions, 1)[0]
            Game.step(action)
            j += 1
            count += 1
        print(Game.get_payoffs())
        print(count)
    end_t = time.time()

    print(j)
    print(end_t - start_t)
