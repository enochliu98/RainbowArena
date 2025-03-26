import functools
from heapq import merge
import numpy as np

from rainbowarena.games.lama import Player
from rainbowarena.games.lama import Round
from rainbowarena.games.lama import Judger

CARD2POINT = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 10
}

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


class LamaGame:
    def __init__(self, num_players=3):
        self.history_action = None
        self.state = None
        self.judger = None
        self.round_over_player = None
        self.round = None
        self.players = None
        self.winner_id = None
        self.np_random = np.random.RandomState()
        self.num_players = num_players
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
        # 获取第一个玩家信息
        player_id = self.round.current_player
        state = self.get_state(player_id)
        self.state = state
        # 获取历史动作信息
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
        is_over = self.judger.judge_round_over(self.players)  # 判断当前小局是否结束
        if is_over is True:
            # (1) 计算分数
            for player_tmp in self.players:
                total_point = 0
                for card, card_num in enumerate(player_tmp.cards):
                    if card_num > 0:
                        total_point += CARD2POINT[card]
                if sum(player_tmp.cards) == 0:
                    player_tmp.point -= 10 if player_tmp.point >= 10 else 1
                player_tmp.point += total_point
            # (2) 判断整局胜负
            winner_id = self.judger.judge_winner(self.players)
            if winner_id != -1:
                self.winner_id = winner_id
            # (3) 更新玩家
            for player_tmp in self.players:
                player_tmp.init_game()
            # (4) 更新dealer
            self.round.dealer.init_game(self.players)
        # ⑤ 更新下一个玩家
        if is_over is True:
            next_id = self.round.current_player  # 如果新的一小局，由上一个执行最后一步的玩家开始执行
        else:
            next_id = (player.player_id + 1) % len(self.players)
            while next_id != player.player_id:
                if self.players[next_id].quit == 0:
                    break
                next_id = (next_id + 1) % len(self.players)
        self.round.current_player = next_id  # 下一个玩家
        # ⑥ 更新下一个状态
        state = self.get_state(next_id)  # 获取新状态
        self.state = state  # 获取新状态

        return state, next_id

    def get_state(self, player_id):
        """

        """
        state = {}

        opponent_id_1 = (player_id + 1) % self.num_players
        opponent_id_2 = (player_id + 2) % self.num_players
        opponent_ids = [opponent_id for opponent_id in range(self.num_players) if opponent_id != player_id]

        # 个人信息
        state["self_cards"] = self.players[player_id].cards
        state["self_point"] = self.players[player_id].point
        state["self_quit"] = self.players[player_id].quit
        # 其他玩家信息
        state["opponent_1_point"] = self.players[opponent_id_1].point
        state["opponent_2_point"] = self.players[opponent_id_2].point
        state["opponent_1_quit"] = self.players[opponent_id_1].quit
        state["opponent_2_quit"] = self.players[opponent_id_2].quit
        # state['opponent_point'] = [self.players[opponent_id].point for opponent_id in opponent_ids]
        # state['opponent_quit'] = [self.players[opponent_id].quit for opponent_id in opponent_ids]
        # 牌桌信息
        state["deck_card"] = self.round.dealer.deck
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
        return self.round.get_legal_actions(player, self.players)

    def get_payoffs(self):
        """
        获取收益
        """
        winner = self.winner_id
        if winner is not None:
            points = [-player.point for player in self.players]
            self.payoffs = list(transform_scores(points))

        return self.payoffs

    def get_num_actions(self):
        """
        动作数目
        """
        return 9

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
    Game = LamaGame()
    import time
    import random

    start_t = time.time()
    j = 0
    for i in range(10000):
        print(i)
        Game.init_game()
        while Game.is_over() is False:
            legal_actions = Game.get_legal_actions()
            action = random.sample(legal_actions, 1)[0]
            Game.step(action)
            j += 1
        print(Game.get_payoffs())
    end_t = time.time()

    print(j)
    print(end_t - start_t)
