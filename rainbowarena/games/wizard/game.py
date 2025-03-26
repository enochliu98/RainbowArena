import functools
from heapq import merge
import numpy as np

from rainbowarena.games.wizard import Player
from rainbowarena.games.wizard import Round
from rainbowarena.games.wizard import Judger
from rainbowarena.games.wizard.rule import PLAYER_NUM, MAX_ROUND, INVISIBLE_DUNS_PRD

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

class WizardGame:
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
        self.idx_1 = 4
        self.idx_2 = 60 // self.num_players + 1
        self.idx_3 = 60

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
                                                         self.round.dealer.ace_color,
                                                         self.round.dun_color)
            # 更新信息
            self.players[dun_winner_id].duns += 1
            self.round.dun_num += 1
            self.round.dun_cards = []
            self.round.dun_color = -1
            is_dun_over = True
            # (2) 判断当前轮是否结束 判断方法：已完成的墩数等于当前轮数加一
            if self.round.dun_num == self.round.round_num + 1:
                # 更新信息
                for player_tmp in self.players:
                    player_tmp.point += 20 + 10 * player_tmp.duns if player_tmp.duns == player_tmp.duns_prd \
                        else -10 * abs(player_tmp.duns - player_tmp.duns_prd)
                for player_tmp in self.players:
                    player_tmp.init_game()
                self.round.dun_num = 0
                self.round.round_num += 1
                self.round.round_player = (self.round.round_player + 1) % len(self.players)
                if self.round.round_num < min(60 // self.num_players, MAX_ROUND):  # 完成最大轮数就停止
                    self.round.dealer.init_game(self.players, self.round.round_num + 1)
                is_round_over = True
                # (3) 判断当前局是否结束 判断方法：已完成轮数等于总轮数
                if self.round.round_num == min(60 // self.num_players, MAX_ROUND):
                    self.winner_id = self.judger.judge_winner(self.players)
        # ⑤ 更新下一个玩家
        if action < self.idx_1:
            next_id = self.round.current_player
        elif action < self.idx_1 + self.idx_2:
            next_id = (self.round.current_player + 1) % len(self.players)
        else:
            if is_dun_over:
                next_id = dun_winner_id  # 当前墩结束，下一个玩家为当前墩的赢家
                if is_round_over:
                    next_id = self.round.round_player
            else:
                next_id = (self.round.current_player + 1) % len(self.players)
        self.round.current_player = next_id  # 下一个玩家
        # ⑥ 更新下一个状态
        state = self.get_state(next_id)  # 获取新状态
        self.state = state  # 获取新状态

        return state, next_id

    def get_state(self, player_id):
        """

        """
        state = {}

        # 个人信息
        state["self_cards"] = self.players[player_id].cards
        state["self_point"] = self.players[player_id].point
        state["self_duns"] = self.players[player_id].duns
        state["self_duns_prd"] = self.players[player_id].duns_prd
        # 其他玩家信息
        state["opponent_point"] = [self.players[(player_id + idx) % self.num_players].point
                                   for idx in range(1, self.num_players)]
        state["opponent_duns"] = [self.players[(player_id + idx) % self.num_players].duns
                                  for idx in range(1, self.num_players)]
        if INVISIBLE_DUNS_PRD:
            state["opponent_duns_prd"] = [-1 for _ in range(self.num_players - 1)]
        else:
            state["opponent_duns_prd"] = [self.players[(player_id + idx) % self.num_players].duns_prd
                                          for idx in range(1, self.num_players)]

        # 牌桌信息
        state["deck_ace_color"] = self.round.dealer.ace_color
        state["deck_dun_color"] = self.round.dun_color
        state["deck_dun_cards"] = self.round.dun_cards
        state["deck_dun_num"] = self.round.dun_num
        state["deck_round_num"] = self.round.round_num

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
            self.payoffs = transform_scores(points)

        return self.payoffs

    def get_num_actions(self):
        """
        动作数目
        """
        return self.idx_1 + self.idx_2 + self.idx_3

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
    Game = WizardGame()
    import time
    import random
    from tqdm import tqdm

    start_t = time.time()
    j = 0
    for i in tqdm(range(5000)):
        # print(i)
        Game.init_game()
        count = 0
        while Game.is_over() is False:
            legal_actions = Game.get_legal_actions()
            action = random.sample(legal_actions, 1)[0]
            Game.step(action)
            j += 1
            count += 1
    end_t = time.time()

    print(j)
    print(end_t - start_t)
