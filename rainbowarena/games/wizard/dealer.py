from rainbowarena.games.wizard.utils import init_cards

import numpy as np


class WizardDealer:

    def __init__(self, players, np_random=np.random.RandomState()):
        """
        整局初始化
        ① 初始化库
        ② 发牌
        """
        self.np_random = np_random  # 随机库
        self.cards = init_cards()  # 初始化牌库里的牌
        self.shuffle()  # 随机洗牌

        for player in players:
            for _ in range(1):  # 当前为第几轮，则发几张牌
                card = self.cards.pop()
                player.cards[card] += 1  # 加入到玩家手牌

        if len(self.cards) >= 0:
            self.deck = self.cards.pop()  # 翻开一张牌
            if self.deck < 52:  # 正常
                self.ace_color = self.deck % 4
            elif self.deck < 56:  # 巫师
                self.ace_color = -1
            else:  # 小丑
                self.ace_color = 4
        else:
            self.deck = None
            self.ace_color = 4

    def init_game(self, players, round_idx):
        """
        一整局中小局初始化，每次都需要重新洗牌发牌
        """
        self.cards = init_cards()  # 初始化牌库里的牌
        self.shuffle()  # 随机洗牌

        for player in players:
            for _ in range(round_idx):  # 当前为第几轮，则发几张牌
                card = self.cards.pop()
                player.cards[card] += 1  # 加入到玩家手牌

        if len(self.cards) > 0:
            self.deck = self.cards.pop()  # 翻开一张牌
            if self.deck < 52:  # 正常
                self.ace_color = self.deck % 4
            elif self.deck < 56:  # 巫师
                self.ace_color = -1
            else:  # 小丑
                self.ace_color = 4
        else:
            self.deck = None
            self.ace_color = 4

    def shuffle(self):
        """
            随机洗牌
        """
        self.np_random.shuffle(self.cards)
