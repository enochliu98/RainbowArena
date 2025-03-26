from rainbowarena.games.lama.utils import init_cards

import numpy as np


class LamaDealer:

    def __init__(self, players, np_random=np.random.RandomState()):
        """
        整局初始化
        """
        self.np_random = np_random  # 随机库，确保随机种子一致

        self.cards = init_cards()  # 初始化牌库里的牌
        self.shuffle()  # 随机洗牌

        for player in players:
            for _ in range(6):  # 每个玩家发6张牌
                card = self.cards.pop()
                player.cards[card] += 1  # 加入到玩家手牌
        self.deck = self.cards.pop()  # 翻开一张牌

    def init_game(self, players):
        """
        一整局中小局初始化，每次都需要重新洗牌发牌
        """
        self.cards = init_cards()  # 初始化牌库里的牌
        self.shuffle()  # 随机洗牌

        for player in players:
            for _ in range(6):  # 每个玩家发6张牌
                card = self.cards.pop()
                player.cards[card] += 1  # 加入到玩家手牌
        self.deck = self.cards.pop()  # 翻开一张牌

    def shuffle(self):
        """
            随机洗牌
        """
        self.np_random.shuffle(self.cards)
