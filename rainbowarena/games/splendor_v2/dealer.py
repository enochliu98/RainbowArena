from rainbowarena.games.splendor_v2.utils import init_cards, init_tokens
from rainbowarena.games.splendor_v2.card import SplendorCard
import numpy as np

class SplendorDealer:
    ''' Initialize a uno dealer class
    '''
    def __init__(self, np_random=np.random.RandomState(), num_players=2):
        self.np_random = np_random  # 随机库
        self.cards = init_cards()  # 初始化牌 
        self.tokens = init_tokens(num_players)  # 初始化宝石
        self.shuffle()  # 随机洗牌
        self.deck = self.init_deck(num_players)  # 初始化桌面上的牌

    def shuffle(self):
        ''' 
            随机洗牌
        '''
        self.np_random.shuffle(self.cards[0])
        self.np_random.shuffle(self.cards[1])
        self.np_random.shuffle(self.cards[2])
        self.np_random.shuffle(self.cards[3])

    def init_deck(self, num_players):
        """
            初始化桌面上的牌
        """
        deck = [[], [], [], []]

        for i in range(num_players+1):
            deck[0].append(self.cards[0].pop())

        for i in range(4):
            deck[1].append(self.cards[1].pop())
            deck[2].append(self.cards[2].pop())
            deck[3].append(self.cards[3].pop())

        return deck
    
    def deal_cards(self, level, position):
        """
            补齐空缺的牌
        """
        if len(self.cards[level]) > 0:  # 有牌
            self.deck[level][position] = self.cards[level].pop()
        else:
            default_card = SplendorCard()
            self.deck[level][position] = default_card 
    