import numpy as np

from rainbowarena.games.splendor import Player
from rainbowarena.games.splendor import Round
from rainbowarena.games.splendor import Judger

FETCH_DIFF_TOKENS = [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3],
                     [0, 3, 4], [0, 2, 4], [0, 2, 3],
                     [0, 1, 4], [0, 1, 3],
                     [0, 1, 2]]

class SplendorGame:
    ''' Provide game APIs for env to run splendor and get corresponding state
    information.
    '''
    def __init__(self, num_players=2):
        self.np_random = np.random.RandomState()
        self.num_players = num_players
        self.payoffs = [0 for _ in range(self.num_players)]

    def init_game(self):
        ''' Initialize players and state.

        Returns:
            dict: first state in one game
            int: current player's id
        '''
        # 初始化公共信息
        self.winner_id = None  # 胜者
        self.payoffs = [0 for _ in range(self.num_players)]

        # 初始化玩家
        self.players = [Player(num) for num in range(self.num_players)]

        # 初始化一个回合
        self.round = Round(self.np_random, self.num_players)
        self.round_over_player = (self.round.current_player + self.num_players - 1) % self.num_players

        # 初始化裁判
        self.judger = Judger()

        # 获取历史动作信息
        self.history_action = []
        self.history_action_baseline = []

        # 获取第一个玩家信息
        player_id = self.round.current_player
        state = self.get_state(player_id)
        self.state = state

        return state, player_id

    def step(self, action, ui=False):
        ''' Perform one draw of the game

        Args:
            action (str): specific action of doudizhu. Eg: '33344'

        Returns:
            dict: next player's state
            int: next player's id
        '''
        # perfrom action
        player = self.players[self.round.current_player]  # 获取当前玩家

        if ui: # 绘制对应的界面
            # 牌桌上的牌
            ui_deck = self.round.dealer.deck

            print("************* DECK INFO **************")
            print("------------ LEVEL:NOBLE -------------")
            for i in range(len(ui_deck[0])):
                print(ui_deck[0][i].cost, ui_deck[0][i].point, ui_deck[0][i].bonus)
            print("--------------------------------------")
            print("------------ LEVEL: 3    -------------")
            for i in range(4):
                print(ui_deck[3][i].cost, ui_deck[3][i].point, ui_deck[3][i].bonus)
            print("--------------------------------------")
            print("------------ LEVEL: 2    -------------")
            for i in range(4):
                print(ui_deck[2][i].cost, ui_deck[2][i].point, ui_deck[2][i].bonus)
            print("--------------------------------------")
            print("------------ LEVEL: 1    -------------")
            for i in range(4):
                print(ui_deck[1][i].cost, ui_deck[1][i].point, ui_deck[1][i].bonus)
            print("--------------------------------------")
            print()

            ui_player = self.players[self.round.current_player]  # 当前玩家
            print(self.round.current_player)
            print("*********** PLAYER:CURRENT ***********")
            print("------------   TOKENS    -------------")
            print(ui_player.current_tokens, ui_player.current_card_tokens)
            print("--------------------------------------")
            print("------------    CARDS    -------------")
            for card in ui_player.current_cards:
                print(card.cost, card.point, card.bonus)
            print("--------------------------------------")
            print("----------   FLIP CARDS    -----------")
            for flip_card in ui_player.current_flip_cards:
                print(flip_card.cost, flip_card.point, flip_card.bonus)
            print("--------------------------------------")
            print("----------   NOBLE CARDS    ----------")
            for noble_card in ui_player.current_noble_cards:
                print(noble_card.cost, noble_card.point, noble_card.bonus)
            print("--------------------------------------")
            print("------------   POINT    -------------")
            print(ui_player.point)
            print("--------------------------------------")
            print("------------   ACTION    -------------")
            print(action)
            print("--------------------------------------")
            print()

            ui_player = self.players[1 - self.round.current_player]  # 其他玩家

            print("*********** PLAYER:OTHER *************")
            print("------------   TOKENS    -------------")
            print(ui_player.current_tokens, ui_player.current_card_tokens)
            print("--------------------------------------")
            print("------------    CARDS    -------------")
            for card in ui_player.current_cards:
                print(card.cost, card.point, card.bonus)
            print("--------------------------------------")
            print("----------   FLIP CARDS    -----------")
            for flip_card in ui_player.current_flip_cards:
                print(flip_card.cost, flip_card.point, flip_card.bonus)
            print("--------------------------------------")
            print("----------   NOBLE CARDS    ----------")
            for noble_card in ui_player.current_noble_cards:
                print(noble_card.cost, noble_card.point, noble_card.bonus)
            print("--------------------------------------")
            print("------------   POINT    -------------")
            print(ui_player.point)
            print("--------------------------------------")
            print()

        self.history_action.append((self.round.current_player, self.extract_action_state(action, player)))  # 存储历史动作信息
        self.history_action_baseline.append((self.round.current_player, self.extract_action_state_baseline(action, player)))

        self.round.proceed_round(player, action)  # 当前玩家执行对应的动作

        if self.round.current_player == self.round_over_player: # 当前回合结束
            winner_id = self.judger.judge_winner(self.players, self.round_over_player)
            if winner_id != -1: # 不等于-1， 有胜者
                self.winner_id = winner_id

        next_id = (player.player_id+1) % len(self.players)  # 下一个玩家
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
        state["self_tokens"] = self.players[player_id].current_tokens
        state["self_card_tokens"] = self.players[player_id].current_card_tokens
        state["self_cards"] = self.players[player_id].current_cards
        state["self_flip_cards"] = self.players[player_id].current_flip_cards
        state["self_noble_cards"] = self.players[player_id].current_noble_cards
        state["self_point"] = self.players[player_id].point
        # 其他玩家信息
        state["other_tokens"] = self.players[1 - player_id].current_tokens
        state["other_card_tokens"] = self.players[1 - player_id].current_card_tokens
        state["other_cards"] = self.players[1 - player_id].current_cards
        state["other_flip_cards"] = self.players[1 - player_id].current_flip_cards
        state["other_noble_cards"] = self.players[1 - player_id].current_noble_cards
        state["other_point"] = self.players[1 - player_id].point
        # 牌桌信息
        state["deck_tokens"] = self.round.dealer.tokens
        state["deck_cards"] = self.round.dealer.deck
        # 区分玩家的信息
        state["self_player"] = player_id
        # 历史动作信息
        state["history_action"] = self.history_action
        state["history_action_baseline"] = self.history_action_baseline
        # 隐藏信息
        state["invisible_info"] = self.round.dealer.cards  # 牌是看不到的

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

    @staticmethod
    def get_num_actions():
        ''' Return the total number of abstract acitons

        Returns:
            int: the total number of abstract actions of doudizhu
        '''
        return 42

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

    def extract_action_state(self, action, player):
        info_card = []
        info_token = [0] * 6
        info_category = []
        info_action = [action]

        if action < 5:
            info_token[action] += 2
            info_card.extend([-1, -1])
            info_category.append(0)
        elif action < 15:
            info_token[FETCH_DIFF_TOKENS[action - 5][0]] += 1
            info_token[FETCH_DIFF_TOKENS[action - 5][1]] += 1
            info_token[FETCH_DIFF_TOKENS[action - 5][2]] += 1
            info_card.extend([-1, -1])
            info_category.append(1)
        elif action < 27:
            level = (action - 15) // 4 + 1
            pos = (action - 15) % 4
            card = self.round.dealer.deck[level][pos]  # 获取牌桌上的牌
            for i in range(5):
                info_token[i] = -card.cost[i]
            info_card.append(card.point)
            info_card.append(card.bonus)
            info_category.append(2)
        elif action < 30:
            pos = action - 27
            card = player.current_flip_cards[pos]  # 获取玩家盖着的牌
            for i in range(5):
                info_token[i] = -card.cost[i]
            info_card.append(card.point)
            info_card.append(card.bonus)
            info_category.append(3)
        elif action < 42:
            info_token[-1] += 1
            level = (action - 30) // 4 +1
            pos = (action - 30) % 4
            card = self.round.dealer.deck[level][pos]  # 获取牌桌上的牌        
            for i in range(5):
                info_token[i] = -card.cost[i]
            info_card.append(card.point)
            info_card.append(card.bonus)    
            info_category.append(4)

        info = info_card + info_token + info_category + info_action  # 2 + 6 + 1 + 1 = 10
        
        return info
    
    def extract_action_state_baseline(self, action, player):
        info_category = []
        info_action = [action]

        if action < 5:
            info_category.append(0)
        elif action < 15:
            info_category.append(1)
        elif action < 27:
            info_category.append(2)
        elif action < 30:
            info_category.append(3)
        elif action < 42: 
            info_category.append(4)

        info = info_category + info_action  # 2
        
        return info

if __name__ == "__main__":
    Game = SplendorGame()
    import time
    import random
    start_t = time.time()
    j = 0
    for i in range(1000):
        print(i)
        Game.init_game()
        while Game.is_over() is False:
            legal_actions = Game.get_legal_actions()
            action = random.sample(legal_actions, 1)[0]
            Game.step(action)
            j += 1
    end_t = time.time()
    print(j)
    print(end_t - start_t)