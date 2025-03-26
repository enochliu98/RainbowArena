import numpy as np
from collections import OrderedDict

from rainbowarena.envs import Env
from rainbowarena.games.ticket2ride import Game
import torch
import copy


class Ticket2RideEnv(Env):

    def __init__(self, config):
        self.name = 'ticket2ride_psro'
        self.game = Game(self.num_players)
        super().__init__(config)

        # 玩家信息
        # 车厢数目
        self.state_train_cars_player_len = 10
        # 车厢卡
        self.state_train_car_cards_player_len = 9
        # 车票卡
        self.state_ticket_cards_player_len = 3 * 30
        # 线路
        self.state_line_map_valid_player_len = 100 * 8 
        self.state_longest_line_player_len = 10
        self.state_board_map_valid_player_len = 36 * 36
        # 分数
        self.state_point_player_len = 10

        self.state_player_len = self.state_train_cars_player_len * 2\
            + self.state_train_car_cards_player_len \
            + self.state_ticket_cards_player_len \
            + self.state_line_map_valid_player_len \
            + self.state_longest_line_player_len * 2 \
            + self.state_board_map_valid_player_len \
            + self.state_point_player_len * 2

        # 桌面信息
        self.state_deck_len = 5

        # 玩家数目信息（self.num_players，做一下embedding）
        self.state_unique_len = 5

        # 历史行动

        # 上述相加
        self.state_len = self.state_player_len + self.state_deck_len + self.state_unique_len

        self.state_shape = [[self.state_len] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, state):
        obs = np.zeros((self.state_len), dtype=int)

        idxs = [
            self.state_player_len,
            self.state_player_len + self.state_deck_len,
            self.state_player_len + self.state_deck_len + self.state_unique_len,
        ]

        obs[: idxs[0]] = extract_player_state(state, self.num_players)
        obs[idxs[0] : idxs[1]] = extract_deck_state(state)
        obs[idxs[1] : idxs[2]] = extract_unique_state(state, self.game.num_players)

        legal_action_id = self._get_legal_actions()
        extracted_state = {'obs': obs, 'legal_actions': legal_action_id}
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = self.game.get_legal_actions()
        extracted_state['action_record'] = self.action_recorder  
        return extracted_state

    def get_payoffs(self):
        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        return action_id # 统一的动作编码

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()  # 获取可用动作
        legal_ids = {action: None for action in legal_actions}  # 转化成dict
        return OrderedDict(legal_ids)



def extract_player_state(state):
    player_state = []

    # 玩家信息
    player_state.extend(extract_train_cars_player(state["self_train_cars"]))
    player_state.extend(extract_train_cars_player(state["other_train_cars"]))
    player_state.extend(state["self_train_car_cards"])
    player_state.extend(extract_ticket_cards_player(state["self_ticket_cards"]))
    player_state.extend(extract_line_map_valid_player(state["self_line_map_valid"], state["other_line_map_valid"], state["lines"]))
    player_state.extend(extract_longest_line_player(state["self_longest_line"]))
    player_state.extend(extract_longest_line_player(state["other_longest_line"]))
    player_state.extend(extract_board_map_valid_player(state["self_board_map_valid"], state["other_board_map_valid"]))
    player_state.extend(extract_point_player(state["self_point"]))
    player_state.extend(extract_point_player(state["other_point"]))

    return np.array(player_state)

def extract_deck_state(state):
    deck_state = []
    
    deck_state.extend(state["deck_train_car_cards"])

    return np.array(deck_state)

def extract_unique_state(state, num_players):
    '''
    embedding编码
    '''
    embedding = torch.nn.Embedding(num_players, 5)
    player = torch.tensor([state["self_player"]])
    embedding_output = embedding(player)[0].detach().numpy()

    return embedding_output

def extract_train_cars_player(train_cars):
    embedding = torch.nn.Embedding(46, 10)
    idx = torch.tensor([train_cars])
    embedding_output = embedding(idx)[0].detach().numpy()

    return embedding_output

def extract_ticket_cards_player(ticket_cards):
    state_ticket_cards = []
    for i in range(30):
        if i < len(ticket_cards):
            state_ticket_cards.extend([ticket_cards[i].start_pos, ticket_cards[i].end_pos, ticket_cards[i].point])
        else:
            state_ticket_cards.extend([0, 0, 0])    

def extract_line_map_valid_player(line_map_valid_self, line_map_valid_other, lines):
    pass
    
def extract_longest_line_player(longest_line):
    embedding = torch.nn.Embedding(7, 10)
    idx = torch.tensor([longest_line])
    embedding_output = embedding(idx)[0].detach().numpy()

    return embedding_output

def extract_board_map_valid_player(board_map_valid_self, board_map_valid_other):
    pass

def extract_point_player(point):
    embedding = torch.nn.Embedding(101, 10)
    idx = torch.tensor([point])
    embedding_output = embedding(idx)[0].detach().numpy()

    return embedding_output