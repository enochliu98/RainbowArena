import numpy as np
from collections import OrderedDict

from rainbowarena.envs import Env
from rainbowarena.games.ticket2ride_small import Game


class Ticket2RideEnv(Env):

    def __init__(self, config):
        self.name = 'ticket2ride_small'
        self.game = Game(self.num_players)
        super().__init__(config)

        # 玩家信息
        # 车厢数目
        self.state_train_cars_player_len = 1
        # 车厢卡
        self.state_train_car_cards_player_len = 5
        # 车票卡
        self.state_ticket_cards_player_len = 3 * 8
        # 线路
        self.state_line_map_valid_player_len = 16 * 5
        self.state_board_map_valid_player_len = 8 * 8
        # 分数
        self.state_point_player_len = 1

        self.state_player_len = self.state_train_cars_player_len * 2 \
                                + self.state_train_car_cards_player_len \
                                + self.state_ticket_cards_player_len \
                                + self.state_line_map_valid_player_len \
                                + self.state_board_map_valid_player_len \
                                + self.state_point_player_len * 2

        # 桌面信息
        self.state_deck_train_car_cards_len = 5
        self.state_deck_ticket_cards_len = 3 * 3

        self.state_deck_len = self.state_deck_ticket_cards_len + self.state_deck_train_car_cards_len

        # 玩家数目信息（self.num_players，做一下embedding）
        self.state_unique_len = self.num_players

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
        obs[idxs[0]: idxs[1]] = extract_deck_state(state)
        obs[idxs[1]: idxs[2]] = extract_unique_state(state, self.game.num_players)

        legal_action_id = self._get_legal_actions()
        extracted_state = {'obs': obs, 'legal_actions': legal_action_id, 'raw_obs': state,
                           'raw_legal_actions': self.game.get_legal_actions(), 'action_record': self.action_recorder}
        return extracted_state

    def get_payoffs(self):
        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        return action_id  # 统一的动作编码

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()  # 获取可用动作
        legal_ids = {action: None for action in legal_actions}  # 转化成dict
        return OrderedDict(legal_ids)


def extract_player_state(state):
    player_state = []

    # 玩家信息
    player_state.append(state["self_train_cars"])
    player_state.append(state["other_train_cars"])
    player_state.extend(state["self_train_car_cards"])
    player_state.extend(extract_ticket_cards_player(state["self_ticket_cards"]))
    player_state.extend(
        extract_line_map_valid_player(state["self_line_map_valid"], state["other_line_map_valid"], state["lines"]))
    player_state.extend(
        extract_board_map_valid_player(state["self_board_map_valid"], state["other_board_map_valid"], state["lines"]))
    player_state.append(state["self_point"])
    player_state.append(state["other_point"])

    return np.array(player_state)


def extract_deck_state(state):
    deck_state = []

    state_train_car_cards = state["deck_train_car_cards"]
    state_ticket_cards = []
    ticket_cards_count = 0
    for ticket_card in state["deck_ticket_cards"]:
        state_ticket_cards.extend(ticket_card)
        ticket_cards_count += 1
    while ticket_cards_count < 3:
        state_ticket_cards.extend([0, 0, 0])
        ticket_cards_count += 1

    deck_state.extend(state_train_car_cards)
    deck_state.extend(state_ticket_cards)

    return np.array(deck_state)


def extract_unique_state(state, num_players):
    """
    embedding编码
    """
    state_unique = [0] * num_players
    state_unique[state["self_player"]] = 1

    return np.array(state_unique)


def extract_ticket_cards_player(ticket_cards):
    state_ticket_cards = []
    for i in range(8):
        if i < len(ticket_cards):
            state_ticket_cards.extend([ticket_cards[i].start_pos, ticket_cards[i].end_pos, ticket_cards[i].point])
        else:
            state_ticket_cards.extend([0, 0, 0])

    return state_ticket_cards


def extract_line_map_valid_player(line_map_valid_self, line_map_valid_other, lines):
    state_line = []

    for i in range(len(lines)):
        if line_map_valid_self == 1:
            state_line.append([1] + lines[i][1:5])
        elif line_map_valid_other == 1:
            state_line.append([2] + lines[i][1:5])
        else:
            state_line.append([0] + lines[i][1:5])

    state_line = np.array(state_line).flatten()

    return state_line


def extract_board_map_valid_player(board_map_valid_self, board_map_valid_other, lines):
    board_map_valid = [[-1] * 8] * 8

    for line in lines:
        board_map_valid[line[3]][line[4]] = 0
        board_map_valid[line[4]][line[3]] = 0

    for i in range(8):
        for j in range(8):
            if board_map_valid_self[i][j] == 1:
                board_map_valid[i][j] = 1
                board_map_valid[j][i] = 1
            elif board_map_valid_other[i][j] == 1:
                board_map_valid[i][j] = 2
                board_map_valid[j][i] = 2

    board_map_valid = np.array(board_map_valid).flatten()

    return board_map_valid
