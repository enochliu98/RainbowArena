import numpy as np
from collections import OrderedDict

from rainbowarena.envs import Env
from rainbowarena.games.ticket2ride_small_v2 import Game


############################################
# version v0
# 包含牌的具体信息
############################################

# class Ticket2RideEnv(Env):
#
#     def __init__(self, config):
#         self.name = 'ticket2ride_small_v2'
#         self.game = Game(2)
#         super().__init__(config)
#
#         """
#         玩家信息 （2+5+24+80+64+2+1=178）
#         ① 车厢数目信息 （1）
#         ② 车厢卡信息 （5）
#         ③ 车票卡信息 （3*8=24）
#         ④ 线路映射信息 （16*5=80）
#         ⑤ 线路地图信息 （8*8=64）
#         ⑥ 分数信息 （1）
#         ⑦ 区分玩家信息 （2）
#         """
#         self.state_train_cars_player_len = 1
#         self.state_train_car_cards_player_len = 5
#         self.state_ticket_cards_player_len = 3 * 8
#         self.state_line_map_valid_player_len = 16 * 5
#         self.state_board_map_valid_player_len = 8 * 8
#         self.state_point_player_len = 1
#         self.state_unique_len = self.num_players
#         self.state_player_len = self.state_train_cars_player_len * 2 \
#                                 + self.state_train_car_cards_player_len \
#                                 + self.state_ticket_cards_player_len \
#                                 + self.state_line_map_valid_player_len \
#                                 + self.state_board_map_valid_player_len \
#                                 + self.state_point_player_len * 2 \
#                                 + self.state_unique_len
#
#         """
#         桌面信息 （5+9=14）
#         ① 车厢卡信息 （5）
#         ② 车票卡信息 （3*3=9）
#         """
#         self.state_deck_train_car_cards_len = 5
#         self.state_deck_ticket_cards_len = 3 * 3
#         self.state_deck_len = self.state_deck_ticket_cards_len + self.state_deck_train_car_cards_len
#
#         self.state_len = self.state_player_len + self.state_deck_len
#
#         self.state_shape = [[self.state_len] for _ in range(self.num_players)]
#         self.action_shape = [None for _ in range(self.num_players)]
#
#     def _extract_state(self, state):
#         """
#         状态特征提取
#         """
#
#         # 【状态特征初始化】
#         obs = np.zeros(self.state_len, dtype=int)
#
#         # 【状态特征填充】
#         idxs = [
#             self.state_player_len,
#             self.state_player_len + self.state_deck_len
#         ]
#         obs[: idxs[0]] = extract_player_state(state)
#         obs[idxs[0]: idxs[1]] = extract_deck_state(state)
#
#         legal_action_id = self._get_legal_actions()
#         extracted_state = {'obs': obs, 'legal_actions': legal_action_id, 'raw_obs': state,
#                            'raw_legal_actions': self.game.get_legal_actions(), 'action_record': self.action_recorder}
#         return extracted_state
#
#     def get_payoffs(self):
#         return np.array(self.game.get_payoffs())
#
#     def _decode_action(self, action_id):
#         return action_id  # 统一的动作编码
#
#     def _get_legal_actions(self):
#         legal_actions = self.game.get_legal_actions()  # 获取可用动作
#         legal_ids = {action: None for action in legal_actions}  # 转化成dict
#         return OrderedDict(legal_ids)
#
#
# def extract_player_state(state):
#     """
#     提取玩家信息
#     """
#     player_state = []
#
#     # 玩家信息
#     player_state.append(state["self_train_cars"])  # 玩家车厢（自己）
#     player_state.append(state["other_train_cars"])  # 玩家车厢（对手）
#     player_state.extend(state["self_train_car_cards"])  # 玩家车厢卡
#     player_state.extend(extract_ticket_cards_player(state["self_ticket_cards"]))  # 玩家车票卡
#     player_state.extend(
#         extract_line_map_valid_player(state["self_line_map_valid"], state["other_line_map_valid"], state["lines"]))
#     player_state.extend(
#         extract_board_map_valid_player(state["self_board_map_valid"], state["other_board_map_valid"], state["lines"]))
#     player_state.append(state["self_point"])  # 玩家分数（自己）
#     player_state.append(state["other_point"])  # 玩家分数（对手）
#     player_state.extend(extract_unique_state(state, 2))  # 区分玩家信息
#
#     return np.array(player_state)
#
#
# def extract_deck_state(state):
#     """
#     提取桌面信息
#     """
#     deck_state = []
#
#     state_train_car_cards = state["deck_train_car_cards"]
#     state_ticket_cards = []
#     ticket_cards_count = 0
#     if state["self_action_flag"] == 2:
#         for ticket_card in state["invisible_deck_ticket_cards"]:
#             state_ticket_cards.extend([ticket_card.start_pos, ticket_card.end_pos, ticket_card.point])
#             ticket_cards_count += 1
#     while ticket_cards_count < 3:
#         state_ticket_cards.extend([0, 0, 0])
#         ticket_cards_count += 1
#
#     deck_state.extend(state_train_car_cards)
#     deck_state.extend(state_ticket_cards)
#
#     return np.array(deck_state)
#
#
# def extract_unique_state(state, num_players):
#     """
#     提取区分玩家信息：onehot编码
#     """
#     state_unique = [0] * num_players
#     state_unique[state["self_player"]] = 1
#
#     return state_unique
#
#
# def extract_ticket_cards_player(ticket_cards):
#     """
#     提取车票信息
#     """
#     state_ticket_cards = []
#     for i in range(8):
#         if i < len(ticket_cards):
#             state_ticket_cards.extend([ticket_cards[i].start_pos, ticket_cards[i].end_pos, ticket_cards[i].point])
#         else:
#             state_ticket_cards.extend([0, 0, 0])
#
#     return state_ticket_cards
#
#
# def extract_line_map_valid_player(line_map_valid_self, line_map_valid_other, lines):
#     """
#     提取线路映射信息
#     """
#     state_line = []
#     for i in range(len(lines)):
#         if line_map_valid_self[i] == 1:  # 自己
#             state_line.extend([1] + lines[i][1:5])
#         elif line_map_valid_other[i] == 1:  # 对手
#             state_line.extend([2] + lines[i][1:5])
#         else:  # 都没有
#             state_line.extend([0] + lines[i][1:5])
#
#     return state_line
#
#
# def extract_board_map_valid_player(board_map_valid_self, board_map_valid_other, lines):
#     """
#     提取线路地图信息
#     """
#     board_map_valid = [-1] * 64
#
#     for line in lines:
#         board_map_valid[line[3] * 8 + line[4]] = 0
#         board_map_valid[line[4] * 8 + line[4]] = 0
#
#     for i in range(8):
#         for j in range(8):
#             if board_map_valid_self[i][j] == 1:
#                 board_map_valid[i * 8 + j] = 1
#                 board_map_valid[j * 8 + i] = 1
#             elif board_map_valid_other[i][j] == 1:
#                 board_map_valid[i * 8 + j] = 2
#                 board_map_valid[j * 8 + i] = 2
#
#     return board_map_valid


############################################
# version v1
# 包含牌的具体信息
############################################

class Ticket2RideEnv(Env):

    def __init__(self, config):
        self.name = 'ticket2ride_small_v2'
        self.game = Game(2)
        super().__init__(config)

        """
        玩家信息 （2+5+8+16+2+2=35）
        ① 车厢数目信息 （1*2=2）
        ② 车厢卡信息 （5）
        ③ 车票卡信息 （8）
        ④ 线路映射信息 （16）
        ⑤ 分数信息 （1*2=2）
        ⑥ 区分玩家信息 （2）
        """

        self.state_train_cars_player_len = 1
        self.state_train_car_cards_player_len = 5
        self.state_ticket_cards_player_len = 8
        self.state_line_map_valid_player_len = 16
        self.state_point_player_len = 1
        self.state_unique_len = self.num_players
        self.state_player_len = self.state_train_cars_player_len * 2 \
                                + self.state_train_car_cards_player_len \
                                + self.state_ticket_cards_player_len \
                                + self.state_line_map_valid_player_len \
                                + self.state_point_player_len * 2 \
                                # + self.state_unique_len

        """
        桌面信息 （5+3=8）
        ① 车厢卡信息 （5）
        ② 车票卡信息 （3）
        """
        self.state_deck_train_car_cards_len = 5
        self.state_deck_ticket_cards_len = 3
        self.state_deck_len = self.state_deck_ticket_cards_len + self.state_deck_train_car_cards_len

        self.state_len = self.state_player_len + self.state_deck_len

        self.state_shape = [[self.state_len] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, state):
        """
        状态特征提取
        """

        # 【状态特征初始化】
        obs = np.zeros(self.state_len, dtype=int)

        # 【状态特征填充】
        idxs = [
            self.state_player_len,
            self.state_player_len + self.state_deck_len
        ]
        obs[: idxs[0]] = extract_player_state(state)
        obs[idxs[0]: idxs[1]] = extract_deck_state(state)

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
    """
    提取玩家信息
    """
    player_state = []

    # 玩家信息
    player_state.append(state["self_train_cars"])  # 玩家车厢 1（自己）【√】
    player_state.append(state["other_train_cars"])  # 玩家车厢 1（对手）【√】
    player_state.extend(state["self_train_car_cards"])  # 玩家车厢卡 5【√】
    player_state.extend(extract_ticket_cards_player(state["self_ticket_cards"]))  # 玩家车票卡 8【√】
    # 已经连通的线路 16【√】
    player_state.extend(
        extract_line_map_valid_player(state["self_line_map_valid"], state["other_line_map_valid"], state["lines"]))
    player_state.append(state["self_point"])  # 玩家分数（自己）【√】
    player_state.append(state["other_point"])  # 玩家分数（对手）【√】
    # player_state.extend(extract_unique_state(state, 2))  # 区分玩家信息

    return np.array(player_state)


def extract_deck_state(state):
    """
    提取桌面信息
    """
    deck_state = []

    state_train_car_cards = state["deck_train_car_cards"]
    state_ticket_cards = []
    ticket_cards_count = 0
    if state["self_action_flag"] == 2:
        for ticket_card in state["invisible_deck_ticket_cards"]:
            state_ticket_cards.append(ticket_card.index)
            ticket_cards_count += 1
    while ticket_cards_count < 3:
        state_ticket_cards.append(-1)  # 默认值
        ticket_cards_count += 1

    deck_state.extend(state_train_car_cards)
    deck_state.extend(state_ticket_cards)

    return np.array(deck_state)


def extract_unique_state(state, num_players):
    """
    提取区分玩家信息：onehot编码
    """
    state_unique = [0] * num_players
    state_unique[state["self_player"]] = 1

    return state_unique


def extract_ticket_cards_player(ticket_cards):
    """
    提取车票信息
    """
    state_ticket_cards = [0] * 8  # 八张车票卡

    for ticket_card in ticket_cards:
        state_ticket_cards[ticket_card.index] = 1

    return state_ticket_cards


def extract_line_map_valid_player(line_map_valid_self, line_map_valid_other, lines):
    """
    提取线路映射信息
    """
    state_line = [0] * 16

    for i in range(16):
        if line_map_valid_self[i] == 1:  # 自己
            state_line[i] = 1
        if line_map_valid_other[i] == 1:  # 对手
            state_line[i] = 2

    return state_line




