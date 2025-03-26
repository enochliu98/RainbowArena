import numpy as np
from collections import OrderedDict

from rainbowarena.envs import Env
from rainbowarena.games.papayoo import Game


class PapayooEnv(Env):

    def __init__(self, config):
        self.name = 'papayoo'
        self.game = Game()
        super().__init__(config)
        """
        玩家信息 (15+4+4=23)
        ① 玩家现有卡（15）
        ② 玩家分数 (1*4=4)
        ③ 区分玩家 (4)
        """

        self.state_cards_player_len = 15
        self.state_point_player_len = 1
        self.state_unique_len = 4
        # 玩家信息
        self.state_player_len = self.state_cards_player_len + \
                                self.state_point_player_len * 4  \
                                # self.state_unique_len

        """
        桌面信息 （4+1+1+1+1=8）
        ① 当前墩已经下的牌 (4)
        ② 当前墩的颜色 (1)
        ③ 第几墩 (1)
        ④ 第几轮 (1)
        ⑤ papayoo卡 (1)
        """
        # 桌面功能卡
        self.state_dun_cards_deck_len = 4
        self.state_dun_color_deck_len = 1
        self.state_dun_num_deck_len = 1
        self.state_round_num_deck_len = 1
        self.state_papayoo_card_deck_len = 1

        # 桌面信息
        self.state_deck_len = self.state_dun_color_deck_len + \
                              self.state_dun_cards_deck_len + \
                              self.state_dun_num_deck_len + \
                              self.state_round_num_deck_len + \
                              self.state_papayoo_card_deck_len

        # 上述相加
        self.state_len = self.state_player_len + self.state_deck_len

        self.state_shape = [[self.state_len] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, state):
        obs = np.zeros(self.state_len, dtype=int)

        idxs = [
            self.state_player_len,
            self.state_player_len + self.state_deck_len,
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
    玩家特征提取
    """
    player_state = []

    player_state.extend(extract_cards_state(state["self_cards"], 15))
    player_state.append(state["self_point"])
    player_state.extend(state["opponent_point"])
    # player_state.extend(extract_unique_state(state["self_player"], 4))

    return np.array(player_state)


def extract_deck_state(state):
    """
    deck特征提取
    """
    deck_state = []

    deck_state.extend(extract_dun_cards_state(state["deck_dun_cards"]))
    deck_state.append(state["deck_dun_color"])
    deck_state.append(state["deck_dun_num"])
    deck_state.append(state["deck_round_num"])
    deck_state.append(state["deck_papayoo_card"])

    return np.array(deck_state)


def extract_cards_state(state, max_num):
    state_cards = []
    for idx, card in enumerate(state):
        if card == 1:
            state_cards.append(idx)

    while len(state_cards) < max_num:
        state_cards.append(-1)

    return state_cards


def extract_unique_state(state, num_players):
    """
    提取区分玩家信息：onehot编码
    """
    state_unique = [0] * num_players
    state_unique[state] = 1

    return state_unique


def extract_dun_cards_state(state):
    """
    提取dun中卡特征
    """
    state_dun_cards = [dun_card[1] for dun_card in state]

    while len(state_dun_cards) < 4:
        state_dun_cards.append(-1)

    return state_dun_cards
