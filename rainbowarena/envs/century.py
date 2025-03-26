import numpy as np
from collections import OrderedDict

from rainbowarena.envs import Env
from rainbowarena.games.century import Game
from rainbowarena.games.century.card import CenturyFunctionCard, CenturyPointCard
import torch
import copy


class CenturyEnv(Env):

    def __init__(self, config):
        self.name = 'century'
        self.game = Game(2)
        super().__init__(config)
        """
        玩家信息 (53+53+8+4+2=120)
        ① 玩家现有的功能卡（53）
        ② 玩家打出的功能卡（53）
        ③ 玩家的token信息（4*2=8）
        ④ 玩家的分数信息（2*2=4）
        ⑤ 区分玩家信息 （2）
        """

        # 玩家现有功能卡信息
        self.state_function_cards_player_len = 53
        # 玩家打出功能卡信息
        self.state_played_function_cards_player_len = 53
        # 玩家token信息
        self.state_token_player_len = 4
        # 玩家分数信息
        self.state_point_player_len = 2
        # 区分玩家信息
        self.state_unique_len = 2
        # 玩家信息
        self.state_player_len = self.state_function_cards_player_len \
                                + self.state_played_function_cards_player_len \
                                + self.state_token_player_len * 2 \
                                + self.state_point_player_len * 2 \
                                + self.state_unique_len

        """
        桌面信息 （6+24+25+5=60）
        ① 桌面功能卡 （6）
        ② 桌面功能卡bonus （4*6=24）
        ③ 桌面分数卡 （5*5=25）
        ④ 桌面分数卡bonus （5）
        """
        # 桌面功能卡
        self.state_function_cards_deck_len = 6
        # 桌面功能卡bonus
        self.state_bonus_function_cards_deck_len = 4 * 6
        # 桌面分数卡
        self.state_point_cards_deck_len = 5 * 5
        # 桌面分数卡bonus
        self.state_bonus_point_cards_deck_len = 5 * 1
        # 桌面信息
        self.state_deck_len = self.state_function_cards_deck_len \
                              + self.state_bonus_function_cards_deck_len \
                              + self.state_point_cards_deck_len \
                              + self.state_bonus_point_cards_deck_len

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

    player_state.extend(extract_function_cards_player(state["self_function_cards"],
                                                      state["other_function_cards"]))
    player_state.extend(extract_function_cards_player(state["self_played_function_cards"],
                                                      state["other_played_function_cards"]))
    player_state.extend(state["self_tokens"])
    player_state.extend(state["other_tokens"])
    player_state.append(len(state["self_point_cards"]))
    player_state.append(state["self_point"])
    player_state.append(len(state["other_point_cards"]))
    player_state.append(state["other_point"])
    player_state.extend(extract_unique_state(state, 2))

    return np.array(player_state)


def extract_deck_state(state):
    """
    deck特征提取
    """
    deck_state = []

    deck_state.extend(extract_function_cards_deck(state["deck_function_cards"]))
    deck_state.extend(extract_function_cards_bonus_deck(state["deck_function_cards_token"]))
    deck_state.extend(extract_point_cards_deck(state["deck_point_cards"]))
    deck_state.extend(state["deck_point_cards_bonus"])

    return np.array(deck_state)


def extract_unique_state(state, num_players):
    """
    提取区分玩家信息：onehot编码
    """
    state_unique = [0] * num_players
    state_unique[state["self_player"]] = 1

    return state_unique


def extract_function_cards_player(self_function_cards_player, other_function_cards_player):
    """
    玩家功能卡信息提取
    """
    state_function_cards_player = [0] * 53
    for function_card in self_function_cards_player:
        state_function_cards_player[function_card.card_id] = 1
    for function_card in other_function_cards_player:
        state_function_cards_player[function_card.card_id] = 2
    return state_function_cards_player


def extract_function_cards_deck(function_cards_deck):
    """
    deck功能卡信息提取
    """
    state_function_cards_deck = []
    for function_card in function_cards_deck:
        if function_card == -1:
            state_function_cards_deck.append(-1)
        else:
            state_function_cards_deck.append(function_card.card_id)

    return state_function_cards_deck


def extract_function_cards_bonus_deck(function_cards_bonus_deck):
    """
    deck功能卡信息提取
    """
    state_function_cards_bonus_deck = []
    for function_card_bonus in function_cards_bonus_deck:
        state_function_cards_bonus_deck.extend(function_card_bonus)

    return state_function_cards_bonus_deck


def extract_point_cards_deck(point_cards_deck):
    """
    deck分数卡信息提取
    """
    state_point_cards_deck = []
    for point_card in point_cards_deck:
        point_card_state = point_card.cost + [point_card.point]
        state_point_cards_deck.extend(point_card_state)

    return state_point_cards_deck
