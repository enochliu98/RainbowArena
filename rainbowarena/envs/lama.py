import numpy as np
from collections import OrderedDict

from rainbowarena.envs import Env
from rainbowarena.games.lama import Game


class LamaEnv(Env):

    def __init__(self, config):
        self.name = 'lama'
        self.game = Game(3)
        super().__init__(config)
        """
        玩家信息 (7+3+3=13)
        ① 玩家现有卡（7）
        ② 玩家分数（1*3=3）
        ③ 玩家退出 (1*3=3)
        """

        self.state_cards_player_len = 7
        self.state_point_player_len = 1
        self.state_quit_player_len = 1
        self.state_unique_len = 3
        # 玩家信息
        self.state_player_len = self.state_cards_player_len + \
                                self.state_quit_player_len * 3 + \
                                self.state_point_player_len * 3
                                # self.state_unique_len

        """
        桌面信息 （1）
        ① 桌面卡 （1）
        """
        # 桌面功能卡
        self.state_card_deck_len = 1
        # 桌面信息
        self.state_deck_len = self.state_card_deck_len

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

    player_state.extend(state["self_cards"])
    player_state.append(state["self_point"])
    player_state.append(state["self_quit"])
    player_state.append(state["opponent_1_point"])
    player_state.append(state["opponent_1_quit"])
    player_state.append(state["opponent_2_point"])
    player_state.append(state["opponent_2_quit"])
    # player_state.extend(extract_unique_state(state, 3))

    return np.array(player_state)


def extract_deck_state(state):
    """
    deck特征提取
    """
    deck_state = [state["deck_card"]]

    return np.array(deck_state)


def extract_unique_state(state, num_players):
    """
    提取区分玩家信息：onehot编码
    """
    state_unique = [0] * num_players
    state_unique[state["self_player"]] = 1

    return state_unique

