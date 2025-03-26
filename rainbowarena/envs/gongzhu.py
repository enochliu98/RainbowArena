import numpy as np
from collections import OrderedDict

from rainbowarena.envs import Env
from rainbowarena.games.gongzhu import Game


class GongzhuEnv(Env):

    def __init__(self, config):
        self.name = 'gongzhu'
        self.game = Game()
        super().__init__(config)
        """
        玩家信息 (13+64+4+52=133)
        ① 玩家现有卡（13）
        ② 玩家现有分数卡 (16*4=64)
        ③ 玩家分数 (1*4=4)
        ④ 剩余的卡片 (52)
        """

        self.state_cards_player_len = 52
        self.state_point_cards_player_len = 16 * 4
        self.state_remaining_cards_len = 52
        # 玩家信息
        self.state_player_len = self.state_cards_player_len + \
                                self.state_remaining_cards_len + \
                                self.state_point_cards_player_len

        """
        桌面信息 （4+1+1=6）
        ① 当前墩已经下的牌 (4)
        ② 第几墩 (1)
        ③ 第几回合 (1)
        """
        # 桌面功能卡
        self.state_dun_cards_deck_len = 54 * 3
        self.state_dun_color_deck_len = 1
        self.state_dun_num_deck_len = 1
        # 桌面信息
        self.state_deck_len = self.state_dun_cards_deck_len

        """
        历史动作信息 (8*2=16) 
        历史8步的动作，每步包含两个信息（动作编号和执行该动作的玩家编号）
        """
        self.state_history_actions_len = 16

        # 上述相加
        self.state_len = self.state_player_len + self.state_deck_len

        self.state_shape = [[self.state_len] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, state):
        obs = np.zeros(self.state_len, dtype=int)

        idxs = [
            self.state_player_len,
            self.state_player_len + self.state_deck_len
        ]

        # print(idxs)

        obs[: idxs[0]] = extract_player_state(state)
        obs[idxs[0]: idxs[1]] = extract_deck_state(state)

        legal_action_id = self._get_legal_actions()
        extracted_state = {'obs': obs, 'legal_actions': legal_action_id, 'raw_obs': state,
                           'raw_legal_actions': self.game.get_legal_actions(), 'action_record': self.action_recorder,
                           'dun_winner_id': None}
        return extracted_state

    def get_payoffs(self):
        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        return action_id  # 统一的动作编码

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()  # 获取可用动作
        legal_ids = {action: None for action in legal_actions}  # 转化成dict
        return OrderedDict(legal_ids)

    def run(self, is_training=False):
        '''
        Run a complete game, either for evaluation or training RL agent.

        Args:
            is_training (boolean): True if for training purpose.

        Returns:
            (tuple) Tuple containing:

                (list): A list of trajectories generated from the environment.
                (list): A list payoffs. Each entry corresponds to one player.

        Note: The trajectories are 3-dimension list. The first dimension is for different players.
              The second dimension is for different transitions. The third dimension is for the contents of each transiton
        '''
        trajectories = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # Agent plays
            if not is_training:
                action, _ = self.agents[player_id].eval_step(state)
            else:
                action = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
            # Save action
            trajectories[player_id].append(action)

            # 特殊信息处理
            if self.game.is_dun_over is True:
                # 当一墩结束后，获取当前一墩的胜利者
                dun_winner_id = self.game.dun_winner_id
                # 将墩胜利者信息加入到每个玩家对应的trajectory中，（-2）代表倒数第二个元素，即状态
                trajectories[player_id][-2]['dun_winner_id'] = dun_winner_id
                trajectories[(player_id - 1) % 4][-2]['dun_winner_id'] = dun_winner_id
                trajectories[(player_id - 2) % 4][-2]['dun_winner_id'] = dun_winner_id
                trajectories[(player_id - 3) % 4][-2]['dun_winner_id'] = dun_winner_id

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.num_players):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        return trajectories, payoffs


def extract_player_state(state):
    """
    玩家特征提取
    """
    player_state = []

    player_state.extend(state["self_cards"])
    player_state.extend(extract_remaining_cards_state(state))
    player_state.extend(extract_point_cards_state(state["self_point_cards"]))
    player_state.extend(extract_point_cards_state(state["opponent_point_cards"][0]))
    player_state.extend(extract_point_cards_state(state["opponent_point_cards"][1]))
    player_state.extend(extract_point_cards_state(state["opponent_point_cards"][2]))

    return np.array(player_state)


def extract_deck_state(state):
    """
    deck特征提取
    """
    deck_state = []

    deck_state.extend(extract_dun_cards_state(state["deck_dun_cards"]))

    return np.array(deck_state)


def extract_history_actions_state(state):
    """
    历史动作特征提取
    """
    history_actions_state = []

    history_actions_info = state["history_action"]
    for history_action in history_actions_info[::-1]:
        if len(history_actions_state) < 16:
            history_actions_state.append(history_action[0])
            history_actions_state.append(history_action[1])
        else:
            break

    # print(len(history_actions_state))

    while len(history_actions_state) < 16:  # 不足则填充
        history_actions_state.extend([-1, -1])

    return np.array(history_actions_state)


def extract_point_cards_state(state):
    point_cards = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 43, 44, 46, 48, 49]
    state_point_cards = [0] * 16

    for idx, point_card in enumerate(point_cards):
        if state[point_card] == 1:
            state_point_cards[idx] = 1

    return state_point_cards


def extract_remaining_cards_state(state):
    state_remaining_cards = [0] * 52

    for idx, card in enumerate(state['self_cards']):
        if card == 1:
            state_remaining_cards[idx] = 1
    for idx, card in enumerate(state['opponent_cards'][0]):
        if card == 1:
            state_remaining_cards[idx] = 1
    for idx, card in enumerate(state['opponent_cards'][1]):
        if card == 1:
            state_remaining_cards[idx] = 1
    for idx, card in enumerate(state['opponent_cards'][2]):
        if card == 1:
            state_remaining_cards[idx] = 1

    return state_remaining_cards


def extract_cards_state_opponent(state, max_num):
    state_cards = []
    for i, point_cards in enumerate(state):
        for idx, card in enumerate(point_cards):
            if card == 1:
                state_cards.append(idx)

        while len(state_cards) < max_num * (i + 1):
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
    state_dun_cards = []

    for dun_card in state:
        state_dun_cards_player = [0] * 54

        state_dun_cards_player[dun_card[1]] = 1
        state_dun_cards_player[dun_card[1] + 1] = 1
        state_dun_cards_player[dun_card[1] + 2] = 1

        state_dun_cards.extend(state_dun_cards_player)

    count = 3 - len(state)

    state_dun_cards.extend([0] * (54 * count))

    return state_dun_cards
