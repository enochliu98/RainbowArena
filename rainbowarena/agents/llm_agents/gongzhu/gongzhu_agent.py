import re
import time
import random
from typing import Dict, Optional
from rainbowarena.agents.llm_agents.base_llm_agent import BaseLLMAgent

card_id2des = {
    0: "Ace of Heart",
    1: "Ace of Club",
    2: "Ace of Spade",
    3: "Ace of Diamond",
    4: "2 of Heart",
    5: "2 of Club",
    6: "2 of Spade",
    7: "2 of Diamond",
    8: "3 of Heart",
    9: "3 of Club",
    10: "3 of Spade",
    11: "3 of Diamond",
    12: "4 of Heart",
    13: "4 of Club",
    14: "4 of Spade",
    15: "4 of Diamond",
    16: "5 of Heart",
    17: "5 of Club",
    18: "5 of Spade",
    19: "5 of Diamond",
    20: "6 of Heart",
    21: "6 of Club",
    22: "6 of Spade",
    23: "6 of Diamond",
    24: "7 of Heart",
    25: "7 of Club",
    26: "7 of Spade",
    27: "7 of Diamond",
    28: "8 of Heart",
    29: "8 of Club",
    30: "8 of Spade",
    31: "8 of Diamond",
    32: "9 of Heart",
    33: "9 of Club",
    34: "9 of Spade",
    35: "9 of Diamond",
    36: "10 of Heart",
    37: "10 of Club",
    38: "10 of Spade",
    39: "10 of Diamond",
    40: "Jack of Heart",
    41: "Jack of Club",
    42: "Jack of Spade",
    43: "Jack of Diamond",
    44: "Queen of Heart",
    45: "Queen of Club",
    46: "Queen of Spade",
    47: "Queen of Diamond",
    48: "King of Heart",
    49: "King of Club",
    50: "King of Spade",
    51: "King of Diamond",
}

color_id2des = {
    -1: 'No Restriction',
    0: 'Heart',
    1: 'Club',
    2: 'Spade',
    3: 'Diamond'
}


class GongzhuAgent(BaseLLMAgent):

    def __init__(
            self,
            llm_info: Optional[Dict[str, str]] = None,
    ):
        self.llm_info = llm_info
        self.stop_token = None
        for key, value in llm_info.items():
            try:
                value = float(value)
            except ValueError:
                pass
            setattr(self, key, value)
        self.messages = [{"role": "system", "content": self.prompt_rule}]
        self.illegal_counts = 0
        self.match_one = 0
        self.match_two = 0
        self.retry_counts = 0
        self.random_counts = 0
        self.use_raw = False
        self.act2des = {
            0: "Playing the Ace of Heart",
            1: "Playing the Ace of Club",
            2: "Playing the Ace of Spade",
            3: "Playing the Ace of Diamond",
            4: "Playing the 2 of Heart",
            5: "Playing the 2 of Club",
            6: "Playing the 2 of Spade",
            7: "Playing the 2 of Diamond",
            8: "Playing the 3 of Heart",
            9: "Playing the 3 of Club",
            10: "Playing the 3 of Spade",
            11: "Playing the 3 of Diamond",
            12: "Playing the 4 of Heart",
            13: "Playing the 4 of Club",
            14: "Playing the 4 of Spade",
            15: "Playing the 4 of Diamond",
            16: "Playing the 5 of Heart",
            17: "Playing the 5 of Club",
            18: "Playing the 5 of Spade",
            19: "Playing the 5 of Diamond",
            20: "Playing the 6 of Heart",
            21: "Playing the 6 of Club",
            22: "Playing the 6 of Spade",
            23: "Playing the 6 of Diamond",
            24: "Playing the 7 of Heart",
            25: "Playing the 7 of Club",
            26: "Playing the 7 of Spade",
            27: "Playing the 7 of Diamond",
            28: "Playing the 8 of Heart",
            29: "Playing the 8 of Club",
            30: "Playing the 8 of Spade",
            31: "Playing the 8 of Diamond",
            32: "Playing the 9 of Heart",
            33: "Playing the 9 of Club",
            34: "Playing the 9 of Spade",
            35: "Playing the 9 of Diamond",
            36: "Playing the 10 of Heart",
            37: "Playing the 10 of Club",
            38: "Playing the 10 of Spade",
            39: "Playing the 10 of Diamond",
            40: "Playing the Jack of Heart",
            41: "Playing the Jack of Club",
            42: "Playing the Jack of Spade",
            43: "Playing the Jack of Diamond",
            44: "Playing the Queen of Heart",
            45: "Playing the Queen of Club",
            46: "Playing the Queen of Spade",
            47: "Playing the Queen of Diamond",
            48: "Playing the King of Heart",
            49: "Playing the King of Club",
            50: "Playing the King of Spade",
            51: "Playing the King of Diamond",
        }
        self.pos2act = {value: key for (key, value) in self.act2des.items()}

    def match(self, answer):
        pattern = r"Action: (.+)"
        match = re.search(pattern, answer)
        return match

    def extract_cards(self, info):
        cards_num = 0
        cards = []
        for i in range(52):
            if info[i] == 1:
                cards.append(i)
                cards_num += 1
        return cards_num, cards

    def extract_point_cards(self, info):
        point_cards = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 43, 44, 46, 48, 49]
        cards_num = 0
        cards = []
        for i in range(16):
            if info[i] == 1:
                cards.append(point_cards[i])
                cards_num += 1
        return cards_num, cards

    def extract_dun_cards(self, info):
        dun_cards = []

        info_1 = info[: 54]
        info_2 = info[54: 108]
        info_3 = info[108: 162]

        for i in range(52):
            if info_1[i] == 1 and info_1[i+1] == 1 and info_1[i+2] == 1:
                dun_cards.append(i)

        for i in range(52):
            if info_2[i] == 1 and info_2[i+1] == 1 and info_2[i+2] == 1:
                dun_cards.append(i)

        for i in range(52):
            if info_3[i] == 1 and info_3[i+1] == 1 and info_3[i+2] == 1:
                dun_cards.append(i)

        return dun_cards

    def _get_board_status(self, player_id, observation):
        observation = observation['obs']

        self.cards_num, self.cards = self.extract_cards(observation[:52])
        self.remaining_cards_num, self.remaining_cards = self.extract_cards(observation[52: 104])
        self.point_cards_num, self.point_cards = self.extract_point_cards(observation[104: 120])
        self.point_cards_num_op_1, self.point_cards_op_1 = self.extract_point_cards(observation[120:136])
        self.point_cards_num_op_2, self.point_cards_op_2 = self.extract_point_cards(observation[136:152])
        self.point_cards_num_op_3, self.point_cards_op_3 = self.extract_point_cards(observation[152:168])

        self.dun_cards = self.extract_dun_cards((observation[168:330]))

        self.player_id = player_id

    def _get_chat_action(
            self, action_mask, observation, player_id, info=None
    ):
        max_retries = 3
        current_retry = 0

        while current_retry < max_retries:
            try:
                self._get_board_status(player_id, observation=observation)
                message = f"Your observation now is :\n"

                message += f"You are Player [{self.player_id}] \n"

                message += f"You currently have {self.cards_num} cards, which are: \n"
                for idx, card in enumerate(self.cards):
                    message += f"({idx + 1}). {card_id2des[card]} \n"

                message += f"There are {self.remaining_cards_num} cards that have not been played yet, which are: \n"
                for idx, card in enumerate(self.remaining_cards):
                    message += f"({idx + 1}). {card_id2des[card]} \n"

                if self.point_cards_num == 0:
                    message += f"You currently have 0 point cards\n"
                else:
                    message += f"You currently have {self.point_cards_num} point cards, which are: \n"
                    for idx, card in enumerate(self.point_cards):
                        message += f"({idx + 1}). {card_id2des[card]} \n"

                if self.point_cards_num_op_1 == 0:
                    message += f"Player [{(self.player_id + 1) % 4}] currently have 0 point cards\n"
                else:
                    message += f"Player [{(self.player_id + 1) % 4}] currently have" \
                               f" {self.point_cards_num_op_1} point cards, which are: \n"
                    for idx, card in enumerate(self.point_cards_op_1):
                        message += f"({idx + 1}). {card_id2des[card]} \n"

                if self.point_cards_num_op_2 == 0:
                    message += f"Player [{(self.player_id + 2) % 4}] currently have 0 point cards\n"
                else:
                    message += f"Player [{(self.player_id + 2) % 4}] currently have" \
                               f" {self.point_cards_num_op_2} point cards, which are: \n"
                for idx, card in enumerate(self.point_cards_op_2):
                    message += f"({idx + 1}). {card_id2des[card]} \n"

                if self.point_cards_num_op_3 == 0:
                    message += f"Player [{(self.player_id + 3) % 4}] currently have 0 point cards\n"
                else:
                    message += f"Player [{(self.player_id + 3) % 4}] currently have" \
                               f" {self.point_cards_num_op_3} point cards, which are: \n"
                    for idx, card in enumerate(self.point_cards_op_3):
                        message += f"({idx + 1}). {card_id2des[card]} \n"

                if len(self.dun_cards) == 0:
                    message += "Currently, no cards have been played in this trick yet\n"
                else:
                    message += f"Currently, {len(self.dun_cards)} cards have been played in this trick. " \
                               f"The cards and the players who played them are:\n"
                    for idx in range(len(self.dun_cards)):
                        message += f"({idx + 1}). Player [{(self.player_id - (len(self.dun_cards) - idx)) % 4}] " \
                                   f"played the {card_id2des[self.dun_cards[idx]]} in this trick\n"

                message += "You should think step by step and output your action. For example: 'Playing the Ace of Heart'\n"
                message += "The player with the highest card will receive the cards of all other players. Please avoid getting cards with negative points, such as hearts, queen of spades (pig), etc."
                message += f"Now you can choose one of the following actions:\n"
                for action in action_mask:
                    message += f"{self.act2des[action]} \n"
                message += f"You will respond with an action, formatted as:\n Action: <action>\n where you replace <action> with your actual action.\n"
                message += f"\nYou should explain why you choose the action\n"
                # print(message)
                answer = self._get_chat_answer(message=message)
                match = self.match(answer)
                if match:
                    pos = match.group(1)
                    pos = pos.rstrip('.')
                    # print(f"\n提取到的行动: {pos} \n")
                    action = self.pos2act.get(pos, -999)
                    # action == None代表匹配动作失败
                    if action == -999:
                        current_retry += 1
                        self.retry_counts += 1
                        self.match_two += 1
                        # print("匹配动作空间失败！")
                    # 表示匹配到了但是动作不合法
                    else:
                        if action in action_mask:
                            break
                        else:
                            # print("It is an illegal action!")
                            current_retry += 1
                            self.retry_counts += 1
                            self.illegal_counts += 1
                else:
                    # print("匹配 Action:<action> 失败！")
                    current_retry += 1
                    self.retry_counts += 1
                    self.match_one += 1
            except Exception as e:
                print(e.__context__, e)
                current_retry += 1
                self.retry_counts += 1
                time.sleep(1)
        # 如果匹配失败或者匹配到了但是动作不合法，则retry
        # retry3次如果没有动作返回，则返回随机动作。
        else:
            action = random.choice(action_mask)
            self.random_counts += 1
            # print("random action:", action)

        return action

    def act(self, player_id, observation, info=None):

        # action_space = self.env.action_space
        action_space = list(range(51))
        legal_actions = observation['legal_actions']
        action_mask = list(legal_actions.keys())
        try:
            action = self._get_chat_action(action_mask, observation, player_id, info=None)
        except Exception as e:
            print(e)
            action = random.choice(action_mask)
            self.random_counts += 1
        return action
