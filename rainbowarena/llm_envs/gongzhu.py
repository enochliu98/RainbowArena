import re
import time
import random

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

class Opponent():

    def __init__(self):
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

    def extract_self_cards(self, info):
        card_num = 0
        cards = []
        for card in info:
            if card != -1:
                card_num += 1
                cards.append(card)
        return card_num, cards

    def extract_opponent_cards(self, info):
        card_num = [0] * 3
        cards = [[] for _ in range(3)]
        for idx, card in enumerate(info):
            if idx < 16:
                if card != -1:
                    card_num[0] += 1
                    cards[0].append(card)
            elif idx < 32:
                if card != -1:
                    card_num[1] += 1
                    cards[1].append(card)
            else:
                if card != -1:
                    card_num[2] += 1
                    cards[2].append(card)

        return card_num, cards

    def extract_player_id(self, info):
        player_id = -1
        if info[0] == 1:
            player_id = 0
        elif info[1] == 1:
            player_id = 1
        elif info[2] == 1:
            player_id = 2
        else:
            player_id = 3

        return player_id

    def extract_dun_cards(self, info):
        dun_cards = []

        for dun_card in info:
            if dun_card != -1:
                dun_cards.append(dun_card)

        return dun_cards

    def _get_board_status(self, observation):
        self.cards_num, self.cards = self.extract_self_cards(observation[:13])
        self.point_cards_num, self.point_cards = self.extract_self_cards(observation[13:29])
        self.score = observation[29]
        self.point_cards_num_ops, self.point_cards_ops = self.extract_opponent_cards(observation[30:78])
        self.score_ops = observation[78:81]
        self.player_id = self.extract_player_id(observation[81:85])
        self.dun_cards = self.extract_dun_cards(observation[85:89])
        self.dun_color = observation[89]
        self.dun_num = observation[90]

    def _get_chat_action(
            self, action_mask, observation, info=None
    ):
        max_retries = 3
        current_retry = 0
        illegal_count = 0

        while current_retry < max_retries:
            try:
                self._get_board_status(observation=observation)
                message = f"Your observation now is :\n"

                if self.cards_num == 13:
                    message += "ATTENTION！A NEW ROUND HAS STARTED\n"

                message += f"You are Player [{self.player_id}] \n"

                message += f"You currently have {self.cards_num} cards, which are: \n"
                for idx, card in enumerate(self.cards):
                    message += f"({idx+1}). {card_id2des[card]} \n"
                if self.point_cards_num == 0:
                    message += f"You currently have 0 point cards\n"
                else:
                    message += f"You currently have {self.point_cards_num} point cards, which are: \n"
                    for idx, card in enumerate(self.point_cards):
                        message += f"({idx+1}). {card_id2des[card]} \n"
                message += f"your current score is {self.score} points \n"

                if self.point_cards_num_ops[0] == 0:
                    message += f"Player [{(self.player_id + 1) % 4}] currently have 0 point cards\n"
                else:
                    message += f"Player [{(self.player_id + 1) % 4}] currently have" \
                               f" {self.point_cards_num_ops[0]} point cards, which are: \n"
                    for idx, card in enumerate(self.point_cards_ops[0]):
                        message += f"({idx+1}). {card_id2des[card]} \n"
                message += f"Player [{(self.player_id + 1) % 4}]'s current score is {self.score_ops[0]} points \n"

                if self.point_cards_num_ops[1] == 0:
                    message += f"Player [{(self.player_id + 2) % 4}] currently have 0 point cards\n"
                else:
                    message += f"Player [{(self.player_id + 2) % 4}] currently have" \
                           f" {self.point_cards_num_ops[1]} point cards, which are: \n"
                for idx, card in enumerate(self.point_cards_ops[1]):
                    message += f"({idx+1}). {card_id2des[card]} \n"
                message += f"Player [{(self.player_id + 2) % 4}]'s current score is {self.score_ops[1]} points \n"

                if self.point_cards_num_ops[2] == 0:
                    message += f"Player [{(self.player_id + 3) % 4}] currently have 0 point cards\n"
                else:
                    message += f"Player [{(self.player_id + 3) % 4}] currently have" \
                               f" {self.point_cards_num_ops[2]} point cards, which are: \n"
                    for idx, card in enumerate(self.point_cards_ops[2]):
                        message += f"({idx+1}). {card_id2des[card]} \n"
                message += f"Player [{(self.player_id + 3) % 4}]'s current score is {self.score_ops[2]} points \n"

                if len(self.dun_cards) == 0:
                    message += "Currently, no cards have been played in this trick yet\n"
                else:
                    message += f"Currently, {len(self.dun_cards)} cards have been played in this trick. " \
                               f"The cards and the players who played them are:\n"
                    for idx in range(len(self.dun_cards)):
                        message += f"({idx+1}). Player [{(self.player_id + 4 - (3 - idx)) % 4}] " \
                                   f"played the {card_id2des[self.dun_cards[idx]]} in this trick\n"

                message += f"The current trick's suit is {color_id2des[self.dun_color]}\n"

                message += f"{self.dun_num} tricks have been completed in the current round\n"


                message += "You should think step by step and output your action. For example: 'Playing the Ace of Heart'\n"
                message += f"Now you can choose one of the following actions:\n"
                for action in action_mask:
                    message += f"{self.act2des[action]} \n"
                message += f"You will respond with an action, formatted as:\n Action: <action>\n where you replace <action> with your actual action.\n"
                message += f"\nYou should explain why you choose the action\n"
                print(message)
                answer = self._get_chat_answer(message=message)
                pattern = r"Action: (.+)"
                match = re.search(pattern, answer)
                if match:
                    pos = match.group(1)
                    pos = pos.rstrip('.')
                    print("提取的行动:", pos)
                    action = int(self.pos2act[pos])

                    Opponent.opponent_action = pos
                    break
                else:
                    print("未找到匹配。")
                    current_retry += 1
            except Exception as e:
                print(e.__context__, e)
                current_retry += 1
                time.sleep(1)
        else:
            indices_of_ones = [i for i, value in enumerate(action_mask) if value == 1]
            action = random.choice(indices_of_ones)
            print("random action:", action)
        if action not in action_mask:
            illegal_count += 1

        return action
