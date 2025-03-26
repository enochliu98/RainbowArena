import re
import time
import random

TICKET_CARDS = [
    [0, 0, 4, 4],
    [1, 0, 7, 5],
    [2, 1, 4, 3],
    [3, 1, 7, 5],
    [4, 1, 6, 4],
    [5, 2, 4, 3],
    [6, 3, 7, 4],
    [7, 5, 6, 2]
]

LINES = [
    [0, 3, 2, 0, 6, 0, 0],
    [1, 1, 4, 0, 2, 0, 0],
    [2, 2, 1, 0, 3, 0, 0],
    [3, 2, 3, 0, 1, 0, 0],
    [4, 1, 5, 1, 3, 0, 0],
    [5, 3, 4, 1, 5, 0, 0],
    [6, 1, 2, 2, 3, 0, 0],
    [7, 2, 1, 2, 6, 0, 0],
    [8, 3, 5, 3, 5, 0, 0],
    [9, 2, 4, 3, 4, 0, 0],
    [10, 3, 3, 3, 6, 0, 0],
    [11, 1, 3, 4, 5, 0, 0],
    [12, 1, 5, 4, 6, 0, 0],
    [13, 2, 1, 4, 7, 0, 0],
    [14, 2, 2, 5, 7, 0, 0],
    [15, 2, 5, 6, 7, 0, 0],
]

COLORS = {
    0: 'Black',
    1: 'White',
    2: 'Red',
    3: 'Blue',
    4: 'Green',
    5: 'Gold',
}


class Opponent():

    def __init__(self):
        self.act2des = {
            0: "Take two gems of the same type (black)",
            1: "Take two gems of the same type (white)",
            2: "Take two gems of the same type (red)",
            3: "Take two gems of the same type (blue)",
            4: "Take two gems of the same type (green)",
            5: "Take one gem (black)",
            6: "Take one gem (white)",
            7: "Take one gem (red)",
            8: "Take one gem (blue)",
            9: "Take one gem (green)",
            10: "Take two gems of different types (black, white)",
            11: "Take two gems of different types (black, red)",
            12: "Take two gems of different types (black, blue)",
            13: "Take two gems of different types (black, green)",
            14: "Take two gems of different types (white, red)",
            15: "Take two gems of different types (white, blue)",
            16: "Take two gems of different types (white, green)",
            17: "Take two gems of different types (red, blue)",
            18: "Take two gems of different types (red, green)",
            19: "Take two gems of different types (blue, green)",
            20: "Take three gems of different types (black, blue, green)",
            21: "Take three gems of different types (black, red, green)",
            22: "Take three gems of different types (black, red, blue)",
            23: "Take three gems of different types (black, white, green)",
            24: "Take three gems of different types (black, white, blue)",
            25: "Take three gems of different types (black, white, red)",
            26: "Take three gems of different types (red, blue, green)",
            27: "Take three gems of different types (white, blue, green)",
            28: "Take three gems of different types (white, red, green)",
            29: "Take three gems of different types (white, red, blue)",
            30: "Purchase the face-up development card from the table (Level 1, Position 1)",
            31: "Purchase the face-up development card from the table (Level 1, Position 2)",
            32: "Purchase the face-up development card from the table (Level 1, Position 3)",
            33: "Purchase the face-up development card from the table (Level 1, Position 4)",
            34: "Purchase the face-up development card from the table (Level 2, Position 1)",
            35: "Purchase the face-up development card from the table (Level 2, Position 2)",
            36: "Purchase the face-up development card from the table (Level 2, Position 3)",
            37: "Purchase the face-up development card from the table (Level 2, Position 4)",
            38: "Purchase the face-up development card from the table (Level 3, Position 1)",
            39: "Purchase the face-up development card from the table (Level 3, Position 2)",
            40: "Purchase the face-up development card from the table (Level 3, Position 3)",
            41: "Purchase the face-up development card from the table (Level 3, Position 4)",
            42: "Purchase the reserved card (Position 1)",
            43: "Purchase the reserved card (Position 2)",
            44: "Purchase the reserved card (Position 3)",
            45: "Reserve the development card from the table (Level 1, Position 1) and take one gold gem",
            46: "Reserve the development card from the table (Level 1, Position 2) and take one gold gem",
            47: "Reserve the development card from the table (Level 1, Position 3) and take one gold gem",
            48: "Reserve the development card from the table (Level 1, Position 4) and take one gold gem",
            49: "Reserve the development card from the table (Level 2, Position 1) and take one gold gem",
            50: "Reserve the development card from the table (Level 2, Position 2) and take one gold gem",
            51: "Reserve the development card from the table (Level 2, Position 3) and take one gold gem",
            52: "Reserve the development card from the table (Level 2, Position 4) and take one gold gem",
            53: "Reserve the development card from the table (Level 3, Position 1) and take one gold gem",
            54: "Reserve the development card from the table (Level 3, Position 2) and take one gold gem",
            55: "Reserve the development card from the table (Level 3, Position 3) and take one gold gem",
            56: "Reserve the development card from the table (Level 3, Position 4) and take one gold gem",
        }
        self.pos2act = {value:key for (key,value) in self.act2des.items()}

    def extract_flip_cards(self, info):
        count = 0
        flip_cards = []
        for i in range(3):
            if info[i*7] != -1:
                flip_cards.append(info[i*7:i*7+7])
                count += 1
            else:
                flip_cards.append(-1)
        return count, flip_cards

    def extract_cards(self, info):
        level1_cards = []
        level2_cards = []
        level3_cards = []

        for i in range(12):
            if i < 4:
                if sum(info[i*7:i*7+7]) == 0:
                    level1_cards.append(-1)
                else:
                    level1_cards.append(info[i*7:i*7+7])
            elif i < 8:
                if sum(info[i*7:i*7+7]) == 0:
                    level2_cards.append(-1)
                else:
                    level2_cards.append(info[i*7:i*7+7])
            else:
                if sum(info[i*7:i*7+7]) == 0:
                    level3_cards.append(-1)
                else:
                    level3_cards.append(info[i*7:i*7+7])

        return level1_cards, level2_cards, level3_cards

    def extract_nobles(self, info):
        count = 0
        nobles = []
        for i in range(3):
            if info[i*7] != -1:
                nobles.append(info[i*7:i*7+7])
                count += 1
            else:
                nobles.append(-1)
        return count, nobles

    def _get_board_status(self, observation):
        self.gems = observation[:6]
        self.card_gems = observation[6:11]
        self.flip_cards_count, self.flip_cards = self.extract_flip_cards(observation[11:32])
        self.score = observation[32]
        self.gems_op = observation[33:39]
        self.card_gems_op = observation[39:44]
        self.flip_cards_count_op, self.flip_cards_op = self.extract_flip_cards(observation[44:65])
        self.score_op = observation[65]
        self.gems_deck = observation[66:72]
        self.noble_deck_count, self.noble_deck = self.extract_nobles(observation[72:93])
        self.level1_cards, self.level2_cards, self.level3_cards = self.extract_cards(observation[93:177])


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
                message += f"The number of each color of your gems: \n" \
                           f"Black ({self.gems[0]}), White ({self.gems[1]}), " \
                           f"Red ({self.gems[2]}), Blue ({self.gems[3]}), " \
                           f"Green ({self.gems[4]}), Gold ({self.gems[5]}) \n"
                message += f"The number of each color of your development cards' gem bonus: \n" \
                           f"Black ({self.card_gems[0]}), White ({self.card_gems[1]}), " \
                           f"Red ({self.card_gems[2]}), Blue ({self.card_gems[3]}), " \
                           f"Green ({self.card_gems[4]})\n"
                message += f"You currently have reserved {self.flip_cards_count} development cards, which are:\n"
                for idx, ticket in enumerate(self.flip_cards):
                    if ticket == -1:
                        message += f"Position {idx+1}. No card\n"
                    else:
                        message += f"Position {idx+1}. " \
                                   f"Cost (Black[{ticket[0]}, " \
                                   f"White[{ticket[1]}, " \
                                   f"Red[{ticket[2]}], " \
                                   f"Blue[{ticket[3]}], " \
                                   f"Green[{ticket[4]}]), " \
                                   f"Prestige Point ({ticket[5]}), " \
                                   f"Gem Bonus ({COLORS[ticket[6]]})\n"
                message += f"Your score is {self.score}\n"

                message += f"The number of each color of your opponent's gems: \n" \
                           f"Black ({self.gems_op[0]}), White ({self.gems_op[1]}), " \
                           f"Red ({self.gems_op[2]}), Blue ({self.gems_op[3]}), " \
                           f"Green ({self.gems_op[4]}), Gold ({self.gems_op[5]}) \n"
                message += f"The number of each color of your opponent's development cards' gem bonus: \n" \
                           f"Black ({self.card_gems_op[0]}), White ({self.card_gems_op[1]}), " \
                           f"Red ({self.card_gems_op[2]}), Blue ({self.card_gems_op[3]}), " \
                           f"Green ({self.card_gems_op[4]})\n"
                message += f"Your opponent currently have reserved {self.flip_cards_count_op} development cards, which are:\n"
                for idx, ticket in enumerate(self.flip_cards_op):
                    if ticket == -1:
                        message += f"Position {idx+1}. No card\n"
                    else:
                        message += f"Position {idx+1}. " \
                                   f"Cost (Black[{ticket[0]}, " \
                                   f"White[{ticket[1]}, " \
                                   f"Red[{ticket[2]}], " \
                                   f"Blue[{ticket[3]}], " \
                                   f"Green[{ticket[4]}]), " \
                                   f"Prestige Point ({ticket[5]}), " \
                                   f"Gem Bonus ({COLORS[ticket[6]]})\n"
                message += f"Your opponent's score is {self.score_op}\n"

                message += f"The number of gems of each color on the table is as follows:\n"
                message += f"Black ({self.gems_deck[0]}), " \
                           f"White ({self.gems_deck[1]}), " \
                           f"Red ({self.gems_deck[2]}), " \
                           f"Blue ({self.gems_deck[3]}), " \
                           f"Green ({self.gems_deck[4]})\n"
                message += f"There are three levels of development cards on the table, each with four development cards\n"
                message += f"The four Level 1 development cards are:\n"
                for idx, ticket in enumerate(self.level1_cards):
                    if ticket == -1:
                        message += f"Position {idx+1}. No card\n"
                    else:
                        message += f"Position {idx+1}. " \
                                   f"Cost (Black[{ticket[0]}, " \
                                   f"White[{ticket[1]}, " \
                                   f"Red[{ticket[2]}], " \
                                   f"Blue[{ticket[3]}], " \
                                   f"Green[{ticket[4]}]), " \
                                   f"Prestige Point ({ticket[5]}), " \
                                   f"Gem Bonus ({COLORS[ticket[6]]})\n"
                message += f"The four Level 2 development cards are:\n"
                for idx, ticket in enumerate(self.level2_cards):
                    if ticket == -1:
                        message += f"Position {idx+1}. No card\n"
                    else:
                        message += f"Position {idx+1}. " \
                                   f"Cost (Black[{ticket[0]}, " \
                                   f"White[{ticket[1]}, " \
                                   f"Red[{ticket[2]}], " \
                                   f"Blue[{ticket[3]}], " \
                                   f"Green[{ticket[4]}]), " \
                                   f"Prestige Point ({ticket[5]}), " \
                                   f"Gem Bonus ({COLORS[ticket[6]]})\n"
                message += f"The four Level 3 development cards are:\n"
                for idx, ticket in enumerate(self.level3_cards):
                    if ticket == -1:
                        message += f"Position {idx+1}. No card\n"
                    else:
                        message += f"Position {idx+1}. " \
                                   f"Cost (Black[{ticket[0]}, " \
                                   f"White[{ticket[1]}, " \
                                   f"Red[{ticket[2]}], " \
                                   f"Blue[{ticket[3]}], " \
                                   f"Green[{ticket[4]}]), " \
                                   f"Prestige Point ({ticket[5]}), " \
                                   f"Gem Bonus ({COLORS[ticket[6]]})\n"
                message += f"There are {self.noble_deck_count} nobles on the table, which are:"
                for idx, ticket in enumerate(self.noble_deck):
                    if ticket == -1:
                        message += f"Position {idx+1}. No card\n"
                    else:
                        message += f"Position {idx+1}. " \
                                   f"Cost (Black[{ticket[0]}, " \
                                   f"White[{ticket[1]}, " \
                                   f"Red[{ticket[2]}], " \
                                   f"Blue[{ticket[3]}], " \
                                   f"Green[{ticket[4]}]), " \
                                   f"Prestige Point ({ticket[5]}), " \
                                   f"Gem Bonus ({COLORS[ticket[6]]})\n"


                message += "You should think step by step and output your action. For example: 'Take two gems of the same type (black) \n"
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
