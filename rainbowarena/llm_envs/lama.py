import re
import time
import random

card_id2des = {
    0: "1",
    1: "2",
    2: "3",
    3: "4",
    4: "5",
    5: "6",
    6: "Lama",
}

class Opponent():

    def __init__(self):
        self.act2des = {
            0: "Playing 1",
            1: "Playing 2",
            2: "Playing 3",
            3: "Playing 4",
            4: "Playing 5",
            5: "Playing 6",
            6: "Playing Lama",
            7: "Drawing a card from the draw pile",
            8: "Passing (You are out for the rest of the round and keep the cards in your hand)",
        }
        self.pos2act = {value: key for (key, value) in self.act2des.items()}

    def _get_board_status(self, observation):
        self.cards_num = sum(observation[:7])
        self.cards = observation[:7]
        self.score = observation[7]
        self.quit = observation[8]
        self.score_op_1 = observation[9]
        self.quit_op_1 = observation[10]
        self.score_op_2 = observation[11]
        self.quit_op_2 = observation[12]
        self.deck_card = observation[13]

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

                message += f"You currently have {self.cards_num} cards, which are: \n"
                message += f"Card 1 (Num:{self.cards[0]}) \n" \
                           f"Card 2 (Num:{self.cards[1]}) \n" \
                           f"Card 3 (Num:{self.cards[2]}) \n" \
                           f"Card 4 (Num:{self.cards[3]}) \n" \
                           f"Card 5 (Num:{self.cards[4]}) \n" \
                           f"Card 6 (Num:{self.cards[5]}) \n" \
                           f"Card Lama (Num:{self.cards[6]}) \n"
                message += f"Your score is {self.score} \n"
                if self.quit == 1:
                    message += "You are out for the rest of the round \n"
                else:
                    message += "You are not out for the rest of the round \n"

                message += f"The next player's score is {self.score_op_1} \n"
                if self.quit_op_1 == 1:
                    message += "The next player is out for the rest of the round \n"
                else:
                    message += "The next player is not out for the rest of the round \n"

                message += f"The previous player's score is {self.score_op_2} \n"
                if self.quit_op_2 == 1:
                    message += "The previous player is out for the rest of the round \n"
                else:
                    message += "The previous player is not out for the rest of the round \n"

                message += f"The top card of the discard pile is {card_id2des[self.deck_card]} \n"

                message += "You should think step by step and output your action. For example: 'Playing 1'\n"
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
