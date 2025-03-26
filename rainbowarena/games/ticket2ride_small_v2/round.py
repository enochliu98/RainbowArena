from rainbowarena.games.ticket2ride_small_v2 import Dealer

POINTS = {1: 1,
          2: 2,
          3: 4,
          }

TICKET_CARDS_AVAILABLE = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
]


class Ticket2RideRound:
    """ Round can call other Classes' functions to keep the game running
    """

    def __init__(self, players, np_random, num_players):
        self.np_random = np_random  # 随机数
        self.dealer = Dealer(players, self.np_random)  # 发牌器
        self.num_players = num_players  # 玩家数目
        self.current_player = self.np_random.randint(0, self.num_players)  # 玩家编号

    def proceed_round(self, player, action):
        """
            动作空间：共三类动作，6+28+8=42
            ① 拿车厢卡：1.5+1=6（5face-up+1draw）2.需要执行两次动作选择
            ② 声明线路：1.12+4*4=28（12条有颜色线路，4条不限制颜色线路）
            ③ 拿车票卡：1.1+7=8（1首先确定是否选择拿车票+7种可拿车票的方式）2.需要执行两次动作选择
        """

        if action < 6:  # 【1】拿车厢卡
            # 【player拿车厢卡】
            if action < 5:  # face-up 拿面朝上的牌
                train_car_card_selected = self.dealer.deck[0][action]
                player.train_car_cards[train_car_card_selected] += 1
                self.dealer.deck[0][action] = -1
            else:  # draw 抽牌
                train_car_card_selected = self.dealer.train_car_cards.pop()
                player.train_car_cards[train_car_card_selected] += 1
            # 【dealer补充车厢卡】
            self.dealer.deal_cards(0)

            # 【player其他信息更新】
            if player.action_flag != 0:  # 还没有开始执行第一个动作，这一步开始执行第一个动作
                if action < 5 and train_car_card_selected == 0:  # 拿面朝上的牌且拿的是彩色牌
                    player.action_flag = -1
                else:
                    player.action_flag = 0

                flag_second_step_failure = True  # ****特殊情况：面朝上的牌都不能选或者没牌，且牌堆里也没牌****
                for train_car_card_tmp in self.dealer.deck[0]:
                    if train_car_card_tmp != -1 and train_car_card_tmp != 0:
                        flag_second_step_failure = False
                        break
                if len(self.dealer.train_car_cards) != 0:
                    flag_second_step_failure = False
                if flag_second_step_failure:
                    player.action_flag = -1

            else:  # 上步执行完了第一个动作，这一步结束肯定就换人了
                player.action_flag = -1

        elif action < 6 + 28:  # 【2】声明线路
            # 【获取线路信息】
            line_selected = action - 6
            line_selected = self.dealer.action_line[line_selected]  # 动作对应的线路
            line_info = self.dealer.board_map[0][line_selected]  # 线路信息
            # 【player更新车厢，线路信息，dealer更新车厢，线路信息】
            player.line_map_valid[line_info[0]] = 1
            player.board_map_valid[line_info[3]][line_info[4]] = 1
            player.board_map_valid[line_info[4]][line_info[3]] = 1
            self.dealer.board_map[0][line_info[0]][5] = 1
            player.train_cars -= line_info[1]
            if line_info[2] == 5:  # 不限制颜色
                train_car_type = (action - 6) - self.dealer.line_action[line_selected] + 1  # 后面加1是因为0为彩色
            else:  # 限制颜色
                train_car_type = line_info[2]
            if player.train_car_cards[train_car_type] >= line_info[1]:  # 当前颜色够用
                player.train_car_cards[train_car_type] -= line_info[1]
                self.dealer.train_car_cards.extend([train_car_type] * line_info[1])
            else:  # 当前颜色不够用
                train_car_cards_1 = player.train_car_cards[train_car_type]
                train_car_cards_2 = line_info[1] - train_car_cards_1
                player.train_car_cards[train_car_type] -= train_car_cards_1
                self.dealer.train_car_cards.extend([train_car_type] * train_car_cards_1)
                player.train_car_cards[0] -= train_car_cards_2
                self.dealer.train_car_cards.extend([0] * train_car_cards_2)
            self.np_random.shuffle(self.dealer.train_car_cards)  # 重新洗牌
            self.dealer.deal_cards(0)  # 发牌
            # 【player更新分数】
            player.point += POINTS[line_info[1]]
            # 【player其他信息更新】
            player.action_flag = -1

        else:  # 【拿车票】
            if action == 34:  # 【选择拿车票】
                player.action_flag = 2
            else:  # 【选择哪些车票】
                # 【获取车票信息】
                ticket_card_selected = TICKET_CARDS_AVAILABLE[action - 35]
                ticket_cards_tobe_delete = []
                # 【player更新车票信息，dealer更新车票信息】
                for i in range(3):
                    if ticket_card_selected[i] == 1:
                        player.ticket_cards.append(self.dealer.deck[1][i])
                        ticket_cards_tobe_delete.append(self.dealer.deck[1][i])
                for ticket_card in ticket_cards_tobe_delete:
                    self.dealer.deck[1].remove(ticket_card)

                self.dealer.deal_cards(1)
                # 【玩家其他信息更新】
                player.action_flag = -1

    def get_legal_actions(self, player):
        """
            获取可用动作：
            ① 拿车厢卡：6 第一步：6种动作都可行（除非没牌）第二步：6种动作都可行（除非有万能牌）
            ② 声明线路：28 判断四个条件: (1)车厢够不够(2)车厢卡够不够(3)是否被占用
            ③ 拿车票卡：8 第一步：1个动作可行（除非没牌）第二步：7个动作都可行（除非没牌）

        """
        legal_actions = []  # 可用动作列表

        for action in range(42):
            if action < 6:  # 【1】拿车厢卡
                if player.action_flag == -1:  # 第一步
                    if action < 5 and self.dealer.deck[0][action] != -1:  # 有牌
                        legal_actions.append(action)
                    if action == 5 and len(self.dealer.train_car_cards) != 0:  # 有牌
                        legal_actions.append(action)
                elif player.action_flag == 0:  # 第二步
                    if action < 5 and self.dealer.deck[0][action] != -1 and self.dealer.deck[0][action] != 0:  # 有牌且不彩
                        legal_actions.append(action)
                    if action == 5 and len(self.dealer.train_car_cards) != 0:  # 有牌
                        legal_actions.append(action)

            elif action < 6 + 28:  # 【2】声明线路
                if player.action_flag != -1:
                    continue
                # 【获取线路信息】
                line_selected = action - 6
                line_selected = self.dealer.action_line[line_selected]  # 动作对应的线路
                line_info = self.dealer.board_map[0][line_selected]  # 线路信息
                # 【判断逻辑】
                is_available = True
                if line_info[2] == 5:  # 不限制颜色
                    train_car_type = (action - 6) - self.dealer.line_action[line_selected] + 1
                else:  # 限制颜色
                    train_car_type = line_info[2]
                if player.train_cars < line_info[1]:  # (1)车厢不够
                    is_available = False
                if player.train_car_cards[train_car_type] + player.train_car_cards[0] < line_info[1]:  # (2)车厢卡不够
                    is_available = False
                if line_info[5] == 1:  # (3)已被占用
                    is_available = False
                if is_available:
                    legal_actions.append(action)

            else:  # 【3】抽车票
                if player.action_flag == -1:  # 第一步
                    if action == 34 and len(self.dealer.deck[1]) > 0:  # 有牌
                        legal_actions.append(action)
                elif player.action_flag == 2:  # 第二步
                    if len(self.dealer.deck[1]) == 3 and action > 34:
                        legal_actions.append(action)
                    elif len(self.dealer.deck[1]) == 2 and action in [35, 36, 38]:
                        legal_actions.append(action)
                    elif len(self.dealer.deck[1]) == 1 and action == 35:
                        legal_actions.append(action)

        return legal_actions
