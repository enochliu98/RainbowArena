class Ticket2RidePlayer:

    def __init__(self, player_id=-1):
        """
        玩家信息，主要包含五部分：
        ① 编号
        ② 车厢
        ③ 车票
        ④ 线路
        ⑤ 分数
        ⑥ 其他
        """

        # 【编号】
        self.player_id = player_id  # 玩家编号（1）
        # 【车厢】
        self.train_cars = 10  # 车厢数（1）
        self.train_car_cards = [0] * 5  # 车厢卡（5）
        # 【车票】
        self.ticket_cards = []  # 车票卡（n）
        # 【线路】
        self.line_map_valid = [0] * 16  # 已经连通的线路（16）
        self.board_map_valid = []  # 已经连通的地图（8*8）
        for _ in range(8):
            lst_tmp = []
            for _ in range(8):
                lst_tmp.append(0)
            self.board_map_valid.append(lst_tmp)

        # 【分数】
        self.point = 0  # 分数（1）
        # 【其他】
        self.action_flag = -1  # -1:无需关注 0:上一步执行完动作类型0的第一个动作 2:上一步执行完动作类型2的第一个动作
