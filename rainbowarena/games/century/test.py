# def generate_action_space(num_upgrade):
#     '''
#     生成可升级的动作空间
#     '''
#     action_space = []
#     for n_upgrade in range(num_upgrade+1):  # [0, 1, 2, 3]
#         for x_1 in range(n_upgrade+1):
#             for x_2 in range(n_upgrade+1):
#                 for x_3 in range(n_upgrade+1):
#                         if x_1 + x_2 + x_3 == n_upgrade:
#                             action_space.append([x_1, x_2, x_3])
#     return action_space
#
# UPGRADE_ACTION_SPACE = generate_action_space(3) # 最大升级数为3
#
# print(UPGRADE_ACTION_SPACE)
#
# print(len(UPGRADE_ACTION_SPACE))

CITY_MAP = []

for _ in range(8):
    list_tmp = []
    for _ in range(8):
        list_tmp.append([])
    CITY_MAP.append(list_tmp)

CITY_MAP[0][0].append(1)

print(CITY_MAP)


CITY_MAP_2 = [[[]] * 8] * 8

CITY_MAP_2[0][0].append(1)

print(CITY_MAP_2)
