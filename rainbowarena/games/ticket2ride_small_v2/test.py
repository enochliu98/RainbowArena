board_map_valid = []  # 已经连通的地图（8*8）
for _ in range(8):
    lst_tmp = []
    for _ in range(8):
        lst_tmp.append(0)
    board_map_valid.append(lst_tmp)

print(board_map_valid)


board_map_valid[0][0] = 100

print(board_map_valid)