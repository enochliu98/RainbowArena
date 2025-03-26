import itertools

# 定义维度范围
num_dimensions = 3
total_number_policies = 3
ranges = [range(total_number_policies)] * num_dimensions

# 使用集合来存储唯一的组合
unique_combinations = set()

# 使用 itertools.product 生成所有组合
for indices in itertools.product(*ranges):
    # 将元组排序，并转化为元组以确保唯一性
    sorted_indices = tuple(sorted(indices))
    unique_combinations.add(sorted_indices)

print(unique_combinations)

# 遍历唯一的组合
for combo in unique_combinations:
    print(combo)
    permutations = set(itertools.permutations(combo))
    print(len(permutations))

    indices_new = [[] for _ in range(4)]
    for perm in permutations:
        for pos in range(4):
            perm_new = list(perm)
            perm_new.insert(pos, 2)
            indices_new[pos].append(tuple(perm_new))

    print(indices_new)
