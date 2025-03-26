def validPath(board_map_valid, source, destination):

    def find(x):
        if p[x] != x:
            p[x] = find(p[x])
        return p[x]

    edges = []
    n = len(board_map_valid)
    for i in range(n):
        for j in range(n):
            if i < j and board_map_valid[i][j] == 1:
                edges.append([i, j])
    p = list(range(n))
    for u, v in edges:
        p[find(u)] = find(v)
    return find(source) == find(destination)


board_map_valid = [[0, 1, 1, 0],
                   [1, 0, 0, 1],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0]]

print(validPath(board_map_valid, 2, 3))