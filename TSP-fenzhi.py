import sys


def tsp_divide_and_conquer(dist):
    n = len(dist)
    min_cost = sys.maxsize  # 初始化最小成本为系统的最大值
    all_routes = []     # 用于存储所有最短路径

    # 用于存储路径的递归函数
    def dfs(path, visited, cost):
        nonlocal min_cost, all_routes  # 上一级的局部变量
        if len(path) == n:  # 如果
            total_cost = cost + dist[path[-1]][path[0]]  # 回到起点的距离
            if total_cost < min_cost:
                min_cost = total_cost
                all_routes = [path + [path[0]]]  # 找到新的最短路径
            elif total_cost == min_cost:
                all_routes.append(path + [path[0]])  # 记录所有最短路径
            return

        for i in range(n):
            if not visited[i]:
                visited[i] = True
                dfs(path + [i], visited, cost + (dist[path[-1]][i] if path else 0))     # 递归遍历所有城市 得到所有连通的路径并得到路径损失
                visited[i] = False

    # 从城市 0 开始
    visited = [False] * n
    visited[0] = True
    dfs([0], visited, 0)

    # 转换路径为城市字母形式
    mapping = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
            5: "F",
            6: "G",
            7: "H"
            }
    all_routes = [[mapping[i] for i in route] for route in all_routes]

    return min_cost, all_routes


# 城市距离矩阵
max = float('inf')
dist = [
    [0, 1, max, max, max, max, 0.5, 1],
    [1, 0, 1, max, max, max, 1, max],
    [max, 1, 0, 1, max, 2, max, max],
    [max, max, 1, 0, 5, max, max, max],
    [max, max, max, 5, 0, 1, max, max],
    [max, max, 2, max, 1, 0, 0.5, 1],
    [0.5, 1, max, max, max, 0.5, 0, max],
    [1, max, max, max, max, 1, max, 0]
]

min_cost, all_routes = tsp_divide_and_conquer(dist)
print(f"最短路径长度: {min_cost}")
print(f"所有最短路径:")
for route in all_routes:
    print(route)
