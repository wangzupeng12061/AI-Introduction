import heapq

MAZE_HEIGHT = 10
MAZE_WIDTH = 10


class State:
    def __init__(self, position, parent, g, h):
        self.position = position
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f


def get_neighbors(pos):
    x, y = pos
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8 directions
    return [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < MAZE_WIDTH and 0 <= y + dy < MAZE_HEIGHT]


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def astar_search(maze, start, end):
    open_list = []
    closed_set = set()
    start_state = State(start, None, 0, manhattan_distance(start, end))
    heapq.heappush(open_list, start_state)

    while open_list:
        current = heapq.heappop(open_list)

        if current.position == end:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        closed_set.add(current.position)
        for neighbor in get_neighbors(current.position):
            if neighbor in closed_set or maze[neighbor[0]][neighbor[1]] == 1:
                continue
            g = current.g + 1
            h = manhattan_distance(neighbor, end)
            heapq.heappush(open_list, State(neighbor, current, g, h))

    return None


def print_maze_with_path(maze, path):
    for i in range(MAZE_HEIGHT):
        for j in range(MAZE_WIDTH):
            if (i, j) in path:
                print('*', end=' ')
            else:
                print(maze[i][j], end=' ')
        print()


maze = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

result = astar_search(maze, (0, 0), (9, 9))
print_maze_with_path(maze, result)
if result:
    print("Path length:", len(result) - 1)
else:
    print("No path found.")
