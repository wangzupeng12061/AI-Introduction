import heapq
import random
class Node: #创建储存每个节点的信息的类
    def __init__(self, state, parent=None, move=""):
        self.state = state # 记录网格的数字分布
        self.parent = parent # 记录该状态的上一状态
        self.move = move # 记录移动的是哪个数字
        self.g = 0  # 代价函数定义为走的步数
        self.h = self.heuristic()
        self.n=self.notformalsit()
        
    def __lt__(self, other):
        return (self.g +self. n*self.h) < (other.g + self.n*other.h)

    def heuristic(self):
        distance = 0
        for i in range(4):
            for j in range(4):
                if self.state[i][j] == final_state[i][j] or self.state[i][j] == 0:
                    continue
                else:
                    final_x, final_y = divmod(self.state[i][j] - 1, 4)
                    distance += abs(i - final_x) + abs(j - final_y)
        return distance
    def notformalsit(self):
        count=1
        for i in range(4):
         for j in range(4):
             if self.state[i][j]!=4*i+j:
                count+=1
        return count

def count_inversions(sequence): # 统计逆序对的个数
    inversions = 0
    for i in range(len(sequence)):
        for j in range(i + 1, len(sequence)):
            if sequence[i] > sequence[j] and sequence[j]!=0:
                inversions += 1
    return inversions

def create_initial_state():
    sequence = list(range(1,16))
    random.shuffle(sequence)
    while(count_inversions(sequence)%2 != 0): # 确保初始序列的逆序对个数为偶数（保证有解）
        random.shuffle(sequence) 
    sequence.append(0)   
    initial_state = []   
    k = 0
    for i in range(4):
        row = []
        for j in range(4):
            row.append(sequence[k])
            k += 1
        initial_state.append(row) 

    return initial_state

def get_neighbors(node):
    neighbors = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    i_0, j_0 = None, None
    for i in range(4):
        for j in range(4):
            if node.state[i][j] == 0:
                i_0, j_0 = i, j
    
    for move in directions:
        next_i, next_j = i_0 + move[0], j_0 + move[1]
        if 0 <= next_i < 4 and 0 <= next_j < 4:
            new_state = [list(row) for row in node.state]
            new_state[i_0][j_0], new_state[next_i][next_j] = new_state[next_i][next_j], new_state[i_0][j_0]
            neighbors.append(Node(new_state, node, node.state[next_i][next_j]))
    return neighbors



def a_star(initial_state, final_state):
    open_list = []
    closed_set = set()
    initial_node = Node(initial_state)
    heapq.heappush(open_list, initial_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        global node_num # 统计拓展的节点个数
        node_num += 1

        if current_node.state == final_state: # 如果达到了最终状态
            path = []
            while current_node:
                if current_node.move:
                    path.append(current_node.state)
                current_node = current_node.parent
            path.reverse()
            
            return path

        closed_set.add(tuple(map(tuple, current_node.state)))

        for neighbor in get_neighbors(current_node):
            if tuple(map(tuple, neighbor.state)) not in closed_set:
                neighbor.g = current_node.g + 1 # 累计用的步数递增
                heapq.heappush(open_list, neighbor)
                
    return None

initial_state = create_initial_state()
print("初始状态")
for row in initial_state:
    print(row)
print("\n")

node_num = 0

final_state = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 0]
]

solution = a_star(initial_state, final_state)
for step,state in enumerate(solution):
    print(f"第{step+1}步")
    for row in state:
        print(row)
    print("\n")
    
print(f"累计步数{len(solution)}")