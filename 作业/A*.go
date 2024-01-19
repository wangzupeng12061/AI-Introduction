package main

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Node struct {
	state         [][]int
	parent        *Node
	heuristicType string
	move          int
	g             int
	h             int
	n             int
}

type PriorityQueue []*Node

func (pq PriorityQueue) Len() int { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool {
	return (pq[i].g + pq[i].n*pq[i].h) < (pq[j].g + pq[j].n*pq[j].h)
}
func (pq PriorityQueue) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }

func (pq *PriorityQueue) Push(x interface{}) {
	item := x.(*Node)
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	*pq = old[0 : n-1]
	return item
}

func (node *Node) manhattanDistance() int {
	distance := 0
	finalState := createFinalState()
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if node.state[i][j] == finalState[i][j] || node.state[i][j] == 0 {
				continue
			} else {
				finalX := (node.state[i][j] - 1) / 4
				finalY := (node.state[i][j] - 1) % 4
				distance += abs(i-finalX) + abs(j-finalY)
			}
		}
	}
	return distance
}

func (node *Node) notformalsit() int {
	count := 1
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if node.state[i][j] != 4*i+j {
				count++
			}
		}
	}
	return count
}

func createFinalState() [][]int {
	return [][]int{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
		{13, 14, 15, 0},
	}
}

// 汉明距离
func (node *Node) hammingDistance() int {
	distance := 0
	finalState := createFinalState()
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if node.state[i][j] != 0 && node.state[i][j] != finalState[i][j] {
				distance++
			}
		}
	}
	return distance
}

// 欧几里得距离
func (node *Node) euclideanDistance() float64 {
	var distance float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if node.state[i][j] != 0 {
				finalX := (node.state[i][j] - 1) / 4
				finalY := (node.state[i][j] - 1) % 4
				dx := float64(i - finalX)
				dy := float64(j - finalY)
				distance += math.Sqrt(dx*dx + dy*dy)
			}
		}
	}
	return distance
}

func (node *Node) calculateHeuristic() float64 {
	switch node.heuristicType {
	case "manhattan":
		return float64(node.manhattanDistance())
	case "hamming":
		return float64(node.hammingDistance())
	case "euclidean":
		return node.euclideanDistance()
	default:
		return float64(node.manhattanDistance())
	}
}

func createInitialState(difficulty int) [][]int {
	finalState := createFinalState()
	state := make([][]int, 4)
	for i := range finalState {
		state[i] = make([]int, 4)
		copy(state[i], finalState[i])
	}

	// 随机移动次数基于难度级别
	rand.Seed(time.Now().UnixNano())
	for moves := 0; moves < difficulty; moves++ {
		neighbors := getNeighbors(&Node{state: state})
		state = neighbors[rand.Intn(len(neighbors))].state
	}

	return state
}

func getNeighbors(node *Node) []*Node {
	neighbors := []*Node{}
	directions := [][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}

	var i_0, j_0 int
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if node.state[i][j] == 0 {
				i_0, j_0 = i, j
			}
		}
	}

	for _, move := range directions {
		nextI, nextJ := i_0+move[0], j_0+move[1]
		if nextI >= 0 && nextI < 4 && nextJ >= 0 && nextJ < 4 {
			newState := make([][]int, 4)
			for i := range node.state {
				row := make([]int, 4)
				copy(row, node.state[i])
				newState[i] = row
			}
			newState[i_0][j_0], newState[nextI][nextJ] = newState[nextI][nextJ], newState[i_0][j_0]
			neighbor := &Node{
				state:         newState,
				parent:        node,
				move:          newState[i_0][j_0],
				heuristicType: node.heuristicType,
			}
			neighbor.h = int(neighbor.calculateHeuristic())
			neighbor.n = neighbor.notformalsit()
			neighbors = append(neighbors, neighbor)
		}
	}
	return neighbors
}

func aStar(initialState, finalState [][]int, heuristicType string) [][]int {
	openList := &PriorityQueue{}
	heap.Init(openList)
	closedSet := make(map[string]bool)
	initialNode := &Node{state: initialState,
		heuristicType: heuristicType,
	}
	initialNode.h = int(initialNode.calculateHeuristic())
	initialNode.n = initialNode.notformalsit()
	heap.Push(openList, initialNode)

	nodeNum := 0

	for openList.Len() > 0 {
		currentNode := heap.Pop(openList).(*Node)
		nodeNum++

		matches := true
		for i := range currentNode.state {
			for j := range currentNode.state[i] {
				if currentNode.state[i][j] != finalState[i][j] {
					matches = false
					break
				}
			}
		}

		if matches {
			var path [][]int
			for currentNode != nil {
				if currentNode.move != 0 {
					path = append([][]int{flatten(currentNode.state)}, path...)
				}
				currentNode = currentNode.parent
			}
			return path
		}

		key := flattenKey(currentNode.state)
		closedSet[key] = true

		for _, neighbor := range getNeighbors(currentNode) {
			if !closedSet[flattenKey(neighbor.state)] {
				neighbor.g = currentNode.g + 1
				heap.Push(openList, neighbor)
			}
		}
	}

	return nil
}

func flatten(state [][]int) []int {
	flattened := make([]int, 0, 16)
	for _, row := range state {
		flattened = append(flattened, row...)
	}
	return flattened
}

func flattenKey(state [][]int) string {
	return fmt.Sprint(flatten(state))
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func main() {

	difficulties := []int{10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000}
	heuristics := []string{"manhattan", "hamming", "euclidean"}

	for _, difficulty := range difficulties {
		fmt.Printf("难度等级: %d\n", difficulty)

		for _, heuristic := range heuristics {
			totalDuration := time.Duration(0)
			totalSteps := 0
			runs := 10

			for i := 0; i < runs; i++ {
				initialState := createInitialState(difficulty)

				// 记录开始时间
				startTime := time.Now()
				solution := aStar(initialState, createFinalState(), heuristic)
				// 记录结束时间
				endTime := time.Now()

				// 累加求解时间和步数
				totalDuration += endTime.Sub(startTime)
				totalSteps += len(solution)
			}

			// 计算平均求解时间和平均步数
			avgDuration := totalDuration / time.Duration(runs)
			avgSteps := totalSteps / runs

			fmt.Printf("使用 %s 启发函数: 平均求解时间: %v, 平均步数: %d\n", heuristic, avgDuration, avgSteps)
		}
	}

}
