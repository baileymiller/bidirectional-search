from matplotlib import pyplot as plt
import numpy as np
from heapq import heappush, heappop
import random 
import matplotlib.pyplot as plt

REVERSE_ACTION = { 'l': 'r', 'r': 'l', 't': 'b', 'b': 't'}

#=========================== State and Action Helpers =============================#
class State:
  def __init__(self, init_val): 
    self.data = np.copy(np.array(init_val))
    i, j = np.where(self.data == 0) 
    self.empty_pos = (i[0], j[0])

    if not self.is_valid(): 
      raise Exception('Initial state is not valid: ', init_val)
    
  def apply(self, action):
    # ensure action is valid
    if action not in ['l', 'r', 't', 'b']:
      raise Exception(action, ' is not a valid action')

    # find new empty position (only if it is within puzzle)
    i, j = self.empty_pos
    new_empty_pos = self.empty_pos
    if action == 'l' and j > 0:
      new_empty_pos = (i, j - 1)
    elif action == 'r' and j < 3:
      new_empty_pos = (i, j + 1)
    elif action == 't' and i > 0:
      new_empty_pos = (i - 1, j)
    elif action == 'b' and i < 3:
      new_empty_pos = (i + 1, j)

    # swap tiles
    self.data[self.empty_pos], self.data[new_empty_pos] = self.data[new_empty_pos], self.data[self.empty_pos]
    self.empty_pos = new_empty_pos

    return self

  def expand(self):
    return {
      action: (1, self.get_copy().apply(action))
      for action in ['l', 'r', 't', 'b']
    }

  def is_valid(self):
    valid_shape = self.data.shape[0] == 4 and self.data.shape[1] == 4
    return valid_shape and np.all([x in self.data for x in range(0, 16)])

  def get_copy(self):
    return State(self.data)

  def __eq__(self, other_state):
    if other_state == None:
      return False
    return np.array_equal(self.data, other_state.data)

  def __gt__(self, other_state):
    return str(self) > str(other_state)

  def __str__(self):
    return str(self.data)

  def __hash__(self):
    return hash(str(self.data.flatten()))

#=========================== Priority Queue =============================#
class PQNode:
  def __init__(self, state, path, cost_from_start):
    self.state = state
    self.path = path
    self.g = cost_from_start

  def __gt__(self, other_node):
    return self.state > other_node.state

class PriorityQueue:
  def __init__(self):
    self.elements = []

  def nonempty(self):
    return bool(self.elements)

  def push(self, element, priority):
    heappush(self.elements, (priority, element))

  def pop(self):
    return heappop(self.elements)[1]
  
  def contains(self, state):
    return any(
      element.state == state
      for priority, element in self.elements
    )
  
  def size(self):
    return len(self.elements)

#=========================== Graph Search Algorithms =============================#
def unidirectional_graph_search(start, goal, heuristic, priority):
  visited = set()
  frontier = PriorityQueue()
  frontier.push(PQNode(start.get_copy(), [], 0), 0)
  curr_node = None
  while frontier.nonempty():
    curr_node = frontier.pop()
    if curr_node in visited:
      continue

    visited.add(curr_node.state)
    if curr_node.state == goal:
      break
    
    next_states = curr_node.state.expand()
    for action in next_states.keys():
      action_cost = next_states[action][0]
      next_state = next_states[action][1]

      if next_state == curr_node.state:
        continue
  
      if next_state in visited:
        continue

      next_path = curr_node.path.copy()
      next_path.append(action)
      next_node = PQNode(next_state, next_path, curr_node.g + action_cost)
      next_node_priority = priority(next_node.g, heuristic(next_state, goal))
      frontier.push(next_node, next_node_priority)
    
  return curr_node.path, len(visited)

def add_sucessors_to_frontier(curr_node, visited, visited_from, frontier, priority, heuristic, goal):
  next_states = curr_node.state.expand()
  for action in next_states.keys():
    action_cost = next_states[action][0]
    next_state = next_states[action][1]

    if next_state == curr_node.state:
      continue

    if next_state in visited:
      continue

    next_path = curr_node.path.copy()
    next_path.append(action)
    next_node = PQNode(next_state, next_path, curr_node.g + action_cost)
    next_node_priority = priority(next_node.g, heuristic(next_state, goal))
    frontier.push(next_node, next_node_priority)
    visited_from[str(next_state)] = (curr_node.state, action)

def expand_next_node(visited, visited_from, frontier, other_visited, priority, heuristic, goal):
  curr_node = frontier.pop()

  if curr_node.state in other_visited:
    return curr_node
  
  if curr_node in visited:
    return None

  visited.add(curr_node.state)

  add_sucessors_to_frontier(curr_node, visited, visited_from, frontier, priority, heuristic, goal)
  return None

def bidirectional_graph_search(start, goal, heuristic, priority, select_direction):
  visitedF = set()
  visitedB = set()

  visitedF.add(start)
  visitedB.add(goal)

  frontierF = PriorityQueue()
  frontierB = PriorityQueue()

  wf = {}
  wt = {}

  add_sucessors_to_frontier(PQNode(start.get_copy(), [], 0), visitedF, wf, frontierF, priority, heuristic, goal)
  add_sucessors_to_frontier(PQNode(goal.get_copy(), [], 0), visitedB, wt, frontierB, priority, heuristic, start)

  num_expansions = 0
  while True:
    direction = select_direction(visitedF, visitedB, frontierF, frontierB, num_expansions)
    if direction == 'forward':
      intersect_node = expand_next_node(visitedF, wf, frontierF, visitedB, priority, heuristic, goal)
    else:
      intersect_node = expand_next_node(visitedB, wt, frontierB, visitedF, priority, heuristic, start)

    num_expansions += 1
    if not intersect_node == None:
      forward_path = []
      curr_state = intersect_node.state 
      while not curr_state == start:
        prev_state, action = wf[str(curr_state)]
        forward_path.append(action)
        curr_state = prev_state
          
      backward_path = []
      curr_state = intersect_node.state
      while not curr_state == goal:
        next_state, action = wt[str(curr_state)]
        backward_path.append(REVERSE_ACTION[action])
        curr_state = next_state

      path = list(reversed(forward_path)) + backward_path
      break

  return path, len(visitedF) + len(visitedB)

#=========================== Heuristics =============================#
def h_trivial(state, goal):
  return 0

#========================= Cost Function ===========================#
def ucs_priority(g, h):
    return g

#=========Forward/Backward Expansion Selection Strategy=============#
def only_forward(_visitedF, _visitedB, _frontierF, _frontierB, _num_expansions):
  return 'forward'

def only_backward(_visitedF, _visitedB, _frontierF, _frontierB, _num_expansions):
  return 'backward'

def get_weighted_random_direction(freq):
  def weighted_random_direction(_visitedF, _visitedB, _frontierF, _frontierB, _num_expansions):
    return 'forward' if random.random() < freq else 'backward'
  return weighted_random_direction

def alternate_directions(_visitedF, _visitedB, _frontierF, _frontierB, num_expansions):
  return 'forward' if num_expansions % 2 == 0 else 'backward'

def magnitude_directions(_visitedF, _visitedB, frontierF, frontierB, _num_expansions):
  return 'forward' if frontierF.size() < frontierB.size() else 'backward'

#=========================== Solvers ================================#
def ucs_unidirectional_forward(start, goal):
  return bidirectional_graph_search(start, goal, h_trivial, ucs_priority, only_forward)

def ucs_unidirectional_backward(start, goal):
  return bidirectional_graph_search(start, goal, h_trivial, ucs_priority, only_backward)

def ucs_bidirectional_random(start, goal, dir_freq = 0.5):
  return bidirectional_graph_search(start, goal, h_trivial, ucs_priority, get_weighted_random_direction(dir_freq))

def ucs_bidirectional_determinstic(start, goal):
  return bidirectional_graph_search(start, goal, h_trivial, ucs_priority, alternate_directions)

def ucs_bidirectional_magnitude(start, goal):
  return bidirectional_graph_search(start, goal, h_trivial, ucs_priority, magnitude_directions)

#=========================== Test Helpers ================================#
def apply_random_valid_action(state):
  new_state = state.get_copy()
  while state == new_state:
    new_state = new_state.apply(random.choice(['l', 'r', 't', 'b']))
  return new_state

def get_random_goal_state():
  return State(np.random.permutation(16).reshape((4, 4)))

def get_random_state_from_goal(goal_state, num_moves):
  state = goal_state.get_copy()
  for _ in range(num_moves):
    state = apply_random_valid_action(state) 
  return state

def validate_path(path, start_state, goal_state):
  start_state = start_state.get_copy()
  [start_state.apply(action) for action in path]
  return start_state == goal_state

def test_random():
  goal_state = get_random_goal_state()
  start_state = get_random_state_from_goal(goal_state, 35)
  print(start_state, '\n', goal_state)
  path, nodes_visited = ucs_bidirectional_random(start_state, goal_state)
  print('path: ', path, '   nodes_visited: ', nodes_visited)
  print('Path is valid: ', validate_path(path, start_state, goal_state))

def test_fixed():
  start_state = State(np.array([
    [11,  4,  7,  5],
    [ 1,  8, 13,  0],
    [ 9,  3, 12, 14],
    [ 2, 15,  6, 10]
  ]))

  goal_state = State(np.array([
    [11,  4,  7,  5],
    [ 1,  3,  8, 13],
    [ 2,  9, 12, 14],
    [15,  6,  0, 10]
  ]))

  f_path, f_nodes_visited = ucs_unidirectional_forward(start_state, goal_state)
  b_path, b_nodes_visited = ucs_unidirectional_backward(start_state, goal_state)
  bi_path, bi_nodes_visited = ucs_bidirectional_random(start_state, goal_state)

  assert(validate_path(f_path, start_state, goal_state))
  assert(validate_path(b_path, start_state, goal_state))
  assert(validate_path(bi_path, start_state, goal_state))

  assert(len(f_path) == len(bi_path) and len(f_path) == len(b_path))

  print('Forward cost: ', f_nodes_visited)
  print('Backward cost: ', b_nodes_visited)
  print('Bi costs', bi_nodes_visited)

def test_weights(hardness = 5, n_samples = 10):
  forward_freq = np.linspace(0, 1, 11)
  forward_freq_cost = np.zeros_like(forward_freq)
  for i in range(n_samples):
    goal_state = get_random_goal_state()
    start_state = get_random_state_from_goal(goal_state, hardness)
    for j in range(len(forward_freq)):
      _, cost = ucs_bidirectional_random(start_state, goal_state, forward_freq[j])
      forward_freq_cost[j] += float(cost)
    print('sample ', i, ' computed.')

  forward_freq_cost /= float(n_samples)

  plt.plot(forward_freq, forward_freq_cost)
  plt.show()

def test_deterministic(hardness = [3, 4, 5, 6, 7, 8] , n_samples = 10):
  deterministic_cost = np.zeros((len(hardness)))
  random_cost = np.zeros((len(hardness)))
  for i in range(len(hardness)):
    for _ in range(n_samples):
      goal_state = get_random_goal_state()
      start_state = get_random_state_from_goal(goal_state, hardness[i])
      _, deterministic_cost_sample = ucs_bidirectional_determinstic(start_state, goal_state)
      _, random_cost_sample = ucs_bidirectional_random(start_state, goal_state, 0.5)
      deterministic_cost[i] += deterministic_cost_sample
      random_cost[i] += random_cost_sample
    print('hardness ', i, ' computed.')

  deterministic_cost /= n_samples
  random_cost /= n_samples

  plt.plot(hardness, deterministic_cost, color='green')
  plt.plot(hardness, random_cost, color='red')
  plt.show()

def test_alternate(hardness = [10, 15, 20] , n_samples = 10):
  alternate_cost = np.zeros((len(hardness)))
  magnitude_cost = np.zeros((len(hardness)))
  for i in range(len(hardness)):
    for _ in range(n_samples):
      goal_state = get_random_goal_state()
      start_state = get_random_state_from_goal(goal_state, hardness[i])
      _, alternate_cost_sample = ucs_bidirectional_determinstic(start_state, goal_state)
      _, magnitude_cost_sample = ucs_bidirectional_magnitude(start_state, goal_state)
      alternate_cost[i] += alternate_cost_sample
      magnitude_cost[i] += magnitude_cost_sample
    print('hardness ', i, ' computed.')

  alternate_cost /= n_samples
  magnitude_cost /= n_samples

  plt.plot(hardness, alternate_cost, color='green')
  plt.plot(hardness, magnitude_cost, color='red')
  plt.show()


test_alternate()