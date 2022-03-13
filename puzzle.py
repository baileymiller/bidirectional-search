from tracemalloc import start
import numpy as np
from heapq import heappush, heappop
import random 

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
    return hash(self) > hash(other_state)

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

#=========================== Heuristics =============================#
def h_trivial(state, goal):
  return 0

#========================= Cost Function ===========================#
def ucs_priority(g, h):
    return g

#=========================== Solvers ================================#
def ucs_forward(start, goal):
  return unidirectional_graph_search(start, goal, h_trivial, ucs_priority)


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
  start_state = get_random_state_from_goal(goal_state, 23)
  print(start_state, '\n', goal_state)
  path, nodes_visited = ucs_forward(start_state, goal_state)
  print('path: ', path, '   nodes_visited: ', nodes_visited)
  print('Path is valid: ', validate_path(path, start_state, goal_state))