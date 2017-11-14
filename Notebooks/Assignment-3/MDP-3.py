import os
import numpy as np
import math as math
import matplotlib.pyplot as plt
from skimage.transform import resize


class Gridcell:
    def __init__(self,value,neighbors, rowIndex, colIndex):
        self.value = value
        self.neighbors = neighbors
        self.rowIndex = rowIndex
        self.colIndex = colIndex
        self.is_terminal = False

    def set_terminal(self, terminal):
        self.is_terminal = terminal

    def set_neighbors(self, indices):
        for i in indices:
            self.neighbors.append(Neighbor(input[i], i, 1.0))

class Neighbor:
    def __init__(self, cell, index, probability):
        self.cell = cell
        self.probability = probability
        self.index = index

    def is_terminal(self):
        if not self.index == 3 | self.index == 7:
            return True
        else:
            return False

rows = 3
cols = 4
input = []
for i in range(rows):
    col_values = []
    for j in range(cols):
        col_values.append(Gridcell(0,[],i,j))
    input.append(col_values)


input[0][3].value = 10
input[1][3].value = -10
input[0][3].set_terminal(True)
input[1][3].set_terminal(True)
input[2][1].set_terminal(True)
input[2][1].value = 0



horizon = 10
number_of_states = 13
max_reward = np.float16(10)
number_of_actions = 5
gamma = 0.9
MAX_REWARD = 10.0
q_states = np.zeros((number_of_states,number_of_actions),dtype=np.float16)
terminal = np.zeros((number_of_states), dtype = np.bool)
R = np.zeros((number_of_states,number_of_states),dtype=np.float16)
Policy = ['']*number_of_states

R[3][12] = 10.0
R[7][12] = -10.0

def get_state_transition_reward(state, new_state):
    return R[state][new_state]

def get_state_transition_probability(state, new_state):
    return 1.0

def get_new_states(input_cell):
    states = []
    if(input_cell.is_terminal):
        states.append((Gridcell(0,[],3,0),'exit'))
        return states
    #go left
    if(input_cell.colIndex > 0):
        states.append((input[input_cell.rowIndex][input_cell.colIndex - 1],'left'))
    #go right
    if(input_cell.colIndex < cols - 1):
        states.append((input[input_cell.rowIndex][input_cell.colIndex + 1],'right'))
    #go up
    if(input_cell.rowIndex > 0):
        states.append((input[input_cell.rowIndex - 1][input_cell.colIndex],'up'))
    #go down
    if(input_cell.rowIndex < rows - 1):
        states.append((input[input_cell.rowIndex + 1][input_cell.colIndex],'down'))

    return states

def get_new_state(input_cell, action):
    if action == 0: #go left
        if(input_cell.colIndex > 0):
            return input[input_cell.rowIndex][input_cell.colIndex - 1]
    elif action == 1: #go right
        if(input_cell.colIndex < cols - 1):
            return input[input_cell.rowIndex][input_cell.colIndex + 1]
    elif action == 2: #go up
        if(input_cell.rowIndex > 0):
            return input[input_cell.rowIndex - 1][input_cell.colIndex]
    elif action == 3: #go down
        if(input_cell.rowIndex < rows - 1):
            return input[input_cell.rowIndex + 1][input_cell.colIndex]
    elif action == 4: #exit
        return None

def get_state_index(input_cell):
    m = cols * (input_cell.rowIndex) + input_cell.colIndex
    return m;

def perform_value_iteration(input):
    V = np.zeros((horizon,number_of_states), dtype=np.float16)
    for h in range(0,1):
        for i in range(rows):
            for j in range(cols):
                V[h][i*cols + j] = 0.0
            input.append(col_values)
    first_policy_processed = False
    for h in range(1,horizon):
        previous_sum = np.sum(V[h - 1])
        for i in range(0,number_of_states-1):
            state = i
            if(terminal[state]):
                V[h][state] = V[h-1][state]
                continue
            values_so_far = []
            actions_so_far = []
            m = i/(rows +1)
            n = i%cols
            #print m,n
            for state_context in get_new_states(input[m][n]):
                new_state = state_context[0]
                action = state_context[1]
                if ((not first_policy_processed) & (action != 'up')):
                    continue
                else:
                    #Only allow Up for the first round
                    new_state_index = get_state_index(new_state)
                    state_transition_probability = get_state_transition_probability(state, new_state_index)
                    state_transition_reward = get_state_transition_reward(state, new_state_index)
                    discounted_future_reward = gamma * V[h-1][new_state_index]
                    value = state_transition_probability *(state_transition_reward +
                                                           discounted_future_reward)
                    values_so_far.append(value)
                    actions_so_far.append(str(action))
                    if(action == 'exit'):
                        terminal[state] = True
            if(len(values_so_far) == 0):
                V[h][state] = 0
                max_action_index = 0
                Policy[state] = 'None'
            else:
                V[h][state] = max(values_so_far)
                max_action_index = values_so_far.index(max(values_so_far))
                Policy[state] = actions_so_far[max_action_index]
            if(V[h][state] >= MAX_REWARD):
                terminal[state] = True
        new_sum = np.sum(V[h])
        first_policy_processed = True #Up policy processed
        print "\nPolicy - Iteration-{}:\n{}".format(h,np.array(Policy[0:number_of_states -1]).reshape((3,4)))
        print "Policy Quality:{}".format(np.sum(V[h]))
        if(h > 5): #Let the first few iterations go through for convergence check.
            if(new_sum - previous_sum) < 0.01:
                print "\nConverged at iteration:{}. Values:".format(h)
                print np.array(V[h][0:number_of_states -1]).reshape((3,4))
                print "\n\nOptimal policy:\n{}".format(np.array(Policy[0:number_of_states -1]).reshape((3,4)))
                return
        #print V[h][0:number_of_states -1]


first_policy_processed = False
perform_value_iteration(input)



