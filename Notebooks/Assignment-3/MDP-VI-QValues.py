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
input[2][1].value = 0



horizon = 20
number_of_states = 12
max_reward = np.float16(10)
number_of_actions = 5
discount_factor = 1.0
q_states = np.zeros((number_of_states,number_of_actions),dtype=np.float16)

def get_new_states(input_cell):
    states = []
    if(input_cell.is_terminal):
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

def perform_value_iteration(input):
    V = np.zeros((horizon,number_of_states), dtype=np.float16)
    for h in range(0,1):
        for i in range(rows):
            for j in range(cols):
                V[h][i*cols + j] = 0.0
            input.append(col_values)

    for h in range(1,horizon):
        previousSum = np.sum(V[h-1])
        for i in range(rows):
            for j in range(cols):
                maxValue = 0.0
                #Ignore the null state
                if((j == 1) & (i == 2)):
                    continue
                new_possible_states = get_new_states(input[i][j])
                stateIndex = i*cols + j
                if (len(new_possible_states) == 0) | (V[h-1][stateIndex] == np.float16(max_reward)):
                    V[h][stateIndex] = V[h-1][stateIndex]
                else:
                    for new_state in new_possible_states:
                        newStateIndex = 0
                        newStateReward = 0.0
                        if(new_state == None):#Exit happening. Terminal state
                            if input[i][j].is_terminal:
                                newStateReward = input[i][j].value
                                newStateIndex = stateIndex
                        else:
                            newStateIndex = new_state[0].rowIndex * cols + new_state[0].colIndex
                            newStateReward = input[new_state[0].rowIndex][new_state[0].colIndex].value
                        if V[h-1][newStateIndex] * discount_factor + newStateReward > maxValue:
                            maxValue = V[h-1][newStateIndex] * discount_factor + newStateReward
                            print maxValue
                    V[h][stateIndex] = maxValue
        if(np.sum(V[h]) - previousSum) < 0.01:
            print "Converged at iteration:{}".format(h)
            print V
            return



perform_value_iteration(input)



