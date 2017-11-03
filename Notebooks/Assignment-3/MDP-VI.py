import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


class Gridcell:
    def __init__(self,value,neighbors):
        self.value = value
        self.neighbors = neighbors

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

input = []
for i in range(12):
    input.append(Gridcell(0,[]))


input[3].value = 10
input[7].value = -10
input[9].value = 0
input[0].set_neighbors([1,4])
input[1].set_neighbors([0,2,5])
input[2].set_neighbors([1,3,6])
input[4].set_neighbors([0,5,8])
input[5].set_neighbors([1,4,6])
input[6].set_neighbors([2,5,7,10])
input[8].set_neighbors([4])
input[10].set_neighbors([11,6])
input[11].set_neighbors([7,10])

horizon = 20
number_of_states = 12
max_reward = 10


def perform_value_iteration(input):
    V = np.zeros((horizon,number_of_states),dtype=np.int16)
    for h in range(0,1):
        for i in range(number_of_states):
            V[h][i] = input[i].value

    for h in range(1,horizon):
        previousSum = np.sum(V[h-1])
        for i in range(number_of_states):
            maxValue = 0
            if (len(input[i].neighbors) == 0) | (V[h-1][i] == np.int16(max_reward)):
                V[h][i] = V[h-1][i]
            else:
                for j in range(len(input[i].neighbors)):
                    neighborIndex = input[i].neighbors[j].index
                    neighborValue = V[h-1][neighborIndex]
                    if V[h-1][i] + neighborValue > maxValue:
                        maxValue = V[h-1][i] + neighborValue
                V[h][i] = maxValue
        if(np.sum(V[h]) - previousSum) < 0.01:
            print "Converged at iteration:{}".format(h)
            print V
            return


perform_value_iteration(input)



