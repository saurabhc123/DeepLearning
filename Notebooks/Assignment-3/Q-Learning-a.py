import numpy as np
import csv as csv

number_of_states = 2
number_of_actions = 2

Q = np.zeros((number_of_states, number_of_actions),dtype=np.float16)
counts = np.zeros((number_of_states, number_of_actions),dtype=np.int16)
gamma = 0.9

def get_alpha1(state,action):
    return 0.1
    w= counts[state][action]
    return 1/(w+1)

def get_alpha():
    return 0.1

def get_values_from_sample(s):
    return int(s[0]) - 1, int(s[1]) - 1,int(s[2]) - 1,int(s[3])

def getMaxForState(new_state):
    return np.max(Q[new_state])

def run_Q_Learning(samples):
    for s in samples:
        initial_state,action,new_state,reward = get_values_from_sample(s)
        alpha = get_alpha()
        old_Q_s_a = Q[initial_state][action]
        sample = reward + gamma * getMaxForState(new_state)
        Q[initial_state][action] = (1 - alpha) * old_Q_s_a + alpha * sample
        #print Q
    print Q

def load_data():
    samples = []
    inputFilenameWithPath = 'q-learning.dat'
    with open(inputFilenameWithPath, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            samples.append(row)
    return samples

samples = load_data()
run_Q_Learning(samples)