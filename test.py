import numpy as np 
from value_iteration import MDP, value_iteration

# Defining our 3x2 MDP
states = np.arange(6)

reward = np.array([
    -0.1,
    -0.1,
    +1,
    -0.1,
    -0.1,
    -0.05
])

actions = np.arange(5)

transition_model = {
    'up': {'left': 0.1, 'up': 0.8, 'right': 0.1, 'down':0, 'stay': 0},
    'right': {'left': 0, 'up': 0.1, 'right': 0.8, 'down': 0.1, 'stay':0},
    'down': {'right': 0.1, 'down': 0.8, 'left': 0.1, 'up':0, 'stay': 0},
    'left': {'down': 0.1, 'left': 0.8, 'up': 0.1, 'right':0, 'stay': 0},
    'stay': {'stay': 1.0, 'left': 0, 'up': 0, 'right': 0, 'down': 0}
}

# Helps building the transition matrix
def get_transtion_probability(state, action):
    transition = np.zeros(6)
    transition[state] += transition_model[action]["stay"]
    if state % 3 == 0:
        transition[state] += transition_model[action]["left"]
    else: 
        transition[state - 1] += transition_model[action]["left"]
 
    if state % 3 == 2:
        transition[state] += transition_model[action]["right"]
    else:
        transition[state + 1] += transition_model[action]["right"]

    if state / 3 == 1:
        transition[state] += transition_model[action]["up"]
    else:
        transition[state + 3] += transition_model[action]["up"]

    if state / 3 == 0:
        transition[state] += transition_model[action]["down"]
    else:
        transition[state - 3] += transition_model[action]["down"]
    return transition

# creating the transition matrix
transition = np.array([[get_transtion_probability(state, action) for action in ["up", "down", "left", "right", "stay"]] for state in states ])
transition = np.swapaxes(transition, 1, 2)

# initiales the MDP
mdp = MDP(states, actions, transition, reward, 0.999)
print("### First run, gamma = 0.999")
print(value_iteration(mdp, 0.001))
print("### second run, gamma = 0.1")
mdp.gamma = 0.1
print(value_iteration(mdp, 0.001))
