#!/bin/bash/python3
import numpy as np

class MDP:
    def __init__(self, states, actions, transition, reward, gamma):
        """Initialises an MDP model with state vector S, Actions A and transition model T
        params:
        state: vector,
        actions: vector,
        transition: tri-dimensional transition matrix representing all P(s'|s, a)
        reward: reward vector, same size as state
        gamma: scalar, discount factor
        """
        self.states = states
        self.actions = actions 
        self.transition = transition
        self.reward = reward
        self.gamma = gamma
        self.n_states = len(states)

def value_iteration(mdp, threshold):
    """ Returns an utility matrix with respect to given MDP
    """
    # current utility function
    util = np.zeros(len(mdp.states))
    # next utility function
    next_util = np.zeros(len(mdp.states))
    # maximum change of an utilitu
    delta = 1e9999
    # iteration counter
    iteration = 0
    while (delta/mdp.n_states > threshold):
        for state in mdp.states:
            next_util[state] = compute_next_utility(state, util, mdp)
        if np.linalg.norm(next_util - util) < delta:
            delta = np.linalg.norm(next_util - util)
        util = np.copy(next_util)
        iteration += 1
    print("iterations to converge: {}".format(iteration))
    return util


def compute_next_utility(state, utility, mdp):
    """Computes next utility value for given state, mdp and current utility.
    """
    action_utilities = [compute_action_utility(state, action, utility, mdp) for action in mdp.actions]
    return mdp.reward[state] + mdp.gamma * max(action_utilities)

def compute_action_utility(state, action, utility, mdp):
    """Computes the sum of utilities*transition_probabilty given an action and a state
    """
    action_utility = 0
    for next_state in mdp.states:
        action_utility += mdp.transition[state, next_state, action]*utility[next_state]
    return action_utility

if __name__ == "__main__":
    # unit testing
    states = np.arange(2)
    actions = np.arange(2)
    reward = np.arange(2.0)
    transition = np.arange(2.0**3).reshape((2,2,2))
    for action in actions:
        for state in states:
            transition[state, :, action] /= np.sum(transition[state, :, action])

    mdp = MDP(states, actions, transition, reward, 0.999)
    
    print("state :", mdp.states)
    print("actions :", mdp.actions)
    print("transition :\n", mdp.transition)
    print("reward :", mdp.reward)

    assert compute_action_utility(0, 0, reward, mdp) == 1
    assert compute_next_utility(0, np.zeros(2), mdp) == 0
    assert compute_next_utility(1, np.zeros(2), mdp) == 1.0
    assert compute_next_utility(1, reward, mdp) == 1.5994
    assert np.argmax(value_iteration(mdp, 0.01)) == 1

