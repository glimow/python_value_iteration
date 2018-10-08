# ROB311 TP2: Utility function of simple MDP

This repository contains an MDP Utility function for ROB311's TP2 at ENSTA ParisTech.
It is separated into two files:

- `value_iteration.py`, that contains a quickly unit-tested implementation of the Value Iteration Algorithm.
    it is heavily inspired from the one in Russel and Norvig's _AI, a modern approach_ chapter 17, but with a tweak in the
    while loop condition to match the course's one. It can be used to compute any MDP's Utility function as long as it's transition
    matrices are availables.
- `test.py` contains a test of our Value Iteration Algorithm used on the 2x3 MDP problem seen in class.

## Installation and usage

To install numpy with pip:

`pip3 install -r requirements.txt`

To run unit tests:

`python3 value_iteration.py`

To run TP2's 3x2 problem:

`python3 test.py`

## Questions

1) It takes 6010 Iterations for the utility to converge with `gamma=0.999` and `threshold=0.01`
2) with `gamma=0.1`, it takes only 4 iterations to converge. We observe that the found policy is the same.