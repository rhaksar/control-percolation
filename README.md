# control-percolation

### Requirements  
- Developed with Python 3.5
- Requires the `numpy` package
- Requires the [simulators](https://github.com/rhaksar/simulators) repository: clone the repository into the root level of this repository 

### Files
- `Analysis.py`: Approximates percolation on a lattice using a set of Galton-Watson branching processes. Provides estimates of the rate of growth and stopping time of percolation. 
- `main.py`: Demonstrates use of different policies with the model approximation on simulations of forest fires.
- `Policies.py`: Defines stochastic/deterministic policies for controlling a forest fire. 
- `Utilities.py`: Helper functions to simplify implementation of model approximation and policies. 