# control-percolation

A repository to support the paper **Controlling Heterogeneous Stochastic Growth Processes on Lattices with Limited Resources**.

Paper citation:
```
@InProceedings{9029805,
  author={R. N. {Haksar} and F. {Solowjow} and S. {Trimpe} and M. {Schwager}},
  booktitle={2019 IEEE 58th Conference on Decision and Control (CDC)}, 
  title={Controlling Heterogeneous Stochastic Growth Processes on Lattices with Limited Resources}, 
  year={2019},
  pages={1315-1322},}
```

### Requirements  
- Developed with Python 3.6
- Requires the `numpy` package
- Requires the [simulators](https://github.com/rhaksar/simulators) repository

### Files
- `Analysis.py`: Approximates percolation on a lattice using a set of Galton-Watson branching processes. Provides estimates of the rate of growth and stopping time of percolation. 
- `main.py`: Demonstrates use of different policies with the model approximation on simulations of forest fires.
- `Policies.py`: Defines stochastic/deterministic policies for controlling a forest fire. 
- `Utilities.py`: Helper functions to simplify implementation of model approximation and policies. 
- `Visualize.py`: Generates box plot visualization of results.
