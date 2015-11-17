# mli
Machine Learning Interface for Rust

This library presently has separate generic traits for learning and genetic algorithms.

In this library a learning algorithm is one that can be trained like a neural network. A genetic algorithm is one
that can be mutated, cloned, and mated with other algorithms. All genetic algorithms must be learning algorithms.

###Plans:
- Add algorithms
  - Neural networks
    - Sigmoid
    - Radial basis function
  - Genetic
    - Tree-based
    - Mesh-based
- Add a genetic algorithm that contains a pool of learning algorithms and uses the most fit one at any given time
