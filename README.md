# nlp-on-gpu-paper

Make it a review paper comparing different methods for implementing a nonlinear sparse, large-scale optimization solver on the GPU. We will focus on methods, not the implementation. For numerical comparison, we will use MadNLP. We will compare:
  - HyKKT method
  - condensed space inequality relaxation
  - condensed then reduce
    
  For sparse solver, we compare two options:
  - CUDSS
  - CUSOLVERRF

  Portability (not our primary focus, but if we want to say something)
  - https://github.com/ORNL/ReSolve/blob/v0.99.1/resolve/LinSolverDirectRocSolverRf.cpp
