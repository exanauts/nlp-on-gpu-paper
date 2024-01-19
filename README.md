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

# Friday, January 19th

* Goal of the optimization paper

Assess the capabilities of three linear solvers to solve nonlinear optimization problem on the GPU:
- Null-space method (aka reduced Hessian, Argos)
- Hybrid-condensed KKT solver (HyKKT)
- Sparse-condensed KKT with equality relaxation strategy

The two last methods require efficient sparse Cholesky available on the GPU.


* Latests developments (https://github.com/exanauts/nlp-on-gpu-paper/tree/main/scripts)

- Implementation of HyKKT in MadNLP, now gives correct result.
    * works on the GPU
    * no iterative refinement (yet): limited precision
    * solve OPF problems with tol=1e-3
    * it looks like CG is the bottleneck in the algorithm
- Full integration of cuDSS into MadNLP for sparse-Cholesky
- Integration of CHOLMOD on the CPU for comparison


* To discuss 

- Improve the HyKKT implementation
    * Implement iterative refinement on the GPU
    * Implement AMD ordering for sparse Cholesky
    * double check the accuracy of the linear solve (and its interplay with CG convergence)
    * identify the computation bottleneck and address them
- Decide what we want to showcase exactly
    * Benchmark on OPF and SCOPF instances?
    * Include additional benchmarks?
        ^ COPS benchmark in ExaModels?
        ^ PDE-constrained optimization?
