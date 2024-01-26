
using Revise
using ExaModels
using MadNLP
using MadNLPHSL
using Test

nlp = goddard_model(12800)

# Test full solve
solver = MadNLPSolver(
    nlp;
    # linear_solver=HybridKKT.CHOLMODSolver,
    # kkt_system=HybridCondensedKKTSystem,
    linear_solver=Ma27Solver,
    lapack_algorithm=MadNLP.CHOLESKY,
    print_level=MadNLP.DEBUG,
    max_iter=100,
    nlp_scaling=true,
    tol=1e-4,
)
# solver.kkt.gamma[] = 1e11
MadNLP.solve!(solver)

