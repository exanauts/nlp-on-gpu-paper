
using Revise
using ExaModels
using MadNLP
using MadNLPHSL
using Test

using CUDA
using MadNLPGPU

# nlp_gpu = goddard_model(12800; backend=CUDABackend())

# Test full solve
solver = MadNLPSolver(
    nlp_gpu;
    linear_solver=MadNLPGPU.CUDSSSolver,
    kkt_system=HybridCondensedKKTSystem,
    # linear_solver=Ma27Solver,
    cudss_algorithm=MadNLP.BUNCHKAUFMAN,
    print_level=MadNLP.DEBUG,
    max_iter=100,
    nlp_scaling=true,
    equality_treatment=MadNLP.EnforceEquality,
    fixed_variable_treatment=MadNLP.MakeParameter,
    tol=1e-4,
)
solver.kkt.gamma[] = 1e12
@profile MadNLP.solve!(solver)
# MadNLP.solve!(solver)

