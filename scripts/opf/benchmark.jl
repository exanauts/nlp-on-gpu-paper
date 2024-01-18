
using PowerModels
using ExaModels
using JuMP
using CUDA
using CUDA.CUSPARSE
using MadNLPGPU

CUDA.allowscalar(false)

PowerModels.silence()

if !@isdefined ac_power_model
    include("model.jl")
end

function solve_hsl(nlp; options...)
    solver_sparse = MadNLPSolver(
        nlp;
        linear_solver=Ma27Solver,
        print_level=MadNLP.INFO,
        options...
    )
    MadNLP.solve!(solver_sparse)
    return solver_sparse
end

function solve_sparse_condensed(nlp; options...)
    solver = MadNLPSolver(
        nlp;
        lapack_algorithm=MadNLP.CHOLESKY,
        cudss_algorithm=MadNLP.CHOLESKY,
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        equality_treatment=MadNLP.RelaxEquality,
        fixed_variable_treatment=MadNLP.RelaxBound,
        options...
    )
    MadNLP.solve!(solver)
    return solver
end

function solve_hybrid(nlp; options...)
    solver = MadNLPSolver(
        nlp;
        lapack_algorithm=MadNLP.CHOLESKY,
        cudss_algorithm=MadNLP.CHOLESKY,
        kkt_system=HybridCondensedKKTSystem,
        equality_treatment=MadNLP.EnforceEquality,
        fixed_variable_treatment=MadNLP.MakeParameter,
        options...
    )
    MadNLP.solve!(solver)
    return solver
end

case = "/home/fpacaud/dev/pglib-opf/pglib_opf_case2000_goc.m"

# HSL Ma27
nlp = ac_power_model(case)
solve_hsl(nlp; tol=1e-3)

# SparseCondensedKKT + CHOLMOD
solve_sparse_condensed(nlp; linear_solver=CHOLMODSolver, tol=1e-3)

# HybridKKT + CHOLMOD
solve_hybrid(nlp; linear_solver=CHOLMODSolver, tol=1e-3)

nlp_gpu = ac_power_model(case; backend=CUDABackend())
# SparseCondensedKKT + CUDSS
solve_sparse_condensed(nlp_gpu; linear_solver=MadNLPGPU.CUDSSSolver, tol=1e-3)

# HybridKKT + CHOLMOD
solve_hybrid(nlp_gpu; linear_solver=MadNLPGPU.CUDSSSolver, tol=1e-3)
