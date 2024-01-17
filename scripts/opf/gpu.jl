
using PowerModels
using ExaModels
using JuMP
using CUDA
using CUDA.CUSPARSE
using MadNLPGPU

CUDA.allowscalar(false)

PowerModels.silence()

include("model.jl")

function solve_sparse_condensed_gpu(nlp)
    solver = MadNLPSolver(
        nlp;
        linear_solver=LapackGPUSolver,
        lapack_algorithm=MadNLP.CHOLESKY,
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        equality_treatment=MadNLP.RelaxEquality,
        fixed_variable_treatment=MadNLP.RelaxBound,
        print_level=MadNLP.INFO,
        max_iter=1000,
        nlp_scaling=true,
        tol=1e-4,
    )
    MadNLP.solve!(solver)
    return solver
end

function solve_hybrid_gpu(nlp)
    solver = MadNLPSolver(
        nlp;
        linear_solver=LapackGPUSolver,
        lapack_algorithm=MadNLP.CHOLESKY,
        kkt_system=HybridCondensedKKTSystem,
        print_level=MadNLP.TRACE,
        equality_treatment=MadNLP.EnforceEquality,
        fixed_variable_treatment=MadNLP.MakeParameter,
        max_iter=1,
        nlp_scaling=true,
        tol=1e-4,
    )
    MadNLP.solve!(solver)
    return solver
end

case = "/home/fpacaud/dev/pglib-opf/pglib_opf_case118_ieee.m"
case = "/home/fpacaud/dev/matpower/data/case9.m"
nlp = ac_power_model(case; backend=CUDABackend())

# solver = solve_sparse_condensed_gpu(nlp)
# solver_gpu = solve_hybrid_gpu(nlp)

solver_gpu = MadNLPSolver(
    nlp;
    linear_solver=LapackGPUSolver,
    lapack_algorithm=MadNLP.CHOLESKY,
    kkt_system=HybridCondensedKKTSystem,
    print_level=MadNLP.DEBUG,
    equality_treatment=MadNLP.EnforceEquality,
    fixed_variable_treatment=MadNLP.MakeParameter,
    max_iter=3,
    nlp_scaling=true,
    tol=1e-4,
)
MadNLP.solve!(solver_gpu)
