
using LinearAlgebra
using CUDA
using MadNLP
using MadNLPHSL
using MadNLPGPU
using HybridKKT

CUDA.allowscalar(false)

function refresh_memory()
    GC.gc(true)
    CUDA.has_cuda() && CUDA.reclaim()
    return
end

function build_ma27_solver(nlp; options...)
    return MadNLPSolver(
        nlp;
        linear_solver=Ma27Solver,
        options...,
    )
end

function build_ma57_solver(nlp; options...)
    return MadNLPSolver(
        nlp;
        linear_solver=Ma57Solver,
        options...,
    )
end

function build_sckkt_solver(nlp; options...)
    solver = MadNLP.MadNLPSolver(
        nlp;
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        equality_treatment=MadNLP.RelaxEquality,
        fixed_variable_treatment=MadNLP.RelaxBound,
        dual_initialized=true,
        options...,
    )
end

function build_hckkt_solver(nlp; gamma=1e7, options...)
    solver = MadNLP.MadNLPSolver(
        nlp;
        kkt_system=HybridKKT.HybridCondensedKKTSystem,
        equality_treatment=MadNLP.EnforceEquality,
        fixed_variable_treatment=MadNLP.MakeParameter,
        options...,
    )
    solver.kkt.gamma[] = gamma
    return solver
end

