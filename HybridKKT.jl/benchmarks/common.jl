
using LinearAlgebra
using CUDA
using MadNLP
using MadNLPHSL
using MadNLPGPU
using HybridKKT

if haskey(ENV, "PGLIB_PATH")
    const PGLIB_PATH = ENV["PGLIB_PATH"]
else
    error("Unable to find path to PGLIB benchmark.\n"*
        "Please set environment variable `PGLIB_PATH` to run benchmark with PowerModels.jl")
end

CUDA.allowscalar(false)

function refresh_memory()
    GC.gc(true)
    CUDA.has_cuda() && CUDA.reclaim()
    return
end

function build_hsl_solver(nlp; options...)
    return MadNLPSolver(
        nlp;
        linear_solver=Ma27Solver,
        options...,
    )
end

function build_sckkt_solver(nlp; options...)
    solver = MadNLP.MadNLPSolver(
        nlp;
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        equality_treatment=MadNLP.RelaxEquality,
        fixed_variable_treatment=MadNLP.RelaxBound,
        options...,
    )
end

function build_hckkt_solver(nlp, gamma; options...)
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

