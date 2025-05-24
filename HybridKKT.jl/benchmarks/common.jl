
using DelimitedFiles
using LinearAlgebra
using SparseArrays
using Pkg.Artifacts

using CUDA
using CUDSS

using MadNLP
using MadNLPHSL
using MadNLPGPU
using MadNLPTests
using HybridKKT
using ExaModels

using NLPModelsIpopt

using HSL_jll

import SuiteSparse: CHOLMOD

if CUDA.has_cuda()
    CUDA.allowscalar(false)
end

function refresh_memory()
    GC.gc(true)
    CUDA.has_cuda() && CUDA.reclaim()
    return
end

function get_ipopt_status(code::Symbol)
    if code == :first_order
        return 1
    elseif code == :acceptable
        return 2
    else
        return 3
    end
end

function solve_ipopt(nlp; gamma=1e7, options...)
    results = ipopt(nlp; options...)
    return (
        status=get_ipopt_status(results.status),
        time_init=0.0,
        total_time=results.elapsed_time,
        time_callbacks=0.0,
        time_linear_solver=0.0,
        iter=results.iter,
        objective=results.objective,
    )
end

function build_hsl_solver(nlp; options...)
    return MadNLPSolver(
        nlp;
        options...,
    )
end
function solve_madnlp_hsl(nlp; options...)
    t_init = CUDA.@elapsed begin
        solver = build_hsl_solver(nlp; options...)
    end
    results = MadNLP.solve!(solver)
    return (
        status=Int(results.status),
        time_init=t_init,
        total_time=solver.cnt.total_time,
        time_callbacks=solver.cnt.eval_function_time,
        time_linear_solver=solver.cnt.linear_solver_time,
        iter=solver.cnt.k,
        objective=results.objective,
    )
end

function build_cudss_solver(nlp; options...)
    return MadNLPSolver(
        nlp;
        linear_solver=MadNLPGPU.CUDSSSolver,
        kkt_system=MadNLP.SparseKKTSystem,
        cudss_algorithm=MadNLP.LU,
        equality_treatment=MadNLP.EnforceEquality,
        fixed_variable_treatment=MadNLP.MakeParameter,
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
function solve_madnlp_sckkt(nlp; options...)
    t_init = CUDA.@elapsed begin
        solver = build_sckkt_solver(nlp; options...)
    end
    results = MadNLP.solve!(solver)
    return (
        status=Int(results.status),
        time_init=t_init,
        total_time=solver.cnt.total_time,
        time_callbacks=solver.cnt.eval_function_time,
        time_linear_solver=solver.cnt.linear_solver_time,
        iter=solver.cnt.k,
        objective=results.objective,
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
function solve_madnlp_hykkt(nlp; options...)
    t_init = CUDA.@elapsed begin
        solver = build_hckkt_solver(nlp; options...)
    end
    results = MadNLP.solve!(solver)
    return (
        status=Int(results.status),
        time_init=t_init,
        total_time=solver.cnt.total_time,
        time_callbacks=solver.cnt.eval_function_time,
        time_linear_solver=solver.cnt.linear_solver_time,
        iter=solver.cnt.k,
        objective=results.objective,
    )
end
