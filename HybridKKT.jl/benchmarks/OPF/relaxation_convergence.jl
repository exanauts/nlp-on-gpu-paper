
using DelimitedFiles
using LinearAlgebra
using Comonicon

using MadNLP
using MadNLPHSL
using MadNLPGPU

using CUDA

using HybridKKT

if !@isdefined ac_power_model
    include(joinpath(@__DIR__, "model.jl"))
end

if haskey(ENV, "PGLIB_PATH")
    const PGLIB_PATH = ENV["PGLIB_PATH"]
else
    error("Unable to find path to PGLIB benchmark.\n"*
        "Please set environment variable `PGLIB_PATH` to run benchmark with PowerModels.jl")
end

const RESULTS_DIR = joinpath(@__DIR__, "..", "results", "condensed")

@main function main(; case="pglib_opf_case9241_pegase.m")
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end
    datafile = joinpath(PGLIB_PATH, case)
    nlp = ac_power_model(datafile)

    # Solve problem with HSL with good accuracy
    solver_ref = MadNLPSolver(
        nlp;
        linear_solver=Ma27Solver,
        max_iter=200,
        nlp_scaling=true,
        tol=1e-8,
    )
    results_ref = MadNLP.solve!(solver_ref)
    obj_ref = results_ref.objective
    x0 = results_ref.solution
    y0 = results_ref.multipliers

    # Solve problem with SparseCondensedKKTSystem with decreasing tolerance.

    #=
        CPU
    =#
    results = zeros(4, 8)
    solver = MadNLP.MadNLPSolver(
        nlp;
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        equality_treatment=MadNLP.RelaxEquality,
        fixed_variable_treatment=MadNLP.RelaxBound,
        linear_solver=HybridKKT.CHOLMODSolver,
        richardson_tol=1e-12,
        print_level=MadNLP.INFO,
        max_iter=1,
    )
    MadNLP.solve!(solver)
    # Warmup
    for (k, tol) in enumerate([1e-2, 1e-3, 1e-4, 1e-5])
        solver = MadNLP.MadNLPSolver(
            nlp;
            kkt_system=MadNLP.SparseCondensedKKTSystem,
            equality_treatment=MadNLP.RelaxEquality,
            fixed_variable_treatment=MadNLP.RelaxBound,
            linear_solver=HybridKKT.CHOLMODSolver,
            richardson_tol=1e-12,
            print_level=MadNLP.INFO,
            max_iter=200,
            tol=tol,
        )
        res= MadNLP.solve!(solver)

        x1 = res.solution
        y1 = res.multipliers
        results[k, 1] = tol
        results[k, 2] = res.iter
        results[k, 3] = solver.cnt.total_time
        results[k, 6] = abs(res.objective - obj_ref) / obj_ref
        results[k, 7] = norm(x1 .- x0, Inf) / norm(x0, Inf)
        results[k, 8] = norm(y1 .- y0, Inf) / norm(y0, Inf)
    end

    #=
        CUDA
    =#
    nlp_gpu = ac_power_model(datafile; backend=CUDABackend())
    # Warm-up
    solver = MadNLP.MadNLPSolver(
        nlp_gpu;
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        equality_treatment=MadNLP.RelaxEquality,
        fixed_variable_treatment=MadNLP.RelaxBound,
        linear_solver=MadNLPGPU.CUDSSSolver,
        cudss_algorithm=MadNLP.CHOLESKY,
        richardson_tol=1e-12,
        print_level=MadNLP.INFO,
        max_iter=1,
    )
    MadNLP.solve!(solver)
    for (k, tol) in enumerate([1e-2, 1e-3, 1e-4, 1e-5])
        solver = MadNLP.MadNLPSolver(
            nlp_gpu;
            kkt_system=MadNLP.SparseCondensedKKTSystem,
            equality_treatment=MadNLP.RelaxEquality,
            fixed_variable_treatment=MadNLP.RelaxBound,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.CHOLESKY,
            richardson_tol=1e-12,
            print_level=MadNLP.INFO,
            max_iter=200,
            tol=tol,
        )
        MadNLP.solve!(solver)

        results[k, 4] = solver.cnt.k
        results[k, 5] = solver.cnt.total_time
    end

    writedlm(joinpath(RESULTS_DIR, "sckkt-stats.txt"), results)
end

