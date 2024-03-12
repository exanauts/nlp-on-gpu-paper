
using Comonicon

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "model.jl"))

const PGLIB_PATH = joinpath(artifact"PGLib_opf", "pglib-opf-23.07")
const RESULTS_DIR = joinpath(@__DIR__, "..", "..", "results", "condensed")

@main function main(; verbose::Bool=false, case="pglib_opf_case2000_goc.m")
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end
    datafile = joinpath(PGLIB_PATH, case)

    print_level = verbose ? MadNLP.INFO : MadNLP.ERROR
    tols = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
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
    results = zeros(length(tols), 8)
    solver = MadNLP.MadNLPSolver(
        nlp;
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        dual_initialized=true,
        equality_treatment=MadNLP.RelaxEquality,
        fixed_variable_treatment=MadNLP.RelaxBound,
        linear_solver=HybridKKT.CHOLMODSolver,
        richardson_tol=1e-12,
        print_level=print_level,
        max_iter=1,
    )
    MadNLP.solve!(solver)
    # Warmup
    for (k, tol) in enumerate(tols)
        solver = MadNLP.MadNLPSolver(
            nlp;
            kkt_system=MadNLP.SparseCondensedKKTSystem,
            dual_initialized=true,
            equality_treatment=MadNLP.RelaxEquality,
            fixed_variable_treatment=MadNLP.RelaxBound,
            linear_solver=HybridKKT.CHOLMODSolver,
            richardson_tol=1e-12,
            print_level=print_level,
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
        tol=1e-5,
        dual_initialized=true,
        equality_treatment=MadNLP.RelaxEquality,
        fixed_variable_treatment=MadNLP.RelaxBound,
        linear_solver=MadNLPGPU.CUDSSSolver,
        cudss_algorithm=MadNLP.CHOLESKY,
        richardson_tol=1e-12,
        print_level=print_level,
        max_iter=1,
    )
    MadNLP.solve!(solver)
    for (k, tol) in enumerate(tols)
        solver = MadNLP.MadNLPSolver(
            nlp_gpu;
            kkt_system=MadNLP.SparseCondensedKKTSystem,
            dual_initialized=true,
            equality_treatment=MadNLP.RelaxEquality,
            fixed_variable_treatment=MadNLP.RelaxBound,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.CHOLESKY,
            richardson_tol=1e-12,
            print_level=print_level,
            max_iter=200,
            tol=tol,
        )
        MadNLP.solve!(solver)

        results[k, 4] = solver.cnt.k
        results[k, 5] = solver.cnt.total_time
    end

    writedlm(joinpath(RESULTS_DIR, "sckkt-stats.txt"), results)
end

