
using DelimitedFiles

using Comonicon

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

const RESULTS_DIR = joinpath(@__DIR__, "..", "..", "results", "opf")

include(joinpath(@__DIR__, "model.jl"))

CUDA.allowscalar(false)

N_COLUMNS = Dict{Symbol, Int}(
    :benchmark_hsl => 6,
    :benchmark_sparse_condensed => 6,
    :benchmark_hybrid => 10,
)

FULL_BENCHMARK = [
    "pglib_opf_case10000_goc.m",
    "pglib_opf_case10192_epigrids.m",
    "pglib_opf_case10480_goc.m",
    "pglib_opf_case1354_pegase.m",
    "pglib_opf_case13659_pegase.m",
    "pglib_opf_case179_goc.m",
    "pglib_opf_case19402_goc.m",
    "pglib_opf_case2000_goc.m",
    "pglib_opf_case20758_epigrids.m",
    "pglib_opf_case2312_goc.m",
    # "pglib_opf_case24464_goc.m",
    "pglib_opf_case2742_goc.m",
    "pglib_opf_case2869_pegase.m",
    "pglib_opf_case30000_goc.m",
    "pglib_opf_case3022_goc.m",
    "pglib_opf_case3970_goc.m",
    "pglib_opf_case4020_goc.m",
    "pglib_opf_case4601_goc.m",
    "pglib_opf_case4619_goc.m",
    "pglib_opf_case4837_goc.m",
    "pglib_opf_case4917_goc.m",
    "pglib_opf_case500_goc.m",
    "pglib_opf_case5658_epigrids.m",
    "pglib_opf_case7336_epigrids.m",
    "pglib_opf_case78484_epigrids.m",
    "pglib_opf_case793_goc.m",
    "pglib_opf_case8387_pegase.m",
    "pglib_opf_case89_pegase.m",
    "pglib_opf_case9241_pegase.m",
    "pglib_opf_case9591_goc.m",
]


function refresh_memory()
    GC.gc(true)
    CUDA.has_cuda() && CUDA.reclaim()
    return
end

function benchmark_hsl(nlp, ntrials; options...)
    ## Warm-up
    solver = MadNLP.MadNLPSolver(nlp; linear_solver=Ma27Solver, max_iter=1, options...)
    MadNLP.solve!(solver)

    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    n_it, obj = 0, 0.0
    status = 0
    ## Benchmark
    for _ in 1:ntrials
        solver = MadNLP.MadNLPSolver(nlp; linear_solver=Ma27Solver, options...)
        results = MadNLP.solve!(solver)

        status += Int(results.status)
        t_total += solver.cnt.total_time
        t_callbacks += solver.cnt.eval_function_time
        t_linear_solver += solver.cnt.linear_solver_time
        n_it += solver.cnt.k
        obj += solver.obj_val
        # Clean memory
        refresh_memory()
    end

    return (
        status / ntrials,
        n_it / ntrials,
        obj / ntrials,
        t_total / ntrials,
        t_callbacks / ntrials,
        t_linear_solver / ntrials,
    )
end

function benchmark_sparse_condensed(nlp, ntrials; options...)
    ## Warm-up
    solver = MadNLP.MadNLPSolver(
        nlp;
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        dual_initialized=true,
        equality_treatment=MadNLP.RelaxEquality,
        fixed_variable_treatment=MadNLP.RelaxBound,
        max_iter=1,
        options...,
    )
    MadNLP.solve!(solver)

    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    n_it, obj = 0, 0.0
    status = 0
    ## Benchmark
    for _ in 1:ntrials
        solver = MadNLP.MadNLPSolver(
            nlp;
            kkt_system=MadNLP.SparseCondensedKKTSystem,
            dual_initialized=true,
            equality_treatment=MadNLP.RelaxEquality,
            fixed_variable_treatment=MadNLP.RelaxBound,
            options...,
        )
        results = MadNLP.solve!(solver)

        status += Int(results.status)
        t_total += solver.cnt.total_time
        t_callbacks += solver.cnt.eval_function_time
        t_linear_solver += solver.cnt.linear_solver_time
        n_it += solver.cnt.k
        obj += solver.obj_val
        # Clean memory
        refresh_memory()
    end

    return (
        status / ntrials,
        n_it / ntrials,
        obj / ntrials,
        t_total / ntrials,
        t_callbacks / ntrials,
        t_linear_solver / ntrials,
    )
end

function benchmark_hybrid(nlp, ntrials; gamma=1e5, options...)
    solver = MadNLP.MadNLPSolver(
        nlp;
        kkt_system=HybridKKT.HybridCondensedKKTSystem,
        equality_treatment=MadNLP.EnforceEquality,
        fixed_variable_treatment=MadNLP.MakeParameter,
        max_iter=1,
        options...,
    )
    MadNLP.solve!(solver)

    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    t_backsolve, t_condensation, t_cg = (0.0, 0.0, 0.0)
    n_it, obj = 0, 0.0
    cg_iters = 0.0
    status = 0
    ## Benchmark
    for _ in 1:ntrials
        solver = MadNLP.MadNLPSolver(
            nlp;
            kkt_system=HybridKKT.HybridCondensedKKTSystem,
            equality_treatment=MadNLP.EnforceEquality,
            fixed_variable_treatment=MadNLP.MakeParameter,
            options...,
        )
        solver.kkt.gamma[] = gamma
        results = MadNLP.solve!(solver)

        status += Int(results.status)
        t_total += solver.cnt.total_time
        t_callbacks += solver.cnt.eval_function_time
        t_linear_solver += solver.cnt.linear_solver_time
        n_it += solver.cnt.k
        obj += solver.obj_val
        cg_iters += sum(solver.kkt.etc[:cg_iters]) / n_it
        t_backsolve += solver.kkt.etc[:time_backsolve]
        t_cg += solver.kkt.etc[:time_cg]
        t_condensation += solver.kkt.etc[:time_condensation]
        # Clean memory
        refresh_memory()
    end

    return (
        status / ntrials,
        n_it / ntrials,
        obj / ntrials,
        t_total / ntrials,
        t_callbacks / ntrials,
        t_linear_solver / ntrials,
        cg_iters / ntrials,
        t_backsolve / ntrials,
        t_cg / ntrials,
        t_condensation / ntrials,
    )
end

function run_benchmark(benchmark, cases, ntrials; use_gpu=false, options...)
    n = length(cases)
    m = N_COLUMNS[Symbol(benchmark)]
    results = zeros(n, m)
    for (k, case) in enumerate(cases)
        @info "Benchmark $(case)"
        datafile = joinpath(PGLIB_PATH, case)
        nlp = if use_gpu
            ac_power_model(datafile; backend=CUDABackend())
        else
            ac_power_model(datafile)
        end
        results[k, :] .= benchmark(nlp, ntrials; options...)
    end
    return results
end

@main function main(;
    solver="all",
    verbose::Bool=false,
    quick::Bool=false,
    tol::Float64=1e-4,
    gamma::Float64=1e7,
    ntrials::Int=1,
)
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end
    if solver ∈ ("all", "sckkt-gpu", "hckkt-gpu") && !CUDA.has_cuda()
        @info("CUDA is not available on this machine.")
    end

    flag = quick ? "short" : "full"
    print_level = verbose ? MadNLP.INFO : MadNLP.ERROR
    gamma_ = Int(log10(gamma))

    # if quick
    cases = if quick
        filter!(e->(occursin("pglib_opf_case",e) && occursin("ieee",e)),readdir(PGLIB_PATH))
    else
        FULL_BENCHMARK
    end

    if solver == "all" || solver == "hsl"
        @info "[CPU] Benchmark SparseKKTSystem+HSL"
        results = run_benchmark(benchmark_hsl, cases, ntrials; tol=tol, print_level=print_level)
        output_file = joinpath(RESULTS_DIR, "pglib-$(flag)-madnlp-hsl-ma27.csv")
        writedlm(output_file, [cases results])
    end

    if solver == "all" || solver == "sckkt-cpu"
        @info "[CPU] Benchmark SparseCondensedKKTSystem+CHOLMOD"
        results = run_benchmark(
            benchmark_sparse_condensed,
            cases,
            ntrials;
            tol=tol,
            linear_solver=HybridKKT.CHOLMODSolver,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "pglib-$(flag)-madnlp-sckkt-cholmod.csv")
        writedlm(output_file, [cases results])
    end

    if solver == "all" || solver == "hckkt-cpu"
        @info "[CPU] Benchmark HybridCondensedKKTSystem+CHOLMOD"
        results = run_benchmark(
            benchmark_hybrid,
            cases,
            ntrials;
            gamma=gamma,
            tol=tol,
            linear_solver=HybridKKT.CHOLMODSolver,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "pglib-$(flag)-madnlp-hckkt-cholmod-$(gamma_).csv")
        writedlm(output_file, [cases results])
    end

    if (solver == "all" || solver == "sckkt-cuda") && CUDA.has_cuda()
        @info "[CUDA] Benchmark SparseCondensedKKTSystem+CUDSS"
        results = run_benchmark(
            benchmark_sparse_condensed,
            cases,
            ntrials;
            use_gpu=true,
            tol=tol,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.CHOLESKY,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "pglib-$(flag)-madnlp-sckkt-cudss-cholesky.csv")
        writedlm(output_file, [cases results])
    end

    if (solver == "all" || solver == "hckkt-cuda") && CUDA.has_cuda()
        @info "[CUDA] Benchmark HybridCondensedKKTSystem+CUDSS"
        results = run_benchmark(
            benchmark_hybrid,
            cases,
            ntrials;
            gamma=gamma,
            use_gpu=true,
            tol=tol,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.BUNCHKAUFMAN,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "pglib-$(flag)-madnlp-hckkt-cudss-ldl-$(gamma_).csv")
        writedlm(output_file, [cases results])
    end
end
