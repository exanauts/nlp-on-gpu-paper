
using Comonicon
using COPSBenchmark

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "distillation.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "..", "..", "results", "cdc")

function benchmark_solver(bench_solver, nlp, ntrials; gamma=1e7, maxit=1000, options...)
    ## Warm-up
    solver = bench_solver(nlp; max_iter=1, options...)
    MadNLP.solve!(solver)
    refresh_memory()

    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    n_it, obj = 0, 0.0
    status = 0
    ## Benchmark
    for _ in 1:ntrials
        solver = bench_solver(nlp; gamma=gamma, max_iter=maxit, options...)
        results = MadNLP.solve!(solver)

        status += Int(results.status)
        t_total += solver.cnt.total_time
        t_callbacks += solver.cnt.eval_function_time
        t_linear_solver += solver.cnt.linear_solver_time
        n_it += solver.cnt.k
        obj += results.objective
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

function run_benchmark(bench_solver, ntrials; gamma=1e7, use_gpu=false, options...)
    depth = [100, 500, 1000, 5000, 10000, 20000, 50000]
    n, m = length(depth), 6
    results = zeros(n, m)
    for (k, n) in enumerate(depth)
        @info "Benchmark $(n)"
        model = distillation_column_model(n)
        nlp = if use_gpu
            distillation_column_model(n; backend=CUDABackend())
        else
            distillation_column_model(n)
        end
        results[k, :] .= benchmark_solver(bench_solver, nlp, ntrials; gamma=gamma, options...)
    end
    return [depth results]
end

@main function main(;
    solver="all",
    verbose::Bool=false,
    quick::Bool=false,
    tol::Float64=1e-6,
    gamma=1e7,
    ntrials::Int=1,
    max_iter::Int=500,
)
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end
    if solver ∈ ("all", "sckkt-gpu", "hckkt-gpu") && !CUDA.has_cuda()
        @info("CUDA is not available on this machine.")
    end

    flag = quick ? "short" : "full"
    print_level = verbose ? MadNLP.INFO : MadNLP.ERROR

    # if quick

    if solver == "all" || solver == "ma27"
        @info "[CPU] Benchmark SparseKKTSystem+ma27"
        results = run_benchmark(
            build_ma27_solver,
            ntrials;
            maxit=max_iter,
            tol=tol,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hsl-ma27.csv")
        writedlm(output_file, results)
    end

    if solver == "all" || solver == "ma57"
        @info "[CPU] Benchmark SparseKKTSystem+ma57"
        results = run_benchmark(
            build_ma57_solver,
            ntrials;
            maxit=max_iter,
            tol=tol,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hsl-ma57.csv")
        writedlm(output_file, results)
    end

    if solver == "all" || solver == "sckkt-cpu"
        @info "[CPU] Benchmark SparseCondensedKKTSystem+CHOLMOD"
        results = run_benchmark(
            build_sckkt_solver,
            ntrials;
            maxit=max_iter,
            tol=tol,
            linear_solver=HybridKKT.CHOLMODSolver,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-sckkt-cholmod.csv")
        writedlm(output_file, results)
    end

    if solver == "all" || solver == "hckkt-cpu"
        @info "[CPU] Benchmark HybridCondensedKKTSystem+CHOLMOD"
        results = run_benchmark(
            build_hckkt_solver,
            ntrials;
            maxit=max_iter,
            gamma=gamma,
            tol=tol,
            linear_solver=HybridKKT.CHOLMODSolver,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hckkt-cholmod.csv")
        writedlm(output_file, results)
    end

    if (solver == "all" || solver == "sckkt-cuda") && CUDA.has_cuda()
        @info "[CUDA] Benchmark SparseCondensedKKTSystem+CUDSS"
        results = run_benchmark(
            build_sckkt_solver,
            ntrials;
            maxit=max_iter,
            use_gpu=true,
            tol=tol,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.CHOLESKY,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-sckkt-cudss-cholesky.csv")
        writedlm(output_file, results)
    end

    if (solver == "all" || solver == "hckkt-cuda") && CUDA.has_cuda()
        @info "[CUDA] Benchmark HybridCondensedKKTSystem+CUDSS"
        results = run_benchmark(
            build_hckkt_solver,
            ntrials;
            maxit=max_iter,
            gamma=gamma,
            use_gpu=true,
            tol=tol,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.BUNCHKAUFMAN,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hckkt-cudss-ldl.csv")
        writedlm(output_file, results)
    end
end

