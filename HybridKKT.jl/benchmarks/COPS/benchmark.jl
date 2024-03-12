
using DelimitedFiles

using Comonicon

using CUDA
using MadNLP
using MadNLPHSL
using MadNLPGPU
using ExaModels
using HybridKKT

using COPSBenchmark

const RESULTS_DIR = joinpath(@__DIR__, "..", "..", "results", "cops")

# Instances
const COPS_INSTANCES_QUICK = [
    (COPSBenchmark.bearing_model, (50, 50), 1e6),
    (COPSBenchmark.camshape_model, (1000,), 1e6), # TODO: result is slightly different
    (COPSBenchmark.elec_model, (50,), 1e6),
    (COPSBenchmark.gasoil_model, (100,), 1e6),
    (COPSBenchmark.marine_model, (100,), 1e6),
    (COPSBenchmark.pinene_model, (100,), 1e5),
    (COPSBenchmark.robot_model, (200,), 1e9),
    (COPSBenchmark.steering_model, (200,), 1e6),
]

const COPS_INSTANCES_MITTELMANN = [
    (COPSBenchmark.bearing_model, (400, 400), 1e6),
    (COPSBenchmark.camshape_model, (6400,), 1e6), # TODO: result is slightly different
    (COPSBenchmark.elec_model, (400,), 1e6),
    (COPSBenchmark.gasoil_model, (3200,), 1e6),
    (COPSBenchmark.marine_model, (1600,), 1e6),
    (COPSBenchmark.pinene_model, (3200,), 1e5),
    (COPSBenchmark.robot_model, (1600,), 1e9),
    (COPSBenchmark.rocket_model, (1600,), 1e9),
    (COPSBenchmark.steering_model, (12800,), 1e10),
]

include(joinpath(@__DIR__, "..", "common.jl"))

function parse_name(cops_instance)
    func, params = cops_instance
    id = split(string(func), '_')[1]
    k = prod(params)
    return "$(id)_$(k)"
end

function benchmark_solver(bench_solver, nlp, ntrials; gamma=1e7, maxit=1000, options...)
    ## Warm-up
    solver = bench_solver(nlp; max_iter=1, options...)
    MadNLP.solve!(solver)

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

function run_benchmark(bench_solver, instances, ntrials; use_gpu=false, options...)
    n, m = length(instances), 6
    results = zeros(n, m)
    for (k, (instance, params, gamma)) in enumerate(instances)
        @info "Benchmark $(parse_name((instance, params)))"
        model = instance(params...)
        nlp = if use_gpu
            ExaModels.ExaModel(model; backend=CUDABackend())
        else
            ExaModels.ExaModel(model)
        end
        results[k, :] .= benchmark_solver(bench_solver, nlp, ntrials; gamma=gamma, options...)
    end
    return results
end

@main function main(;
    solver="all",
    verbose::Bool=false,
    quick::Bool=false,
    tol::Float64=1e-4,
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
    instances = if quick
        COPS_INSTANCES_QUICK
    else
        COPS_INSTANCES_MITTELMANN
    end
    index = [parse_name(it) for it in instances]

    if solver == "all" || solver == "ma27"
        @info "[CPU] Benchmark SparseKKTSystem+ma27"
        results = run_benchmark(
            build_ma27_solver,
            instances,
            ntrials;
            maxit=max_iter,
            tol=tol,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hsl-ma27.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "ma57"
        @info "[CPU] Benchmark SparseKKTSystem+ma57"
        results = run_benchmark(
            build_ma57_solver,
            instances,
            ntrials;
            maxit=max_iter,
            tol=tol,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hsl-ma57.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "sckkt-cpu"
        @info "[CPU] Benchmark SparseCondensedKKTSystem+CHOLMOD"
        results = run_benchmark(
            build_sckkt_solver,
            instances,
            ntrials;
            maxit=max_iter,
            tol=tol,
            linear_solver=HybridKKT.CHOLMODSolver,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-sckkt-cholmod.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "hckkt-cpu"
        @info "[CPU] Benchmark HybridCondensedKKTSystem+CHOLMOD"
        results = run_benchmark(
            build_hckkt_solver,
            instances,
            ntrials;
            maxit=max_iter,
            tol=tol,
            linear_solver=HybridKKT.CHOLMODSolver,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hckkt-cholmod.csv")
        writedlm(output_file, [index results])
    end

    if (solver == "all" || solver == "sckkt-cuda") && CUDA.has_cuda()
        @info "[CUDA] Benchmark SparseCondensedKKTSystem+CUDSS"
        results = run_benchmark(
            build_sckkt_solver,
            instances,
            ntrials;
            maxit=max_iter,
            use_gpu=true,
            tol=tol,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.CHOLESKY,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-sckkt-cudss-cholesky.csv")
        writedlm(output_file, [index results])
    end

    if (solver == "all" || solver == "hckkt-cuda") && CUDA.has_cuda()
        @info "[CUDA] Benchmark HybridCondensedKKTSystem+CUDSS"
        results = run_benchmark(
            build_hckkt_solver,
            instances,
            ntrials;
            maxit=max_iter,
            use_gpu=true,
            tol=tol,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.BUNCHKAUFMAN,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hckkt-cudss-ldl.csv")
        writedlm(output_file, [index results])
    end
end

