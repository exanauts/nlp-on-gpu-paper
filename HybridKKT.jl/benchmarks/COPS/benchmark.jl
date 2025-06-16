
import Comonicon
using COPSBenchmark

include(joinpath(@__DIR__, "..", "common.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "..", "..", "results", "cops")

# Instances
const COPS_INSTANCES_QUICK = [
    (COPSBenchmark.bearing_model, (50, 50), 1e6),
    (COPSBenchmark.camshape_model, (1000,), 1e6),
    (COPSBenchmark.elec_model, (50,), 1e6),
    (COPSBenchmark.gasoil_model, (100,), 1e6),
    (COPSBenchmark.marine_model, (100,), 1e6),
    (COPSBenchmark.pinene_model, (100,), 1e5),
    (COPSBenchmark.robot_model, (200,), 1e9),
    (COPSBenchmark.steering_model, (200,), 1e6),
]

const COPS_INSTANCES_FULL = [
    # Mittelmann instances
    (COPSBenchmark.bearing_model, (400, 400), 1e6),
    (COPSBenchmark.camshape_model, (6400,), 1e6),
    (COPSBenchmark.elec_model, (400,), 1e6),
    (COPSBenchmark.gasoil_model, (3200,), 1e6),
    (COPSBenchmark.marine_model, (1600,), 1e6),
    (COPSBenchmark.pinene_model, (3200,), 1e5),
    (COPSBenchmark.robot_model, (1600,), 1e9),
    (COPSBenchmark.rocket_model, (12800,), 1e9),
    (COPSBenchmark.steering_model, (12800,), 1e10),
    # Large-scale instances
    (COPSBenchmark.bearing_model, (800, 800), 1e6),
    (COPSBenchmark.camshape_model, (12800,), 1e6),
    (COPSBenchmark.elec_model, (800,), 1e6),
    (COPSBenchmark.gasoil_model, (12800,), 1e6),
    (COPSBenchmark.marine_model, (12800,), 1e6),
    (COPSBenchmark.pinene_model, (12800,), 1e5),
    (COPSBenchmark.robot_model, (12800,), 1e9),
    (COPSBenchmark.rocket_model, (51200,), 1e11),
    (COPSBenchmark.steering_model, (51200,), 1e9),
]

function parse_name(cops_instance)
    func, params = cops_instance
    id = split(string(func), '_')[1]
    k = prod(params)
    return "$(id)_$(k)"
end

function benchmark_solver(bench_solver, nlp, ntrials; gamma=1e7, maxit=1000, options...)
    ## Warm-up
    results = bench_solver(nlp; max_iter=1, options...)

    t_init, t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0, 0.0)
    n_it, obj = 0, 0.0
    status = 0
    ## Benchmark
    for _ in 1:ntrials
        results = bench_solver(nlp; gamma=gamma, max_iter=maxit, options...)

        status += results.status
        t_total += results.total_time
        t_callbacks += results.time_callbacks
        t_linear_solver += results.time_linear_solver
        n_it += results.iter
        obj += results.objective
        t_init += results.time_init
        # Clean memory
        refresh_memory()
    end

    return (
        status / ntrials,
        n_it / ntrials,
        obj / ntrials,
        t_total / ntrials,
        t_init / ntrials,
        t_callbacks / ntrials,
        t_linear_solver / ntrials,
    )
end

function run_benchmark(bench_solver, instances, ntrials; use_gpu=false, options...)
    n, m = length(instances), 7
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

Comonicon.@main function main(;
    solver="all",
    verbose::Bool=false,
    quick::Bool=false,
    tol::Float64=1e-6,
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
        COPS_INSTANCES_FULL
    end
    index = [parse_name(it) for it in instances]

    if solver == "all" || solver == "ipopt"
        @info "[CPU] Benchmark Ipopt+ma57"
        results = run_benchmark(
            solve_ipopt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=1800.0,
            hsllib=HSL_jll.libhsl_path,
            linear_solver="ma57",
            tol=tol,
            print_level=0,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-ipopt-hsl-ma57.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "ma27"
        @info "[CPU] Benchmark SparseKKTSystem+ma27"
        results = run_benchmark(
            solve_madnlp_hsl,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=1800.0,
            linear_solver=Ma27Solver,
            tol=tol,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hsl-ma27.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "ma57"
        @info "[CPU] Benchmark SparseKKTSystem+ma57"
        results = run_benchmark(
            solve_madnlp_hsl,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=1800.0,
            linear_solver=Ma57Solver,
            tol=tol,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hsl-ma57.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "ma86"
        @info "[CPU] Benchmark SparseKKTSystem+ma86"
        results = run_benchmark(
            solve_madnlp_hsl,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=1800.0,
            linear_solver=Ma86Solver,
            tol=tol,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hsl-ma86.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "sckkt-cpu"
        @info "[CPU] Benchmark SparseCondensedKKTSystem+CHOLMOD"
        results = run_benchmark(
            solve_madnlp_sckkt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=1800.0,
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
            solve_madnlp_hykkt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=1800.0,
            tol=tol,
            linear_solver=HybridKKT.CHOLMODSolver,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hckkt-cholmod.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "hckkt-ma86"
        @info "[CPU] Benchmark HybridCondensedKKTSystem+ma86"
        BLAS.set_num_threads(1)
        results = run_benchmark(
            solve_madnlp_hykkt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=1800.0,
            ma86_num_threads=8,
            tol=tol,
            linear_solver=Ma86Solver,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hckkt-ma86.csv")
        writedlm(output_file, [index results])
    end

    if (solver == "all" || solver == "sckkt-cuda") && CUDA.has_cuda()
        @info "[CUDA] Benchmark SparseCondensedKKTSystem+CUDSS"
        results = run_benchmark(
            solve_madnlp_sckkt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=1800.0,
            use_gpu=true,
            tol=tol,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-sckkt-cudss-ldl.csv")
        writedlm(output_file, [index results])
    end

    if (solver == "all" || solver == "hckkt-cuda") && CUDA.has_cuda()
        @info "[CUDA] Benchmark HybridCondensedKKTSystem+CUDSS"
        results = run_benchmark(
            solve_madnlp_hykkt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=1800.0,
            use_gpu=true,
            tol=tol,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-madnlp-hckkt-cudss-ldl.csv")
        writedlm(output_file, [index results])
    end
end

