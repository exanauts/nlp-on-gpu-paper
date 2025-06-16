
import Comonicon
using CUTEst

include(joinpath(@__DIR__, "..", "common.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "..", "..", "results", "cutest")

# Instances
const CUTEST_INSTANCES_QUICK = CUTEst.select_sif_problems(; min_var=100, max_var=200, min_con=1)
const CUTEST_INSTANCES_FULL = CUTEst.select_sif_problems(; min_var=1000, min_con=1)

EXCLUDE = [
    # MadNLP running into error
    # Ipopt running into error
    "EG3", # lfact blows up
    # Problems that are hopelessly large
    "TAX213322",
    "TAXR213322",
    "TAX53322",
    "TAXR53322",
    "YATP1LS",
    "YATP2LS",
    "YATP1CLS",
    "YATP2CLS",
    "CYCLOOCT",
    "CYCLOOCF",
    "LIPPERT1",
    "GAUSSELM",
    "BA-L52LS",
    "BA-L73LS",
    "BA-L21LS",
]


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
    for (k, instance) in enumerate(instances)
        @info "Benchmark $(instance)"
        nlp_ = CUTEst.CUTEstModel(instance; decode=false)
        nlp = if use_gpu
            MadNLPTests.SparseWrapperModel(CuArray, nlp_)
        else
            nlp_
        end
        try
            results[k, :] .= benchmark_solver(bench_solver, nlp, ntrials; options...)
        catch ex
            println("Fail to solve $(instance): $(ex)")
            results[k, 1] = -1.0
        end
        finalize(nlp_)
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
        CUTEST_INSTANCES_QUICK
    else
        CUTEST_INSTANCES_FULL
    end

    filter!(e->!(e in EXCLUDE), instances)
    instances = instances

    index = instances

    if solver == "all" || solver == "ipopt"
        @info "[CPU] Benchmark Ipopt+ma57"
        results = run_benchmark(
            solve_ipopt,
            instances,
            ntrials;
            max_iter=max_iter,
            max_wall_time=900.0,
            hsllib=HSL_jll.libhsl_path,
            linear_solver="ma57",
            tol=tol,
            print_level=0,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-ipopt-hsl-ma57.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "knitro"
        @info "[CPU] Benchmark Knitro+ma57"
        results = run_benchmark(
            solve_knitro,
            instances,
            ntrials;
            maxit=max_iter,
            maxtime=900.0,
            opttol=tol,
            outlev=0,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-knitro-hsl-ma57.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "ma27"
        @info "[CPU] Benchmark SparseKKTSystem+ma27"
        results = run_benchmark(
            solve_madnlp_hsl,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=900.0,
            linear_solver=Ma27Solver,
            tol=tol,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-hsl-ma27.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "ma57"
        @info "[CPU] Benchmark SparseKKTSystem+ma57"
        results = run_benchmark(
            solve_madnlp_hsl,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=900.0,
            linear_solver=Ma57Solver,
            tol=tol,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-hsl-ma57.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "ma86"
        @info "[CPU] Benchmark SparseKKTSystem+ma86"
        results = run_benchmark(
            solve_madnlp_hsl,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=900.0,
            linear_solver=Ma86Solver,
            tol=tol,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-hsl-ma86.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "pardiso"
        @info "[CPU] Benchmark SparseKKTSystem+pardiso"
        results = run_benchmark(
            solve_madnlp_hsl,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=900.0,
            linear_solver=PardisoSolver,
            tol=tol,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-pardiso.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "sckkt-cpu"
        @info "[CPU] Benchmark SparseCondensedKKTSystem+CHOLMOD"
        results = run_benchmark(
            solve_madnlp_sckkt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=900.0,
            tol=tol,
            linear_solver=HybridKKT.CHOLMODSolver,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-sckkt-cholmod.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "sckkt-ma86"
        @info "[CPU] Benchmark SparseCondensedKKTSystem+Ma86"
        BLAS.set_num_threads(1)
        results = run_benchmark(
            solve_madnlp_sckkt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=900.0,
            tol=tol,
            linear_solver=Ma86Solver,
            ma86_num_threads=8,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-sckkt-ma86.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "sckkt-pardiso"
        @info "[CPU] Benchmark SparseCondensedKKTSystem+Pardiso"
        results = run_benchmark(
            solve_madnlp_sckkt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=900.0,
            tol=tol,
            linear_solver=PardisoSolver,
            pardiso_algorithm=MadNLP.CHOLESKY,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-sckkt-pardiso.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "hckkt-cpu"
        @info "[CPU] Benchmark HybridCondensedKKTSystem+CHOLMOD"
        results = run_benchmark(
            solve_madnlp_hykkt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=900.0,
            tol=tol,
            linear_solver=HybridKKT.CHOLMODSolver,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-hckkt-cholmod.csv")
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
            max_wall_time=900.0,
            ma86_num_threads=8,
            tol=tol,
            linear_solver=Ma86Solver,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-hckkt-ma86.csv")
        writedlm(output_file, [index results])
    end

    if solver == "all" || solver == "hckkt-pardiso"
        @info "[CPU] Benchmark HybridCondensedKKTSystem+Pardiso"
        BLAS.set_num_threads(1)
        results = run_benchmark(
            solve_madnlp_hykkt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=900.0,
            linear_solver=PardisoSolver,
            pardiso_algorithm=MadNLP.CHOLESKY,
            tol=tol,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-hckkt-pardiso.csv")
        writedlm(output_file, [index results])
    end

    if (solver == "all" || solver == "sckkt-cuda") && CUDA.has_cuda()
        @info "[CUDA] Benchmark SparseCondensedKKTSystem+CUDSS"
        results = run_benchmark(
            solve_madnlp_sckkt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=900.0,
            use_gpu=true,
            tol=tol,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-sckkt-cudss-ldl.csv")
        writedlm(output_file, [index results])
    end

    if (solver == "all" || solver == "hckkt-cuda") && CUDA.has_cuda()
        @info "[CUDA] Benchmark HybridCondensedKKTSystem+CUDSS"
        results = run_benchmark(
            solve_madnlp_hykkt,
            instances,
            ntrials;
            maxit=max_iter,
            max_wall_time=900.0,
            use_gpu=true,
            tol=tol,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
            print_level=print_level,
        )
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-hckkt-cudss-ldl.csv")
        writedlm(output_file, [index results])
    end
end

