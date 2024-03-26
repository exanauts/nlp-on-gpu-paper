
using Comonicon

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "model.jl"))

# Setup #
const PGLIB_PATH = joinpath(artifact"PGLib_opf", "pglib-opf-23.07")
const RESULTS_DIR = joinpath(@__DIR__, "..", "..", "results", "kkt")

function benchmark_cudss(K, ntrials; structure="SPD")
    n = size(K, 1)
    full = CUSPARSE.CuSparseMatrixCSR(K)
    view = 'F'

    matrix = CUDSS.CudssMatrix(full, structure, view)
    config = CUDSS.CudssConfig()
    data = CUDSS.CudssData()
    x_gpu = CUDA.zeros(Float64, n)
    b_gpu = CUDA.zeros(Float64, n)
    w_gpu = CUDA.zeros(Float64, n)

    (time_analysis, time_factorization, time_backsolve, accuracy) = (0.0, 0.0, 0.0, 0.0)
    for _ in 1:ntrials
        solver = CUDSS.CudssSolver(matrix, config, data)
        time_analysis += CUDA.@elapsed CUDA.@sync begin
            CUDSS.cudss("analysis", solver, x_gpu, b_gpu)
        end
        time_factorization += CUDA.@elapsed CUDA.@sync begin
            CUDSS.cudss("factorization", solver, x_gpu, b_gpu)
        end
        fill!(b_gpu, 1.0)
        time_backsolve += CUDA.@elapsed CUDA.@sync begin
            CUDSS.cudss("solve", solver, x_gpu, b_gpu)
        end
        w_gpu .= b_gpu
        mul!(w_gpu, full, x_gpu, -1.0, 1.0)

        accuracy += norm(w_gpu, Inf)
    end
    return (
        n,
        nnz(K),
        time_analysis / ntrials,
        time_factorization / ntrials,
        time_backsolve / ntrials,
        accuracy / ntrials,
    )
end

function benchmark_cholmod(K, ntrials)
    n = size(K, 1)
    A = CHOLMOD.Sparse(K)
    (time_analysis, time_factorization, time_backsolve, accuracy) = (0.0, 0.0, 0.0, 0.0)

    for _ in 1:ntrials
        time_analysis += @elapsed begin
            solver = CHOLMOD.symbolic(A)
        end
        time_factorization += @elapsed begin
            CHOLMOD.cholesky!(solver, K; check=false)
        end

        b = ones(n)
        x = zeros(n)
        w = zeros(n)
        time_backsolve += @elapsed begin
            B = CHOLMOD.Dense(b)
            X = CHOLMOD.solve(CHOLMOD.CHOLMOD_A, solver, B)
            copyto!(x, X)
        end
        w .= b
        mul!(w, K, x, -1.0, 1.0)
        accuracy += norm(w, Inf)
    end

    return (
        n,
        nnz(K),
        time_analysis / ntrials,
        time_factorization / ntrials,
        time_backsolve / ntrials,
        accuracy / ntrials,
    )
end

@main function main(;
    case="pglib_opf_case78484_epigrids.m",
    ntrials::Int=10,
    gamma::Float64=1e7
)
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end

    columns = ["cholmod", "cudss-cholesky", "cudss-lu", "cudss-ldl"]
    results = zeros(4, 6)

    datafile = joinpath(PGLIB_PATH, case)
    nlp = ac_power_model(datafile)
    display(nlp)

    solver = build_hckkt_solver(nlp; gamma=gamma, max_iter=1, print_level=MadNLP.ERROR)
    MadNLP.solve!(solver)

    # Take condensed KKT system at iteration 1
    K = solver.kkt.aug_com
    K = 0.5 .* (K + K')
    # K = K + K' - Diagonal(K) ?
    m, n = size(K)
    nz = nnz(K)

    println("case: $case")
    println("Size of K: $m × $n")
    println("nnz(K): $nz")

    results[1, :] .= benchmark_cholmod(K, ntrials)
    results[2, :] .= benchmark_cudss(K, ntrials; structure="SPD")
    results[3, :] .= benchmark_cudss(K, ntrials; structure="S")
    results[4, :] .= benchmark_cudss(K, ntrials; structure="G")

    output_file = joinpath(RESULTS_DIR, "benchmark_cholesky.txt")
    writedlm(output_file, [columns results])
end
