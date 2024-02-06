
using DelimitedFiles
using Comonicon

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "..", "scripts", "opf", "model.jl"))

# Setup #
const RESULTS_DIR = joinpath(@__DIR__, "..", "results", "kkt")

function benchmark_kkt(solver, ntrials)
    MadNLP.solve!(solver)
    refresh_memory()

    t_build, t_factorization, t_backsolve = (0.0, 0.0, 0.0)
    delta_err = 0.0
    for _ in 1:ntrials
        t_build += CUDA.@elapsed begin
            MadNLP.build_kkt!(solver.kkt)
        end
        t_factorization += CUDA.@elapsed begin
            MadNLP.factorize!(solver.kkt.linear_solver)
        end
        t_backsolve += CUDA.@elapsed begin
            MadNLP.solve_refine_wrapper!(solver.d, solver, solver.p, solver._w4)
        end

        dsol = solver.d
        w = solver._w4

        copyto!(MadNLP.full(w), MadNLP.full(solver.p))
        mul!(w, solver.kkt, dsol, -1.0, 1.0)

        wsol = MadNLP.full(w)
        delta_err += norm(wsol, Inf)
    end
    return (
        t_build / ntrials,
        t_factorization / ntrials,
        t_backsolve / ntrials,
        delta_err / ntrials,
    )
end

@main function main(;
    case="pglib_opf_case9241_pegase.m",
    ntrials::Int=3,
    gamma::Float64=1e7,
)
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end

    columns = ["hsl", "sckkt-cpu", "hckkt-cpu", "sckkt-cuda", "hckkt-cuda"]
    results = zeros(5, 4)

    datafile = joinpath(PGLIB_PATH, case)
    nlp = ac_power_model(datafile)

    @info "Benchmark KKT with HSL"
    solver = build_hsl_solver(nlp; max_iter=1, print_level=MadNLP.ERROR)
    results[1, :] .= benchmark_kkt(solver, ntrials)

    @info "Benchmark KKT with SparseCondensedKKTSystem+CHOLMOD"
    solver = build_sckkt_solver(nlp; max_iter=1, print_level=MadNLP.ERROR, linear_solver=HybridKKT.CHOLMODSolver)
    results[2, :] .= benchmark_kkt(solver, ntrials)

    @info "Benchmark KKT with HybridCondensedKKTSystem+CHOLMOD"
    solver = build_hckkt_solver(nlp, gamma; max_iter=1, print_level=MadNLP.ERROR, linear_solver=HybridKKT.CHOLMODSolver)
    results[3, :] .= benchmark_kkt(solver, ntrials)


    nlp_gpu = ac_power_model(datafile; backend=CUDABackend())

    @info "Benchmark KKT with SparseCondensedKKTSystem+cuDSS"
    solver = build_sckkt_solver(nlp_gpu; max_iter=1, print_level=MadNLP.ERROR, linear_solver=MadNLPGPU.CUDSSSolver, cudss_algorithm=MadNLP.CHOLESKY)
    results[4, :] .= benchmark_kkt(solver, ntrials)

    @info "Benchmark KKT with HybridCondensedKKTSystem+cuDSS"
    solver = build_hckkt_solver(nlp_gpu, gamma; max_iter=1, print_level=MadNLP.ERROR, linear_solver=MadNLPGPU.CUDSSSolver, cudss_algorithm=MadNLP.CHOLESKY)
    results[5, :] .= benchmark_kkt(solver, ntrials)

    output_file = joinpath(RESULTS_DIR, "benchmark_kkt.txt")
    writedlm(output_file, [columns results])
end

