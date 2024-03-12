
using Comonicon

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "model.jl"))

const PGLIB_PATH = joinpath(artifact"PGLib_opf", "pglib-opf-23.07")
const RESULTS_DIR = joinpath(@__DIR__, "..", "..", "results", "hybrid")

function solve_hybrid(nlp, gamma; options...)
    solver = MadNLP.MadNLPSolver(
        nlp;
        kkt_system=HybridKKT.HybridCondensedKKTSystem,
        equality_treatment=MadNLP.EnforceEquality,
        fixed_variable_treatment=MadNLP.MakeParameter,
        options...,
    )
    solver.kkt.gamma[] = gamma
    results = MadNLP.solve!(solver)
    cg_iters = solver.kkt.etc[:cg_iters]
    accuracy = solver.kkt.etc[:accuracy]
    return (
        status=results.status,
        n_iter=solver.cnt.k,
        cg_iters=cg_iters,
        accuracy=accuracy,
        t_total=solver.cnt.total_time,
        t_linear_solver=solver.cnt.linear_solver_time,
        t_cg=solver.kkt.etc[:time_cg],
        t_condensation=solver.kkt.etc[:time_condensation],
    )
end

@main function main(; solver="all", case="pglib_opf_case9241_pegase.m", tol=1e-4)
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end
    # Build model
    datafile = joinpath(PGLIB_PATH, case)

    # CPU
    if solver ∈ ("all", "cpu")
        nlp = ac_power_model(datafile)
        results_cpu = zeros(5, 6)
        # Warm-up
        results = solve_hybrid(
            nlp, 1e4;
            max_iter=1,
            print_level=MadNLP.INFO,
            linear_solver=HybridKKT.CHOLMODSolver,
            tol=tol,
        )
        # Extract results
        for (k, gamma) in enumerate([1e4, 1e5, 1e6, 1e7, 1e8])
            gamma_ = Int(log10(gamma))
            results = solve_hybrid(
                nlp, gamma;
                max_iter=200,
                print_level=MadNLP.INFO,
                linear_solver=HybridKKT.CHOLMODSolver,
                tol=tol,
            )
            output_file = joinpath(RESULTS_DIR, "hybrid-convergence-$(gamma_).txt")
            writedlm(output_file, [results.cg_iters results.accuracy])

            results_cpu[k, 1] = gamma
            results_cpu[k, 2] = results.n_iter
            results_cpu[k, 3] = results.t_condensation
            results_cpu[k, 4] = results.t_cg
            results_cpu[k, 5] = results.t_linear_solver
            results_cpu[k, 6] = results.t_total
        end
        writedlm(joinpath(RESULTS_DIR, "hybrid-stats-cpu.txt"), results_cpu)
    end

    # GPU solve
    if solver ∈ ("all", "cuda") && CUDA.has_cuda()
        results_cuda = zeros(5, 6)
        nlp_gpu = ac_power_model(datafile; backend=CUDABackend())
        # Warm-up
        results = solve_hybrid(
            nlp_gpu, 1e4;
            max_iter=1,
            print_level=MadNLP.INFO,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.BUNCHKAUFMAN,
            tol=tol,
        )
        # Extract results
        for (k, gamma) in enumerate([1e4, 1e5, 1e6, 1e7, 1e8])
            gamma_ = Int(log10(gamma))
            results = solve_hybrid(
                nlp_gpu, gamma;
                max_iter=200,
                print_level=MadNLP.INFO,
                linear_solver=MadNLPGPU.CUDSSSolver,
                cudss_algorithm=MadNLP.BUNCHKAUFMAN,
                tol=tol,
            )

            results_cuda[k, 1] = gamma
            results_cuda[k, 2] = results.n_iter
            results_cuda[k, 3] = results.t_condensation
            results_cuda[k, 4] = results.t_cg
            results_cuda[k, 5] = results.t_linear_solver
            results_cuda[k, 6] = results.t_total
        end
        writedlm(joinpath(RESULTS_DIR, "hybrid-stats-cuda.txt"), results_cuda)
    end
end

