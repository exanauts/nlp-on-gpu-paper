
using DelimitedFiles
using NLPModels
using ExaModels
using MadNLP
using MadNLPHSL

using HybridKKT
using CUDA
using MadNLPGPU

COPS_DIR = joinpath(@__DIR__, "..", "..", "examples", "cops")
for model in readdir(COPS_DIR)
    if isfile(joinpath(COPS_DIR, model))
        include(joinpath(COPS_DIR, model))
    end
end

benchmark_sizes = Dict(
    "robot_model" => [10000],
    "marine_model" => [10000, 20000, 50000, 100000, 200000],
    "steering_model" => [10000, 20000, 50000, 100000, 200000, 500000, 1000000],
    "dirichlet_model" => [10, 20, 50, 100, 200, 500],
    "gasoil_model" => [10000, 20000, 50000, 100000, 200000],
    "pinene_model" => [10000, 20000, 50000, 100000, 200000],
)

function clean_memory()
    GC.gc(true)
    CUDA.reclaim(true)
    GC.gc(true)
end

function solve_madnlp(nlp; options...)
    # Warm-up
    solver = MadNLPSolver(
        nlp;
        max_iter=1,
        options...
    )
    MadNLP.solve!(solver)

    solver = MadNLPSolver(
        nlp;
        options...
    )
    if isa(solver.kkt, HybridCondensedKKTSystem)
        solver.kkt.gamma[] = 1e7
    end

    res = MadNLP.solve!(solver)

    return (
        Int(res.status),
        NLPModels.get_nvar(nlp),
        NLPModels.get_ncon(nlp),
        NLPModels.get_nnzj(nlp),
        NLPModels.get_nnzh(nlp),
        solver.cnt.k,
        res.objective,
        solver.cnt.total_time,
        solver.cnt.linear_solver_time,
        solver.cnt.eval_function_time,
    )
end

function run_benchmark(instance, sizes; use_gpu=false, options...)
    results = zeros(length(sizes), 10)
    # Test full solve
    for (k, N) in enumerate(sizes)
        @info "Size: $(N)"
        nlp = use_gpu ? instance(N; backend=CUDABackend()) : instance(N)
        results[k, :] .= solve_madnlp(nlp; options...)

        clean_memory()
    end
    return [sizes results]
end


for instance in (marine_model, )
    sizes = benchmark_sizes[string(instance)]
    results = run_benchmark(instance, sizes; tol=1e-6, linear_solver=Ma57Solver)
    name = split(string(instance), '_')[1]
    writedlm("$(name).txt", results)
end

