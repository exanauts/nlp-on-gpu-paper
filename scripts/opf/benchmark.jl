
using PowerModels
using ExaModels
using JuMP
using CUDA
using CUDA.CUSPARSE
using MadNLPGPU

CUDA.allowscalar(false)

PowerModels.silence()

if !@isdefined ac_power_model
    include("model.jl")
end
include(joinpath(@__DIR__), "..", "benchmark.jl")

case = "/home/fpacaud/dev/pglib-opf/pglib_opf_case2000_goc.m"

#=
    CPU
=#
# HSL Ma27
nlp = ac_power_model(case)
solve_hsl(nlp; tol=1e-3)

# SparseCondensedKKT + CHOLMOD
solve_sparse_condensed(nlp; linear_solver=CHOLMODSolver, tol=1e-3)

# HybridKKT + CHOLMOD
solve_hybrid(nlp; linear_solver=CHOLMODSolver, tol=1e-3)

#=
    CUDA
=#
nlp_gpu = ac_power_model(case; backend=CUDABackend())
# SparseCondensedKKT + CUDSS
solve_sparse_condensed(nlp_gpu; linear_solver=MadNLPGPU.CUDSSSolver, tol=1e-3)

# HybridKKT + CHOLMOD
solve_hybrid(nlp_gpu; linear_solver=MadNLPGPU.CUDSSSolver, tol=1e-3)
