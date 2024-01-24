
using CUDA
using CUDA.CUSPARSE
using MadNLPGPU

CUDA.allowscalar(false)

if !@isdefined ac_power_model
    include("model.jl")
end

case = "/home/fpacaud/dev/pglib-opf/pglib_opf_case118_ieee.m"

nlp_gpu = ac_power_model(case; backend=CUDABackend())

solver = MadNLPSolver(
    nlp_gpu;
    linear_solver=MadNLPGPU.CUDSSSolver,
    lapack_algorithm=MadNLP.CHOLESKY,
    kkt_system=HybridCondensedKKTSystem,
    equality_treatment=MadNLP.EnforceEquality,
    fixed_variable_treatment=MadNLP.MakeParameter,
    print_level=MadNLP.DEBUG,
    inertia_correction_method=MadNLP.InertiaBased,
    max_iter=200,
    nlp_scaling=true,
    tol=1e-4,
)
solver.kkt.gamma[] = 1e7
MadNLP.solve!(solver)

