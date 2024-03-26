
using MadNLP
using HybridKKT


if !@isdefined ac_power_model
    include("model.jl")
end

case = "/home/fpacaud/dev/pglib-opf/pglib_opf_case30000_goc.m"

nlp = ac_power_model(case)

solver = MadNLPSolver(
    nlp;
    linear_solver=HybridKKT.CHOLMODSolver,
    lapack_algorithm=MadNLP.CHOLESKY,
    kkt_system=HybridCondensedKKTSystem,
    print_level=MadNLP.DEBUG,
    inertia_correction_method=MadNLP.InertiaBased,
    max_iter=200,
    nlp_scaling=true,
    tol=1e-8,
)

solver.kkt.gamma[] = 1e7
MadNLP.solve!(solver)

