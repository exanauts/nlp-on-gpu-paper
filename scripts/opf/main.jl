
using PowerModels
using ExaModels
using MadNLPHSL

PowerModels.silence()

function test_accuracy(solver)
    kkt = solver.kkt
    b = solver.p #MadNLP.UnreducedKKTVector(kkt)
    x = solver.d #MadNLP.UnreducedKKTVector(kkt)
    MadNLP.full(x) .= 1.0
    MadNLP.build_kkt!(kkt)
    MadNLP.factorize!(kkt.linear_solver)
    MadNLP.solve!(kkt, x)

    mul!(b, kkt, x)

    b_sol = full(b) |> Array
    b_ref = ones(length(b))

    println(norm(b_sol .- b_ref, Inf))
end

case = "/home/fpacaud/dev/pglib-opf/pglib_opf_case118_ieee.m"
# case = "/home/fpacaud/dev/pglib-opf/pglib_opf_case300_ieee.m"
case = "/home/fpacaud/dev/pglib-opf/pglib_opf_case1354_pegase.m"
# case = "/home/fpacaud/dev/pglib-opf/pglib_opf_case2000_goc.m"
# case = "/home/fpacaud/dev/pglib-opf/pglib_opf_case4837_goc.m"
# case = "/home/fpacaud/dev/pglib-opf/pglib_opf_case4917_goc.m"

nlp = ac_power_model(case)
# solve_hsl(nlp)

# solver = solve_hybrid(nlp)


solver = MadNLPSolver(
    nlp;
    # linear_solver=LapackCPUSolver,
    linear_solver=CHOLMODSolver,
    lapack_algorithm=MadNLP.CHOLESKY,
    kkt_system=HybridCondensedKKTSystem,
    print_level=MadNLP.DEBUG,
    inertia_correction_method=MadNLP.InertiaBased,
    max_iter=200,
    nlp_scaling=true,
    tol=1e-4,
)
solver.kkt.gamma[] = 1e6
MadNLP.solve!(solver)

# MadNLP.factorize_wrapper!(solver)
# kkt = solver.kkt
# b = MadNLP.UnreducedKKTVector(kkt)
# x = MadNLP.UnreducedKKTVector(kkt)
# w = MadNLP.UnreducedKKTVector(kkt)
# x.values .= 0
# b.values .= 1
# # x.values .= solver.p.values
# # MadNLP.dual(x) .= 1.0
# b_ref = copy(x.values)
# MadNLP.factorize!(kkt.linear_solver)


# norm_b = norm(full(b), Inf)
# copyto!(full(w), full(b))

# # Richardson
# MadNLP.solve!(kkt, w)
# axpy!(1., full(w), full(x))
# copyto!(full(w), full(b))
# mul!(w, kkt, x, -1.0, 1.0)

# norm_w = norm(full(w), Inf)
# norm_x = norm(full(x), Inf)
# b_sol = full(b)
# residual_ratio = norm_w / (min(norm_x, 1e6 * norm_b) + norm_b)
