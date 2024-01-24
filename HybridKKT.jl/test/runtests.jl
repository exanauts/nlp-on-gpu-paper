
using Test
using LinearAlgebra
using SparseArrays
using HybridKKT
using NLPModels
using MadNLP
using CUTEst

function initialize_kkt!(kkt, cb)
    MadNLP.initialize!(kkt)
    # Compute initial values for Hessian and Jacobian
    x0 = NLPModels.get_x0(cb.nlp)
    y0 = NLPModels.get_y0(cb.nlp)
    # Update Jacobian manually
    jac = MadNLP.get_jacobian(kkt)
    MadNLP._eval_jac_wrapper!(cb, x0, jac)
    MadNLP.compress_jacobian!(kkt)
    # Update Hessian manually
    hess = MadNLP.get_hessian(kkt)
    MadNLP._eval_lag_hess_wrapper!(cb, x0, y0, hess)
    MadNLP.compress_hessian!(kkt)

    MadNLP._set_aug_diagonal!(kkt)
    MadNLP.build_kkt!(kkt)
    return
end

function test_hybrid_kkt(nlp)
    # Parameters
    linear_solver = LapackCPUSolver

    # Callback
    ind_cons = MadNLP.get_index_constraints(
        nlp,
    )
    cb = MadNLP.create_callback(
        MadNLP.SparseCallback,
        nlp,
    )

    # Build reference KKT system (here SparseKKTSystem)
    kkt_ref = MadNLP.create_kkt_system(
        MadNLP.SparseKKTSystem, cb, ind_cons, linear_solver;
    )
    initialize_kkt!(kkt_ref, cb)
    MadNLP.factorize!(kkt_ref.linear_solver)
    x_ref = MadNLP.UnreducedKKTVector(kkt_ref)
    MadNLP.full(x_ref) .= 1.0
    MadNLP.solve!(kkt_ref, x_ref)

    # Build HybridCondensedKKTSystem
    kkt = MadNLP.create_kkt_system(
        HybridCondensedKKTSystem, cb, ind_cons, linear_solver;
    )
    initialize_kkt!(kkt, cb)
    MadNLP.factorize!(kkt.linear_solver)
    x = MadNLP.UnreducedKKTVector(kkt)
    MadNLP.full(x) .= 1.0
    MadNLP.solve!(kkt, x)

    # Test backsolve returns the same values as with SparseKKTSystem.
    @test x.values ≈ x_ref.values atol=1e-6

    # Test consistency of Jacobian
    n = get_nvar(nlp)
    @test kkt.jt_csc' == kkt_ref.jac_com[:, 1:n]
    @test kkt.jt_csc[:, kkt.ind_eq] == kkt.G_csc'

    # Test KKT multiplication
    b = MadNLP.UnreducedKKTVector(kkt)
    mul!(b, kkt, x)
    @test MadNLP.full(b) ≈ ones(length(b)) atol=1e-6
end

@testset "HybridCondensedKKTSystem" begin
    # Use a small instance in CUTEst for our tests.
    nlp = CUTEstModel("HS35")

    # Test HybridKKTSystem is returning the correct result
    test_hybrid_kkt(nlp)

    # Test full solve
    solver = MadNLPSolver(
        nlp;
        kkt_system=HybridCondensedKKTSystem,
        linear_solver=LapackCPUSolver,
        lapack_algorithm=MadNLP.CHOLESKY,
        print_level=MadNLP.ERROR,
        tol=1e-5,
    )
    stats = MadNLP.solve!(solver)

    # Reference
    solver_ref = MadNLPSolver(
        nlp;
        kkt_system=MadNLP.SparseKKTSystem,
        linear_solver=LapackCPUSolver,
        print_level=MadNLP.ERROR,
        tol=1e-5,
    )
    stats_ref = MadNLP.solve!(solver_ref)

    finalize(nlp)

    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test stats.iter == stats_ref.iter
    @test stats.solution ≈ stats_ref.solution atol=1e-6
end

