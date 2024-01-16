
using Revise
using Test
using CUTEst

include("hybrid.jl")

if @isdefined nlp
    finalize(nlp)
end

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
    options = MadNLP.MadNLPOptions(; linear_solver=linear_solver)
    options_linear_solver = MadNLP.default_options(linear_solver)
    cnt = MadNLP.MadNLPCounters(; start_time=time())

    # Callback
    ind_cons = MadNLP.get_index_constraints(
        nlp,
        options.fixed_variable_treatment,
        options.equality_treatment,
    )
    inner_cb = MadNLP.create_callback(
        MadNLP.SparseCallback,
        nlp,
        options,
    )
    # Build local KKT system
    kkt = MadNLP.create_kkt_system(
        HybridCondensedKKTSystem, inner_cb, options, options_linear_solver, cnt, ind_cons,
    )
    initialize_kkt!(kkt, inner_cb)
    MadNLP.factorize!(kkt.linear_solver)

    kkt_ref = MadNLP.create_kkt_system(
        MadNLP.SparseKKTSystem, inner_cb, options, options_linear_solver, cnt, ind_cons,
    )
    initialize_kkt!(kkt_ref, inner_cb)
    MadNLP.factorize!(kkt_ref.linear_solver)
    x_ref = MadNLP.UnreducedKKTVector(kkt_ref)
    MadNLP.full(x_ref) .= 1.0
    MadNLP.solve!(kkt_ref, x_ref)

    # Test consistency of Jacobian
    n = get_nvar(nlp)
    @test kkt.jt_csc' == kkt_ref.jac_com[:, 1:n]
    @test kkt.jt_csc[:, kkt.ind_eq] == kkt.G_csc'

    b = MadNLP.UnreducedKKTVector(kkt)
    x = MadNLP.UnreducedKKTVector(kkt)
    MadNLP.full(x) .= 1.0
    MadNLP.solve!(kkt, x)

    @test x.values ≈ x_ref.values
    mul!(b, kkt_ref, x_ref)
    mul!(b, kkt, x_ref)
    @test MadNLP.full(b) ≈ ones(length(b)) atol=1e-6
end

nlp = CUTEstModel("HS35")

# Test HybridKKTSystem is returning the correct result
test_hybrid_kkt(nlp)

# Test full solve
solver = MadNLPSolver(
    nlp;
    linear_solver=LapackCPUSolver,
    lapack_algorithm=MadNLP.CHOLESKY,
    kkt_system=HybridCondensedKKTSystem,
    print_level=MadNLP.INFO,
    max_iter=1000,
    nlp_scaling=true,
    tol=1e-5,
)
MadNLP.solve!(solver)

finalize(nlp)
