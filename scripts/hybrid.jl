
using LinearAlgebra
using Printf
using NLPModels
using SparseArrays
using MadNLP
using Krylov
import MadNLP: SparseMatrixCOO, full

include("kernels.jl")

function _extract_subjacobian(jac::SparseMatrixCOO{Tv, Ti}, index_rows::AbstractVector{Int}) where {Tv, Ti}
    m, n = size(jac)
    nrows = length(index_rows)
    @assert nrows <= m

    # Scan inequality constraints.
    is_row_selected = zeros(Bool, m)
    new_index = zeros(Ti, m)
    cnt = 1
    for ind in index_rows
        is_row_selected[ind] = true
        new_index[ind] = cnt
        cnt += 1
    end

    # Count nnz
    nnzg = 0
    for (i, j) in zip(jac.I, jac.J)
        if is_row_selected[i]
            nnzg += 1
        end
    end

    G_i = zeros(Ti, nnzg)
    G_j = zeros(Ti, nnzg)
    G_v = zeros(Tv, nnzg)

    k, cnt = 1, 1
    for (i, j) in zip(jac.I, jac.J)
        if is_row_selected[i]
            G_i[cnt] = new_index[i]
            G_j[cnt] = j
            G_v[cnt] = k
            cnt += 1
        end
        k += 1
    end

    G = sparse(G_i, G_j, G_v, nrows, n)
    mapG = convert.(Int, nonzeros(G))

    return G, mapG
end

# Model linear operator G K Gᵀ  (dimension me x me)
struct SchurComplementOperator{T, VT, SMT, LS}
    K::LS      # dimension n x n
    G::SMT     # dimension me x n
    buf1::VT   # dimension n
end

Base.size(S::SchurComplementOperator) = (size(S.G, 1), size(S.G, 1))
Base.eltype(S::SchurComplementOperator{T}) where T = T

function SchurComplementOperator(
    K::MadNLP.AbstractLinearSolver,
    G::AbstractMatrix,
    buf::AbstractVector{T},
) where T
    return SchurComplementOperator{T, typeof(buf), typeof(G), typeof(K)}(
        K, G, buf,
    )
end

function LinearAlgebra.mul!(y::VT, S::SchurComplementOperator{T, VT}, x::VT, alpha::Number, beta::Number) where {T, VT}
    y .= beta .* y
    mul!(S.buf1, S.G', x, alpha, zero(T))
    MadNLP.solve!(S.K, S.buf1)
    mul!(y, S.G, S.buf1, one(T), one(T))
    return y
end


struct HybridCondensedKKTSystem{T, VT, MT, QN, VI, VI32, VInd, LS, LS2, EXT} <: MadNLP.AbstractCondensedKKTSystem{T, VT, MT, QN}
    # Hessian
    hess::VT      # dimension nnzh
    hess_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT  # dimension n x n
    hess_csc_map::Union{Nothing, VI}

    # Full Jacobian
    jac::VT       # dimension nnzj
    jt_coo::SparseMatrixCOO{T,Int32,VT, VI32}
    jt_csc::MT
    jt_csc_map::Union{Nothing, VI}

    # Jacobian of equality constraints
    G_csc::MT
    G_csc_map::Union{Nothing, VI}

    # Schur-complement operator
    S::SchurComplementOperator{T,VT,MT,LS}

    gamma::Ref{T}

    quasi_newton::QN
    reg::VT       # dimension n + mi
    pr_diag::VT   # dimension n + mi
    du_diag::VT   # dimension me + mi
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT

    # Buffer
    buffer1::VT   # dimension me + mi
    buffer2::VT   # dimension me + mi
    buffer3::VT   # dimension n
    buffer4::VT   # dimension mi
    buffer5::VT   # dimension mi
    buffer6::VT   # dimension me

    # Condensed system Kγ
    aug_com::MT   # dimension n x n

    # slack diagonal buffer
    diag_buffer::VT
    dptr::AbstractVector
    hptr::AbstractVector
    jptr::AbstractVector

    # LinearSolver
    linear_solver::LS
    iterative_linear_solver::LS2

    # Info
    ind_ineq::VInd  # dimension mi
    ind_eq::VInd    # dimension me
    ind_lb::VI
    ind_ub::VI

    ext::EXT
    # Stats
    etc::Dict{Symbol, Any}
end

# Build KKT system directly from SparseCallback
function MadNLP.create_kkt_system(
    ::Type{HybridCondensedKKTSystem},
    cb::MadNLP.SparseCallback{T,VT},
    opt,
    opt_linear_solver,
    cnt,
    ind_cons
) where {T, VT}

    n = cb.nvar
    m = cb.ncon
    ind_ineq = ind_cons.ind_ineq
    mi = length(ind_ineq)
    VI = typeof(ind_ineq)

    ind_eq = if isa(ind_ineq, Vector)
        setdiff(1:m, ind_ineq)
    else
        ind_ineq_host = Vector(ind_ineq)
        VI(setdiff(1:m, ind_ineq_host))
    end
    me = m - mi

    # Evaluate sparsity pattern
    jac_sparsity_I = MadNLP.create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = MadNLP.create_array(cb, Int32, cb.nnzj)
    MadNLP._jac_sparsity_wrapper!(cb,jac_sparsity_I, jac_sparsity_J)

    quasi_newton = MadNLP.create_quasi_newton(opt.hessian_approximation, cb, n)
    hess_sparsity_I, hess_sparsity_J = MadNLP.build_hessian_structure(cb, opt.hessian_approximation)

    MadNLP.force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + mi
    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)

    reg = VT(undef, n_tot)
    pr_diag = VT(undef, n_tot)
    du_diag = VT(undef, m)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)
    buffer1 = VT(undef, m)
    buffer2 = VT(undef, m)
    buffer3 = VT(undef, n)
    buffer4 = VT(undef, mi)
    buffer5 = VT(undef, mi)
    buffer6 = VT(undef, me)
    hess = VT(undef, n_hess)
    jac = VT(undef, n_jac)
    diag_buffer = VT(undef, m)
    fill!(jac, zero(T))

    hess_raw = SparseMatrixCOO(n, n, hess_sparsity_I, hess_sparsity_J, hess)

    jt_coo = SparseMatrixCOO(
        n, m,
        jac_sparsity_J,
        jac_sparsity_I,
        jac,
    )

    jt_csc, jt_csc_map = MadNLP.coo_to_csc(jt_coo)
    hess_com, hess_csc_map = MadNLP.coo_to_csc(hess_raw)
    aug_com, dptr, hptr, jptr = MadNLP.build_condensed_aug_symbolic(
        hess_com,
        jt_csc
    )

    # Build Jacobian of equality constraints
    jac_coo = SparseMatrixCOO(
        m, n,
        Vector(jac_sparsity_I),
        Vector(jac_sparsity_J),
        Vector(jac),
    )
    G_csc_, G_csc_map_ = _extract_subjacobian(jac_coo, Vector(ind_eq))
    MT = typeof(hess_com)
    G_csc = MT(G_csc_)
    G_csc_map = VI(G_csc_map_)

    gamma = Ref{T}(1000)

    cnt.linear_solver_time += @elapsed begin
        linear_solver = opt.linear_solver(aug_com; opt = opt_linear_solver)
    end

    buf1 = VT(undef, n)
    S = SchurComplementOperator(linear_solver, G_csc, buf1)

    iterative_linear_solver = Krylov.CgSolver(me, me, VT)

    ext = MadNLP.get_sparse_condensed_ext(VT, hess_com, jptr, jt_csc_map, hess_csc_map)
    etc = Dict{Symbol, Any}(
        :cg_iters=>Int[],
        :accuracy=>Float64[],
    )

    return HybridCondensedKKTSystem(
        hess, hess_raw, hess_com, hess_csc_map,
        jac, jt_coo, jt_csc, jt_csc_map,
        G_csc, G_csc_map,
        S,
        gamma,
        quasi_newton,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        buffer1, buffer2, buffer3, buffer4, buffer5, buffer6,
        aug_com, diag_buffer, dptr, hptr, jptr,
        linear_solver, iterative_linear_solver,
        ind_ineq, ind_eq, ind_cons.ind_lb, ind_cons.ind_ub,
        ext, etc,
    )
end

function MadNLP.initialize!(kkt::HybridCondensedKKTSystem)
    fill!(kkt.reg, 1.0)
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
    fill!(kkt.hess, 0.0)
    fill!(kkt.l_lower, 0.0)
    fill!(kkt.u_lower, 0.0)
    fill!(kkt.l_diag, 1.0)
    fill!(kkt.u_diag, 1.0)
    fill!(nonzeros(kkt.hess_com), 0.) # so that mul! in the initial primal-dual solve has no effect
end

function MadNLP.is_inertia_correct(kkt::HybridCondensedKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == size(kkt.aug_com, 1))
end

# mul!
# TODO: problem with synchronization
function LinearAlgebra.mul!(w::MadNLP.AbstractKKTVector{T}, kkt::HybridCondensedKKTSystem, x::MadNLP.AbstractKKTVector, alpha, beta) where T
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)
    mi = length(kkt.ind_ineq)

    # Decompose results
    xx = view(full(x), 1:n)
    xs = view(full(x), n+1:n+mi)
    xz = view(full(x), n+mi+1:n+mi+m)

    # Decompose buffers
    wx = view(full(w), 1:n)
    ws = view(full(w), n+1:n+mi)
    wz = view(full(w), n+mi+1:n+mi+m)

    wz_ineq = view(wz, kkt.ind_ineq)
    xz_ineq = view(xz, kkt.ind_ineq)
    # wz_ineq = kkt.buffer4
    # xz_ineq = kkt.buffer5

    # index_copy!(xz_ineq, xz, kkt.ind_ineq)

    mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta)

    mul!(wx, kkt.jt_csc, xz, alpha, one(T))
    mul!(wz, kkt.jt_csc', xx, alpha, beta)
    # index_copy!(wz_ineq, wz, kkt.ind_ineq)
    axpy!(-alpha, xs, wz_ineq)

    ws .= beta.*ws .- alpha.* xz_ineq

    # index_copy!(wz, kkt.ind_ineq, wz_ineq)

    MadNLP._kktmul!(w,x,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)
    return w
end

# get_jacobian
MadNLP.get_jacobian(kkt::HybridCondensedKKTSystem) = kkt.jac

# compress_jacobian!
function MadNLP.compress_jacobian!(kkt::HybridCondensedKKTSystem)
    fill!(nonzeros(kkt.jt_csc), 0.0)
    MadNLP.transfer!(kkt.jt_csc, kkt.jt_coo, kkt.jt_csc_map)
    nonzeros(kkt.G_csc) .= kkt.jac[kkt.G_csc_map]
    return
end

# jtprod!
function MadNLP.jtprod!(y::AbstractVector, kkt::HybridCondensedKKTSystem, x::AbstractVector)
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    x_ineq = view(x, kkt.ind_ineq)
    mul!(view(y, 1:n), kkt.jt_csc, x)
    y[size(kkt.jt_csc,1)+1:end] .= -x_ineq
    return y
end

function MadNLP.compress_hessian!(kkt::HybridCondensedKKTSystem)
    MadNLP.transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end


# build_kkt!
function MadNLP.build_kkt!(kkt::HybridCondensedKKTSystem)
    n = size(kkt.hess_com, 1)
    mi = length(kkt.ind_ineq)
    m = size(kkt.jt_csc, 2)

    Σx = view(kkt.pr_diag, 1:n)
    Σs = view(kkt.pr_diag, n+1:n+mi)
    Σd = kkt.du_diag # TODO: add support

    fill!(kkt.diag_buffer, 0.0)
    index_copy!(kkt.diag_buffer, kkt.ind_ineq, Σs)
    # Regularization for equality
    fixed!(kkt.diag_buffer, kkt.ind_eq, kkt.gamma[])
    MadNLP.build_condensed_aug_coord!(kkt)
    return
end

# solve!
function MadNLP.solve!(kkt::HybridCondensedKKTSystem{T}, w::MadNLP.AbstractKKTVector)  where T
    (n,m) = size(kkt.jt_csc)
    mi = length(kkt.ind_ineq)
    G = kkt.G_csc

    # Decompose buffers
    wx = MadNLP._madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+mi)
    wc = view(full(w), n+mi+1:n+mi+m)

    r1 = kkt.buffer3
    vs = kkt.buffer4
    wz = kkt.buffer5
    wy = kkt.buffer6

    index_copy!(wy, wc, kkt.ind_eq)
    index_copy!(wz, wc, kkt.ind_ineq)

    Σs = view(kkt.pr_diag, n+1:n+mi)

    MadNLP.reduce_rhs!(w.xp_lr, MadNLP.dual_lb(w), kkt.l_diag, w.xp_ur, MadNLP.dual_ub(w), kkt.u_diag)

    # Condensation
    fill!(kkt.buffer1, zero(T))
    vs .= Σs .* wz .+ ws
    index_copy!(kkt.buffer1, kkt.ind_ineq, vs)
    mul!(wx, kkt.jt_csc, kkt.buffer1, one(T), one(T))

    #  Golub & Greif
    r1 .= wx
    mul!(r1, G', wy, kkt.gamma[], one(T))    # r1 = wx + γ Gᵀ wy
    wx .= r1                               # (save for later)
    MadNLP.solve!(kkt.linear_solver, r1)   # r1 = (Kγ)⁻¹ [wx + γ Gᵀ wy]
    mul!(wy, G, r1, one(T), -one(T))       # -wy + G (Kγ)⁻¹ [wx + γ Gᵀ wy]

    # Solve Schur-complement system with a Krylov iterative method.
    Krylov.solve!(kkt.iterative_linear_solver, kkt.S, wy; atol=1e-12, rtol=0.0, verbose=0)

    cg_iter = kkt.iterative_linear_solver.stats.niter
    push!(kkt.etc[:cg_iters], cg_iter)

    copyto!(wy, kkt.iterative_linear_solver.x)

    # Extract solution of Golub & Greif
    mul!(wx, G', wy, -one(T), one(T))
    MadNLP.solve!(kkt.linear_solver, wx)

    # Extract condensation
    mul!(kkt.buffer2, kkt.jt_csc', wx)
    vj = view(kkt.buffer2, kkt.ind_ineq)
    vs .= ws    # (save a copy of ws for later)
    ws .= vj .- wz
    wz .= Σs .* ws .- vs

    index_copy!(wc, kkt.ind_ineq, wz)
    index_copy!(wc, kkt.ind_eq, wy)

    MadNLP.finish_aug_solve!(kkt, w)
    return w
end

# Custom iterative-refinement
function MadNLP.solve_refine_wrapper!(
    d,
    solver::MadNLP.MadNLPSolver{T, VT, VI, KKT},
    p,
    w,
) where {T, VT, VI, KKT<:HybridCondensedKKTSystem{T}}
    copyto!(d.values, p.values)
    MadNLP.solve!(solver.kkt, d)

    # Compute backsolve's error
    copyto!(full(w), full(p))
    mul!(w, solver.kkt, d, -one(T), one(T))
    norm_w = norm(full(w), Inf)
    MadNLP.@debug(solver.logger, @sprintf("%4i %6.2e", 0, norm_w))

    # push!(solver.kkt.etc[:accuracy], norm_w)
    return true
end

#=
    GPU specific code (require MadNLPGPU).
=#

function MadNLP.compress_hessian!(kkt::HybridCondensedKKTSystem{T, VT, MT}) where {T, VT, MT<:CUDA.CUSPARSE.CuSparseMatrixCSC{T, Int32}}
    fill!(kkt.hess_com.nzVal, zero(T))
    MadNLPGPU._transfer!(CUDABackend())(kkt.hess_com.nzVal, kkt.ext.hess_com_ptr, kkt.ext.hess_com_ptrptr, kkt.hess_raw.V; ndrange = length(kkt.ext.hess_com_ptrptr)-1)
    KernelAbstractions.synchronize(CUDABackend())
end

function MadNLP.compress_jacobian!(kkt::HybridCondensedKKTSystem{T, VT, MT}) where {T, VT, MT<:CUDA.CUSOLVER.CuSparseMatrixCSC{T, Int32}}
    fill!(kkt.jt_csc.nzVal, zero(T))
    if length(kkt.ext.jt_csc_ptrptr) > 1 # otherwise error is thrown
        MadNLPGPU._transfer!(CUDABackend())(kkt.jt_csc.nzVal, kkt.ext.jt_csc_ptr, kkt.ext.jt_csc_ptrptr, kkt.jt_coo.V; ndrange = length(kkt.ext.jt_csc_ptrptr)-1)
    end
    KernelAbstractions.synchronize(CUDABackend())
    nonzeros(kkt.G_csc) .= kkt.jac[kkt.G_csc_map]
end

# N.B: we use the custom function implemented in MadNLPGPU
# for KKT multiplication as symv is not supported on the GPU.
function LinearAlgebra.mul!(
    w::MadNLP.AbstractKKTVector{T,VT},
    kkt::HybridCondensedKKTSystem,
    x::MadNLP.AbstractKKTVector,
    alpha = one(T), beta = zero(T)
) where {T, VT <: CuVector{T}}
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)
    mi = length(kkt.ind_ineq)

    # Decompose results
    xx = view(full(x), 1:n)
    xs = view(full(x), n+1:n+mi)
    xz = view(full(x), n+mi+1:n+mi+m)

    # Decompose buffers
    wx = view(full(w), 1:n)
    ws = view(full(w), n+1:n+mi)
    wz = view(full(w), n+mi+1:n+mi+m)

    wz_ineq = kkt.buffer4
    xz_ineq = kkt.buffer5


    # First block / x
    mul!(wx, kkt.hess_com , xx, alpha, beta)
    mul!(wx, kkt.hess_com', xx, alpha, one(T))
    mul!(wx, kkt.jt_csc,  xz, alpha, one(T))
    MadNLPGPU.diag_operation(CUDABackend())(
        wx, kkt.hess_com.nzVal, xx, alpha,
        kkt.ext.diag_map_to,
        kkt.ext.diag_map_fr;
        ndrange = length(kkt.ext.diag_map_to)
    )
    KernelAbstractions.synchronize(CUDABackend())

    # Second block / s
    # N.B. axpy! does not support SubArray
    index_copy!(xz_ineq, xz, kkt.ind_ineq)
    ws .= beta .* ws .- alpha .* xz_ineq

    # Third block / y
    mul!(wz, kkt.jt_csc', xx, alpha, beta)
    # Implements wz = wz - alpha * xs
    index_copy!(wz_ineq, wz, kkt.ind_ineq)
    axpy!(-alpha, xs, wz_ineq)
    index_copy!(wz, kkt.ind_ineq, wz_ineq)

    MadNLP._kktmul!(w,x,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)
    return
end
