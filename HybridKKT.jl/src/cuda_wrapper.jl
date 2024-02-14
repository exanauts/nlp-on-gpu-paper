
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
    if length(kkt.ind_eq) > 0
        transfer_coef!(kkt.G_csc, kkt.G_csc_map, kkt.jac, kkt.ind_eq_jac)
    end
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
