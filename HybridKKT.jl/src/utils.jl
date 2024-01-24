
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

