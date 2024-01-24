module HybridKKT

using LinearAlgebra
using SparseArrays
using Printf

import Krylov
import NLPModels
import MadNLP
import MadNLPGPU
import MadNLP: SparseMatrixCOO, full

using CUDA
using KernelAbstractions

export HybridCondensedKKTSystem

include("utils.jl")
include("kernels.jl")
include("hybrid.jl")
include("cuda_wrapper.jl")
include("cholmod.jl")

end # module HybridKKT
