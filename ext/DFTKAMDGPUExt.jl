module DFTKAMDGPUExt
using AMDGPU
using LinearAlgebra
import DFTK: GPU
using DFTK

DFTK.synchronize_device(::GPU{<:AMDGPU.ROCArray}) = AMDGPU.synchronize()

# Temporary workaround to not trigger https://github.com/JuliaGPU/AMDGPU.jl/issues/734
function LinearAlgebra.cholesky(A::Hermitian{T, <:AMDGPU.ROCArray}) where {T}
    Acopy, info = AMDGPU.rocSOLVER.potrf!(A.uplo, copy(A.data))
    LinearAlgebra.Cholesky(Acopy, A.uplo, info)
end

end
