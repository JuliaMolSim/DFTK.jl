module DFTKAMDGPUExt
using AMDGPU
using PrecompileTools
using LinearAlgebra
import DFTK: CPU, GPU, precompilation_workflow
using DFTK

DFTK.synchronize_device(::GPU{<:AMDGPU.ROCArray}) = AMDGPU.synchronize()

function DFTK.memory_usage(::GPU{<:AMDGPU.ROCArray})
    merge(DFTK.memory_usage(CPU()), (; gpu=AMDGPU.memory_stats().live))
end

# Temporary workaround to not trigger https://github.com/JuliaGPU/AMDGPU.jl/issues/734
function LinearAlgebra.cholesky(A::Hermitian{T, <:AMDGPU.ROCArray}) where {T}
    Acopy, info = AMDGPU.rocSOLVER.potrf!(A.uplo, copy(A.data))
    LinearAlgebra.Cholesky(Acopy, A.uplo, info)
end

# Temporary workaround for SVD. See https://github.com/JuliaGPU/AMDGPU.jl/issues/837
function LinearAlgebra.LAPACK.gesdd!(jobz::Char, A::AMDGPU.ROCArray{T}) where {T}
    AMDGPU.rocSOLVER.gesvd!(jobz, jobz, A)
end

# Temporary workaround for 5-argumet mul!, where performance is very bad when array
# element types and scaling factors types differ.
# See https://github.com/JuliaGPU/AMDGPU.jl/issues/866#issuecomment-3636981853
# Scaling a Float/Complex matrix with an Integer:
function LinearAlgebra.mul!(C::AMDGPU.ROCArray{T}, A::AMDGPU.ROCArray{T}, B::AMDGPU.ROCArray{T},
                            α::U, β::U) where {T<:Union{AbstractFloat,Complex}, U<:Integer}
    LinearAlgebra.mul!(C, A, B, T(α), T(β))
end
# Scaling a Complex matrix with a Float:
function LinearAlgebra.mul!(C::AMDGPU.ROCArray{T}, A::AMDGPU.ROCArray{T}, B::AMDGPU.ROCArray{T},
                            α::U, β::U) where {T<:Complex, U<:AbstractFloat}
    LinearAlgebra.mul!(C, A, B, T(α), T(β))
end

# Ensure precompilation is only performed if an AMD GPU is available
if AMDGPU.functional()
    # Precompilation block with a basic workflow
    @setup_workload begin
        # very artificial silicon ground state example
        a = 10.26
        lattice = a / 2 * [[0 1 1.];
                        [1 0 1.];
                        [1 1 0.]]
        pseudofile = joinpath(@__DIR__, "..", "test", "gth_pseudos", "Si.pbe-hgh.upf")
        Si = ElementPsp(:Si, Dict(:Si => pseudofile))
        atoms     = [Si, Si]
        positions = [ones(3)/8, -ones(3)/8]
        magnetic_moments = [2, -2]

        @compile_workload begin
            precompilation_workflow(lattice, atoms, positions, magnetic_moments;
                                    architecture=GPU(ROCArray))
        end
    end
end

end
