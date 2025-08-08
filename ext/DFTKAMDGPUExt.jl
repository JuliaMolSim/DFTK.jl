module DFTKAMDGPUExt
using AMDGPU
using PrecompileTools
using LinearAlgebra
import DFTK: GPU, precompilation_workflow
using DFTK

DFTK.synchronize_device(::GPU{<:AMDGPU.ROCArray}) = AMDGPU.synchronize()

# Temporary workaround to not trigger https://github.com/JuliaGPU/AMDGPU.jl/issues/734
function LinearAlgebra.cholesky(A::Hermitian{T, <:AMDGPU.ROCArray}) where {T}
    Acopy, info = AMDGPU.rocSOLVER.potrf!(A.uplo, copy(A.data))
    LinearAlgebra.Cholesky(Acopy, A.uplo, info)
end

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
