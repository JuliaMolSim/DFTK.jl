module DFTKAMDGPUExt
using AMDGPU
using PrecompileTools
import Libdl
using LinearAlgebra
import DFTK: CPU, GPU, precompilation_workflow
using DFTK

DFTK.synchronize_device(::GPU{<:AMDGPU.ROCArray}) = AMDGPU.synchronize()

function DFTK.memory_usage(::GPU{<:AMDGPU.ROCArray})
    merge(DFTK.memory_usage(CPU()), (; gpu=AMDGPU.memory_stats().live))
end

global const libroctx = Ref{String}("")

function __init__()
    # Register rocTX instrumentation callbacks if available
    libroctx[] = Libdl.find_library("libroctx64")
    if libroctx[] != ""
        function push_range(message::String)
            ccall((:roctxRangePushA, libroctx[]), Cvoid, (Cstring,), message)
        end

        function pop_range(sync_device::Bool)
            ccall((:roctxRangePop, libroctx[]), Cvoid, ())
            if sync_device
                AMDGPU.synchronize()
            end
        end

        DFTK.register_instrumentation_callback(
            "ROC-TX",
            push_range,
            pop_range)
    else
        @warn "libroctx64 is unavailable, ROCm instrumentation will be disabled."
    end
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
