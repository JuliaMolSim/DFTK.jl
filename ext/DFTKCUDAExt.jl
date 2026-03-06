module DFTKCUDAExt
using CUDA
using PrecompileTools
import DFTK: CPU, GPU, DispatchFunctional, precompilation_workflow, DispatchFloat
using DftFunctionals
using DFTK
using Libxc
import ForwardDiff: Dual

DFTK.synchronize_device(::GPU{<:CUDA.CuArray}) = CUDA.synchronize()

function DFTK.memory_usage(::GPU{<:CUDA.CuArray})
    merge(DFTK.memory_usage(CPU()), (; gpu=CUDA.memory_stats().live))
end

function DftFunctionals.potential_terms(fun::DispatchFunctional,
                                        ρ::CUDA.CuMatrix{<:DispatchFloat}, args...)
    @assert Libxc.has_cuda()
    potential_terms(fun.inner, ρ, args...)
end

# Ensure DFTK's custom ForwardDiff rule for FFTs is used.
# See: https://github.com/JuliaGPU/CUDA.jl/issues/3018
function Base.:*(p::CUFFT.CuFFTPlan{T,S,K,false},
                 x::CuArray{<:Complex{<:Dual{Tg}}}) where {T,S,K,Tg}
    DFTK.dual_fft_mul(p, x)
end

# Insure pre-compilation can proceed without error (old Julia/packages versions)
if Libxc.has_cuda() && !isnothing(Base.get_extension(Libxc, :LibxcCudaExt))

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
                                    architecture=GPU(CuArray))
        end
    end
end

end
