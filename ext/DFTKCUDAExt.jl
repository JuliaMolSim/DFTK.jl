module DFTKCUDAExt
using CUDA
using PrecompileTools
import DFTK: GPU, DispatchFunctional, precompilation_workflow
using DftFunctionals
using DFTK
using Libxc

DFTK.synchronize_device(::GPU{<:CUDA.CuArray}) = CUDA.synchronize()

for fun in (:potential_terms, :kernel_terms)
    @eval function DftFunctionals.$fun(fun::DispatchFunctional,
                                       ρ::CUDA.CuMatrix{Float64}, args...)
        @assert Libxc.has_cuda()
        $fun(fun.inner, ρ, args...)
    end
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
