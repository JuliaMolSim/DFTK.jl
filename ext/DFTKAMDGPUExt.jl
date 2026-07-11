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

# NOTE: the AMDGPU `LinearAlgebra` workarounds needed by the iterative eigensolver
# (Cholesky, SVD/gesdd!, and 5-argument mul!) now live in LOBPCG.jl's AMDGPU
# extension, which is loaded automatically whenever DFTK and AMDGPU are both present.

# Ensure precompilation is only performed if an AMD GPU is available
# AMDGPU pre-compiliation is currently broken on Julia > 1.10,
# see https://github.com/JuliaMolSim/DFTK.jl/issues/1278
if AMDGPU.functional() && VERSION < v"1.11"
    # Precompilation block with a basic workflow
    @setup_workload begin
        # very artificial silicon ground state example
        a = 10.26
        lattice = a / 2 * [[0 1 1.];
                        [1 0 1.];
                        [1 1 0.]]
        pseudofile = joinpath(@__DIR__, "..", "test", "pseudos", "gth", "Si.pbe-hgh.upf")
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
