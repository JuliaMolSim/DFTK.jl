"""
DFTK --- The density-functional toolkit. Provides functionality for experimenting
with plane-wave density-functional theory algorithms.
"""
module DFTK

using Printf
using Markdown
using LinearAlgebra
using StaticArrays

# Core functionality
export Vec3
export Mat3
include("core/constants.jl")

export PlaneWaveBasis
export set_kpoints!
export basis_ρ
export PotLocal
include("core/PlaneWaveBasis.jl")
include("core/PotLocal.jl")

export Kinetic
include("core/Kinetic.jl")

export PspHgh
export eval_psp_local_fourier
export eval_psp_local_real
export eval_psp_projection_radial
include("core/PspHgh.jl")

include("core/occupation.jl")

export compute_density
include("core/compute_density.jl")

export PotHartree
include("core/PotHartree.jl")

export PotNonLocal
include("core/PotNonLocal.jl")

export PotXc
include("core/PotXc.jl")

export Hamiltonian
export apply_hamiltonian!
include("core/Hamiltonian.jl")

export PreconditionerKinetic
include("core/Preconditioner.jl")

export lobpcg
include("core/lobpcg.jl")

include("core/scf_damped.jl")
include("core/scf_nlsolve.jl")

# Utilities
export self_consistent_field
export build_local_potential
export build_nonlocal_projectors
export determine_grid_size
export guess_gaussian_sad
export guess_hcore
export kblock_as_matrix
export load_psp
include("utils/build_local_potential.jl")
include("utils/build_nonlocal_projectors.jl")
include("utils/determine_grid_size.jl")
include("utils/guess_gaussian_sad.jl")
include("utils/guess_hcore.jl")
include("utils/kblock_as_matrix.jl")
include("utils/self_consistent_field.jl")
include("utils/load_psp.jl")

end # module DFTK
