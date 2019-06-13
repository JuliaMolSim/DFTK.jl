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
export gcoords
export PotLocal
include("core/PlaneWaveBasis.jl")
include("core/PotLocal.jl")

export Kinetic
include("core/Kinetic.jl")

#
# TODO
#

export PspHgh
export eval_psp_local_fourier
export eval_psp_projection_radial
include("PspHgh.jl")

export compute_density
include("compute_density.jl")

export PotHartree
include("PotHartree.jl")

export PotNonLocal
include("PotNonLocal.jl")

export Hamiltonian
export apply_fourier!
export apply_real!
include("Hamiltonian.jl")

export PreconditionerKinetic
include("Preconditioner.jl")

export lobpcg
include("lobpcg.jl")

export self_consistent_field
include("self_consistent_field.jl")

# Utilities
export determine_grid_size
export build_local_potential
include("utils/determine_grid_size.jl")
include("utils/build_local_potential.jl")

end # module DFTK
