"""
DFTK --- The density-functional toolkit. Provides functionality for experimenting
with plane-wave density-functional theory algorithms.
"""
module DFTK

using Printf
using Markdown
using LinearAlgebra
using Interpolations
using Memoization

include("common/asserting.jl")
include("common/constants.jl")
export Vec3
export Mat3
include("common/types.jl")
include("common/check_real.jl")
include("common/spherical_harmonics.jl")
export Smearing
include("Smearing.jl")

export Model
export PlaneWaveBasis
export determine_grid_size
export G_vectors
export Kpoint
export G_to_r
export G_to_r!
export r_to_G
export r_to_G!
include("Model.jl")
include("PlaneWaveBasis.jl")

export RealFourierArray
export from_real
export from_fourier
include("RealFourierArray.jl")

export Element
export charge_nuclear
export charge_ionic
export n_elec_valence
export n_elec_core
include("Element.jl")

export PspHgh
export eval_psp_local_fourier
export eval_psp_local_real
export eval_psp_projection_radial
include("pseudo/PspHgh.jl")

export term_external
export term_hartree
export term_nonlocal
export term_xc
include("energy_term_operator.jl")
include("Kinetic.jl")
include("PotNonLocal.jl")
include("terms/term_external.jl")
include("terms/term_hartree.jl")
include("terms/term_nonlocal.jl")
include("terms/term_xc.jl")

export find_fermi_level
export find_occupation
export find_occupation_bandgap
include("occupation.jl")

export compute_density
include("densities.jl")

export Hamiltonian
export HamiltonianBlock
export update_hamiltonian
export update_hamiltonian!
export update_energies_hamiltonian!
export update_energies!
export update_energies
export print_energies
export kblock
include("Hamiltonian.jl")
include("HamiltonianBlock.jl")

export PreconditionerTPA
include("eigen/preconditioners.jl")

export lobpcg_hyper
export lobpcg_scipy
export lobpcg_itsolve
export diag_full
export diagonalise_all_kblocks
include("eigen/diag.jl")

export KerkerMixing
export SimpleMixing
include("scf/mixing.jl")
export scf_nlsolve_solver
export scf_damping_solver
export scf_anderson_solver
export scf_CROP_solver
include("scf/scf_solvers.jl")
export self_consistent_field
include("scf/self_consistent_field.jl")
export direct_minimization
include("scf/direct_minimization.jl")

export energy_ewald
include("energy_ewald.jl")

#
# Utilities
#
export bzmesh_uniform
export bzmesh_ir_wedge
export kgrid_size_from_minimal_spacing
include("bzmesh.jl")

export guess_density
include("guess_density.jl")
export load_psp
export list_psp
include("pseudo/load_psp.jl")

export energy_nuclear_psp_correction
export energy_nuclear_ewald
include("energy_nuclear.jl")

export compute_entropy_term
include("entropy.jl")

export pymatgen_lattice
export pymatgen_bandstructure
export pymatgen_structure
export high_symmetry_kpath
export compute_bands
export plot_bands
include("postprocess/band_structure.jl")

export DOS
export LDOS
export NOS
include("postprocess/DOS.jl")

export model_free_electron
export model_dft
export model_hcore
export model_reduced_hf
include("standard_models.jl")

export EtsfFolder
export load_lattice
export load_basis
export load_model
export load_density
export load_atoms
include("external/etsf_nanoquanta.jl")
include("external/abinit.jl")

export forces
include("forces.jl")

end # module DFTK
