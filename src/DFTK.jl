"""
DFTK --- The density-functional toolkit. Provides functionality for experimenting
with plane-wave density-functional theory algorithms.
"""
module DFTK

using Printf
using Markdown
using LinearAlgebra
using Interpolations

include("common/asserting.jl")
include("common/constants.jl")
export Vec3
export Mat3
include("common/types.jl")
include("common/spherical_harmonics.jl")
export smearing_fermi_dirac
export smearing_gaussian
export smearing_methfessel_paxton_1
export smearing_methfessel_paxton_2
include("common/smearing_functions.jl")

export Model
export PlaneWaveBasis
export determine_grid_size
export basis_Cρ
export Kpoint
export G_to_r
export G_to_r!
export r_to_G
export r_to_G!
include("Model.jl")
include("PlaneWaveBasis.jl")

export Density
export fourier
export real
include("Density.jl")

export Species
export charge_nuclear
export charge_ionic
export n_elec_valence
export n_elec_core
include("Species.jl")

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
include("occupation.jl")

export compute_density
include("compute_density.jl")

export Hamiltonian
export HamiltonianBlock
export update_hamiltonian
export update_hamiltonian!
export update_energies_hamiltonian!
export update_energies!
export update_energies
export kblock
include("Hamiltonian.jl")
include("HamiltonianBlock.jl")

export PreconditionerNone
export PreconditionerTPA
include("Preconditioner.jl")

export lobpcg_hyper
export lobpcg_scipy
export lobpcg_itsolve
export diagonalise_all_kblocks
include("eigen/diag.jl")

export scf_nlsolve_solver
export scf_damping_solver
export scf_anderson_solver
export scf_CROP_solver
export self_consistent_field!
export diag_lobpcg
include("scf/self_consistent_field.jl")
include("scf/scf_solvers.jl")

export energy_ewald
include("energy_ewald.jl")

#
# Utilities
#
export bzmesh_uniform
export bzmesh_ir_wedge
include("bzmesh.jl")

export guess_density
include("guess_density.jl")
export load_psp
include("pseudo/load_psp.jl")

export energy_nuclear_psp_correction
export energy_nuclear_ewald
include("energy_nuclear.jl")

export high_symmetry_kpath
export compute_bands
export pymatgen_lattice
export pymatgen_bandstructure
export pymatgen_structure
include("postprocess/compute_bands.jl")

export model_free_electron
export model_dft
export model_hcore
export model_reduced_hf
include("standard_models.jl")

export EtsfFolder
export load_basis
export load_model
export load_density
export load_composition
include("external/etsf_nanoquanta.jl")

end # module DFTK
