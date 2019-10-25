"""
DFTK --- The density-functional toolkit. Provides functionality for experimenting
with plane-wave density-functional theory algorithms.
"""
module DFTK

using Printf
using Markdown
using LinearAlgebra

#
# Core functionality
#
include("core/asserting.jl")
include("core/constants.jl")

export Vec3
export Mat3
include("core/types.jl")

export smearing_fermi_dirac
export smearing_gaussian
export smearing_methfessel_paxton_1
export smearing_methfessel_paxton_2
include("core/smearing_functions.jl")

export Model
export PlaneWaveModel
export basis_Cρ
export Kpoint
include("core/Model.jl")
include("core/PlaneWaveModel.jl")

export Density
export fourier
export real
include("core/Density.jl")

export Species
export charge_nuclear
export charge_ionic
export n_elec_valence
export n_elec_core
include("core/Species.jl")

export PspHgh
export eval_psp_local_fourier
export eval_psp_local_real
export eval_psp_projection_radial
include("core/PspHgh.jl")

export term_external
export term_hartree
export term_nonlocal
export term_xc
include("core/energy_term_operator.jl")
include("core/Kinetic.jl")
include("core/PotNonLocal.jl")
include("core/term_external.jl")
include("core/term_hartree.jl")
include("core/term_nonlocal.jl")
include("core/term_xc.jl")

export find_fermi_level
include("core/occupation.jl")

export compute_density
include("core/compute_density.jl")

export Hamiltonian
export HamiltonianBlock
export update_hamiltonian
export update_hamiltonian!
export update_energies_hamiltonian!
export update_energies!
export update_energies
export kblock
include("core/Hamiltonian.jl")
include("core/HamiltonianBlock.jl")

export PreconditionerKinetic
include("core/Preconditioner.jl")

export diag_lobpcg_hyper
export diag_lobpcg_scipy
export diag_lobpcg_itsolve
include("core/diag.jl")

export scf_nlsolve_solver
export scf_damping_solver
export scf_anderson_solver
export scf_CROP_solver
export self_consistent_field!
export diag_lobpcg
include("core/self_consistent_field.jl")
include("core/scf_solvers.jl")

export energy_ewald
include("core/energy_ewald.jl")

#
# Utilities
#
export bzmesh_uniform
export bzmesh_ir_wedge
include("utils/bzmesh.jl")

export determine_grid_size
export guess_gaussian_sad
export load_psp
include("utils/determine_grid_size.jl")
include("utils/guess_gaussian_sad.jl")
include("utils/load_psp.jl")

export energy_nuclear_psp_correction
export energy_nuclear_ewald
include("utils/energy_nuclear.jl")

export determine_high_symmetry_kpath
export compute_bands
export pymatgen_bandstructure
export pymatgen_structure
include("utils/compute_bands.jl")

export model_free_electron
export model_dft
export model_hcore
export model_reduced_hf
include("utils/standard_models.jl")

export EtsfFolder
export load_basis
export load_model
export load_density
export load_composition
include("utils/etsf_nanoquanta.jl")

end # module DFTK
