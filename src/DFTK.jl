"""
DFTK --- The density-functional toolkit. Provides functionality for experimenting
with plane-wave density-functional theory algorithms.
"""
module DFTK

using Printf
using Markdown
using LinearAlgebra
using Interpolations
using Requires
using TimerOutputs
using spglib_jll

include("common/timer.jl")
include("common/asserting.jl")
export unit_to_au
include("common/constants.jl")
include("common/units.jl")
export Vec3
export Mat3
include("common/types.jl")
include("common/check_real.jl")
include("common/spherical_harmonics.jl")
export Smearing
include("Smearing.jl")

export Model
export PlaneWaveBasis
export determine_fft_size
export G_vectors
export r_vectors
export Kpoint
export G_to_r
export G_to_r!
export r_to_G
export r_to_G!
export model_atomic
export model_DFT
export model_PBE
export model_LDA
include("Model.jl")
include("PlaneWaveBasis.jl")

export RealFourierArray
export from_real
export from_fourier
include("RealFourierArray.jl")

export ElementPsp
export ElementCohenBergstresser
export ElementCoulomb
export charge_nuclear
export charge_ionic
export n_elec_valence
export n_elec_core
include("elements.jl")

export PspHgh
export eval_psp_local_fourier
export eval_psp_local_real
export eval_psp_projection_radial
include("pseudo/PspHgh.jl")

export Energies
include("energies.jl")

export Hamiltonian
export HamiltonianBlock
export energy_hamiltonian
export ene_ops
export forces
export Kinetic
export ExternalFromFourier
export ExternalFromReal
export AtomicLocal
export PowerNonlinearity
export Hartree
export Xc
export AtomicNonlocal
export Ewald
export PspCorrection
export Entropy
export Magnetic
export energy_ewald
export energy_psp_correction
export apply_kernel
export compute_kernel
include("terms/terms.jl")

export fermi_level
export find_occupation
export find_occupation_bandgap
include("occupation.jl")

export compute_density
include("densities.jl")
include("interpolation.jl")

export PreconditionerTPA
export PreconditionerNone
include("eigen/preconditioners.jl")

export lobpcg_hyper
export lobpcg_itsolve
export diag_full
export diagonalize_all_kblocks
include("eigen/diag.jl")

export KerkerMixing
export SimpleMixing
export DielectricMixing
export HybridMixing
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

#
# Utilities
#
include("symmetry.jl")
export standardize_atoms
export bzmesh_uniform
export bzmesh_ir_wedge
export kgrid_size_from_minimal_spacing
include("bzmesh.jl")

export guess_density
include("guess_density.jl")
export load_psp
export list_psp
include("pseudo/load_psp.jl")
include("pseudo/list_psp.jl")

export pymatgen_structure
export ase_atoms
export EtsfFolder
export load_lattice
export load_basis
export load_model
export load_density
export load_atoms
include("external/etsf_nanoquanta.jl")
include("external/abinit.jl")
include("external/load_from_python.jl")
include("external/ase.jl")
include("external/pymatgen.jl")

export high_symmetry_kpath
export compute_bands
export plot_band_data
export plot_bandstructure
include("postprocess/band_structure.jl")

export DOS
export LDOS
export NOS
include("postprocess/DOS.jl")
export compute_χ0
export apply_χ0
include("postprocess/chi0.jl")

function __init__()
    # Use "@require" to only include fft_generic.jl once IntervalArithmetic or
    # DoubleFloats has been loaded (via a "using" or an "import").
    # See https://github.com/JuliaPackaging/Requires.jl for details.
    #
    # The global variable GENERIC_FFT_LOADED makes sure that things are only
    # included once.
    @require IntervalArithmetic="d1acc4aa-44c8-5952-acd4-ba5d80a2a253" begin
        include("intervals_workarounds.jl")
        !isdefined(DFTK, :GENERIC_FFT_LOADED) && include("fft_generic.jl")
    end
    @require DoubleFloats="497a8b3b-efae-58df-a0af-a86822472b78" begin
        !isdefined(DFTK, :GENERIC_FFT_LOADED) && include("fft_generic.jl")
    end
end

end # module DFTK
