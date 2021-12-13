"""
DFTK --- The density-functional toolkit. Provides functionality for experimenting
with plane-wave density-functional theory algorithms.
"""
module DFTK

using Printf
using Markdown
using LinearAlgebra
using Requires
using TimerOutputs
using spglib_jll
using Unitful
using UnitfulAtomic

export Vec3
export Mat3
export mpi_nprocs
export mpi_master
export setup_threading, disable_threading
include("common/timer.jl")
include("common/asserting.jl")
include("common/constants.jl")
include("common/ortho.jl")
include("common/types.jl")
include("common/spherical_harmonics.jl")
include("common/split_evenly.jl")
include("common/mpi.jl")
include("common/threading.jl")
include("common/printing.jl")

export PspHgh
include("pseudo/NormConservingPsp.jl")
include("pseudo/PspHgh.jl")

export ElementPsp
export ElementCohenBergstresser
export ElementCoulomb
export charge_nuclear
export charge_ionic
export atomic_symbol
export n_elec_valence
export n_elec_core
include("elements.jl")

export Smearing
export Model
export PlaneWaveBasis
export compute_fft_size
export G_vectors, G_vectors_cart, r_vectors, r_vectors_cart
export Gplusk_vectors, Gplusk_vectors_cart
export Kpoint
export G_to_r
export G_to_r!
export r_to_G
export r_to_G!
include("Smearing.jl")
include("Model.jl")
include("structure.jl")
include("PlaneWaveBasis.jl")
include("fft.jl")
include("orbitals.jl")
include("show.jl")

export Energies
include("Energies.jl")

export Hamiltonian
export HamiltonianBlock
export energy_hamiltonian
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
export Anyonic
export apply_kernel
export compute_kernel
include("terms/terms.jl")

include("occupation.jl")
export compute_density
export total_density
export spin_density
export ρ_from_total_and_spin
include("densities.jl")
include("interpolation_transfer.jl")
export compute_transfer_matrix
export transfer_blochwave
export transfer_blochwave_kpt

export PreconditionerTPA
export PreconditionerNone
export lobpcg_hyper
export diag_full
export diagonalize_all_kblocks
include("eigen/preconditioners.jl")
include("eigen/diag.jl")

export model_atomic
export model_DFT
export model_PBE
export model_LDA
include("standard_models.jl")

export KerkerMixing, KerkerDosMixing, SimpleMixing, DielectricMixing
export LdosMixing, HybridMixing, χ0Mixing
export scf_nlsolve_solver
export scf_damping_solver
export scf_anderson_solver
export scf_CROP_solver
export self_consistent_field
export direct_minimization
export newton
export load_scfres, save_scfres
include("scf/chi0models.jl")
include("scf/mixing.jl")
include("scf/scf_solvers.jl")
include("scf/self_consistent_field.jl")
include("scf/direct_minimization.jl")
include("scf/newton.jl")
include("scf/scfres.jl")
include("scf/potential_mixing.jl")

export symmetry_operations
export standardize_atoms
export bzmesh_uniform
export bzmesh_ir_wedge
export kgrid_from_minimal_spacing, kgrid_from_minimal_n_kpoints
include("symmetry.jl")
include("bzmesh.jl")

export guess_density
export random_density
export load_psp
export list_psp
include("guess_density.jl")
include("pseudo/load_psp.jl")
include("pseudo/list_psp.jl")

export pymatgen_structure
export ase_atoms
export load_lattice
export load_basis
export load_model
export load_density
export load_atoms
export load_magnetic_moments
export run_wannier90
include("external/abinit.jl")
include("external/load_from_python.jl")
include("external/load_from_file.jl")
include("external/ase.jl")
include("external/pymatgen.jl")
include("external/stubs.jl")  # Function stubs for conditionally defined methods

export compute_bands
export high_symmetry_kpath
export plot_bandstructure
include("postprocess/band_structure.jl")

export compute_forces
export compute_forces_cart
include("postprocess/forces.jl")
export compute_stresses
include("postprocess/stresses.jl")
export compute_dos
export compute_ldos
export compute_nos
export plot_dos
include("postprocess/dos.jl")
export compute_χ0
export apply_χ0
include("postprocess/chi0.jl")
export compute_current
include("postprocess/current.jl")

# ForwardDiff workarounds
include("workarounds/dummy_inplace_fft.jl")
include("workarounds/forwarddiff_rules.jl")


function __init__()
    # Use "@require" to only include fft_generic.jl once IntervalArithmetic or
    # DoubleFloats has been loaded (via a "using" or an "import").
    # See https://github.com/JuliaPackaging/Requires.jl for details.
    #
    # The global variable GENERIC_FFT_LOADED makes sure that things are
    # only included once.
    @require IntervalArithmetic="d1acc4aa-44c8-5952-acd4-ba5d80a2a253" begin
        include("workarounds/intervals.jl")
        !isdefined(DFTK, :GENERIC_FFT_LOADED) && include("workarounds/fft_generic.jl")
    end
    @require DoubleFloats="497a8b3b-efae-58df-a0af-a86822472b78" begin
        !isdefined(DFTK, :GENERIC_FFT_LOADED) && include("workarounds/fft_generic.jl")
    end
    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80"    include("plotting.jl")
    @require JLD2="033835bb-8acc-5ee8-8aae-3f567f8a3819"     include("external/jld2io.jl")
    @require WriteVTK="64499a7a-5c06-52f2-abe2-ccb03c286192" include("external/vtkio.jl")
    @require NCDatasets="85f8d34a-cbdd-5861-8df4-14fed0d494ab" begin
        include("external/etsf_nanoquanta.jl")
    end
    @require wannier90_jll="c5400fa0-8d08-52c2-913f-1e3f656c1ce9" begin
        include("external/wannier90.jl")
    end
end

end # module DFTK
