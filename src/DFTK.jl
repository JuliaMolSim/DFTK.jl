"""
DFTK --- The density-functional toolkit. Provides functionality for experimenting
with plane-wave density-functional theory algorithms.
"""
module DFTK

using LinearAlgebra
using Markdown
using Printf
using Requires
using TimerOutputs
using spglib_jll
using Unitful
using UnitfulAtomic
using ForwardDiff
using AbstractFFTs
using GPUArraysCore
using Random
using ChainRulesCore
using PrecompileTools

export Vec3
export Mat3
export mpi_nprocs
export mpi_master
export setup_threading, disable_threading
include("common/timer.jl")
include("common/ortho.jl")
include("common/types.jl")
include("common/spherical_bessels.jl")
include("common/spherical_harmonics.jl")
include("common/split_evenly.jl")
include("common/mpi.jl")
include("common/threading.jl")
include("common/printing.jl")
include("common/cis2pi.jl")
include("architecture.jl")
include("common/zeros_like.jl")
include("common/norm.jl")

export PspHgh
export PspUpf
include("pseudo/NormConservingPsp.jl")
include("pseudo/PspHgh.jl")
include("pseudo/PspUpf.jl")

export ElementPsp
export ElementCohenBergstresser
export ElementCoulomb
export ElementGaussian
export charge_nuclear
export charge_ionic
export atomic_symbol
export n_elec_valence
export n_elec_core
include("elements.jl")

export SymOp
include("SymOp.jl")

export Smearing
export Model
export PlaneWaveBasis
export compute_fft_size
export G_vectors, G_vectors_cart, r_vectors, r_vectors_cart
export Gplusk_vectors, Gplusk_vectors_cart
export Kpoint
export ifft
export irfft
export ifft!
export fft
export fft!
export create_supercell
export cell_to_supercell
include("Smearing.jl")
include("Model.jl")
include("structure.jl")
include("PlaneWaveBasis.jl")
include("fft.jl")
include("orbitals.jl")
include("show.jl")
include("supercell.jl")

export Energies
include("Energies.jl")

export Hamiltonian
export HamiltonianBlock
export energy_hamiltonian
export Kinetic
export ExternalFromFourier
export ExternalFromReal
export AtomicLocal
export LocalNonlinearity
export Hartree
export Xc
export AtomicNonlocal
export Ewald
export PspCorrection
export Entropy
export Magnetic
export PairwisePotential
export Anyonic
export apply_kernel
export compute_kernel
export BlowupIdentity
export BlowupCHV
export BlowupAbinit
include("DispatchFunctional.jl")
include("terms/terms.jl")

export AbstractFermiAlgorithm, FermiBisection, FermiTwoStage
include("occupation.jl")
export compute_density
export total_density
export spin_density
export ρ_from_total_and_spin
include("densities.jl")
include("transfer.jl")
include("interpolation.jl")
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
export model_DFT, model_PBE, model_LDA, model_SCAN
include("standard_models.jl")

export KerkerMixing, KerkerDosMixing, SimpleMixing, DielectricMixing
export LdosMixing, HybridMixing, χ0Mixing
export FixedBands, AdaptiveBands
export scf_damping_solver
export scf_anderson_solver
export scf_CROP_solver
export self_consistent_field
export ResponseOptions
export direct_minimization
export newton
export load_scfres, save_scfres
include("scf/chi0models.jl")
include("scf/mixing.jl")
include("scf/scf_solvers.jl")
include("scf/nbands_algorithm.jl")
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

export DensityConstructionMethod
export AtomicDensity
export RandomDensity
export CoreDensity
export ValenceDensityGaussian
export ValenceDensityPseudo
export ValenceDensityAuto
export guess_density
export random_density
include("density_methods.jl")

export load_psp
export list_psp
export attach_psp
include("pseudo/load_psp.jl")
include("pseudo/list_psp.jl")
include("pseudo/attach_psp.jl")

export DFTKPotential
export atomic_system, periodic_system  # Reexport from AtomsBase
export run_wannier90
include("external/atomsbase.jl")
include("external/interatomicpotentials.jl")
include("external/stubs.jl")  # Function stubs for conditionally defined methods

export compute_bands
export plot_bandstructure
export irrfbz_path
include("postprocess/band_structure.jl")

export compute_forces
export compute_forces_cart
include("postprocess/forces.jl")
export compute_stresses_cart
include("postprocess/stresses.jl")
export compute_dos
export compute_ldos
export plot_dos
include("postprocess/dos.jl")
export compute_χ0
export apply_χ0
include("response/cg.jl")
include("response/chi0.jl")
include("response/hessian.jl")
export compute_current
include("postprocess/current.jl")

# Workarounds
include("workarounds/dummy_inplace_fft.jl")
include("workarounds/forwarddiff_rules.jl")
include("workarounds/gpu_arrays.jl")


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
    @require wannier90_jll="c5400fa0-8d08-52c2-913f-1e3f656c1ce9" begin
        include("external/wannier90.jl")
    end
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba"   include("workarounds/cuda_arrays.jl")
    @require AMDGPU="21141c5a-9bdb-4563-92ae-f87d6854732e" include("workarounds/roc_arrays.jl")
end

# Precompilation block with a basic workflow
@setup_workload begin
    # very artificial silicon ground state example
    a = 10.26
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]
    magnetic_moments = [2, -2]

    @compile_workload begin
        model = model_LDA(lattice, atoms, positions;
                          magnetic_moments, temperature=0.1, spin_polarization=:collinear)
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2])
        ρ0 = guess_density(basis, magnetic_moments)
        scfres = self_consistent_field(basis; ρ=ρ0, tol=1e-2, maxiter=3, callback=identity)
    end
end
end # module DFTK
