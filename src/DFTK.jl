module DFTK

using DocStringExtensions
using LinearAlgebra
using Markdown
using Printf
using TimerOutputs
using Unitful
using UnitfulAtomic
using ForwardDiff
using AbstractFFTs
using GPUArraysCore
using Random
using ChainRulesCore
using PrecompileTools

@template METHODS =
"""
$(TYPEDSIGNATURES)

$(DOCSTRING)
"""

export Vec3
export Mat3
export mpi_nprocs
export mpi_master
export setup_threading, disable_threading
include("common/timer.jl")
include("common/constants.jl")
include("common/ortho.jl")
include("common/types.jl")
include("common/spherical_bessels.jl")
include("common/spherical_harmonics.jl")
include("common/split_evenly.jl")
include("common/mpi.jl")
include("common/threading.jl")
include("common/printing.jl")
include("common/cis2pi.jl")
include("common/versioninfo.jl")
include("architecture.jl")
include("common/zeros_like.jl")
include("common/norm.jl")
include("common/quadrature.jl")
include("common/hankel.jl")
include("common/hydrogenic.jl")
include("common/derivatives.jl")

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
export atomic_mass
export n_elec_valence
export n_elec_core
include("elements.jl")

export SymOp
include("SymOp.jl")

export Smearing
export Model
export MonkhorstPack, ExplicitKpoints
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
export kgrid_from_maximal_spacing, kgrid_from_minimal_n_kpoints
include("Smearing.jl")
include("Model.jl")
include("structure.jl")
include("bzmesh.jl")
include("PlaneWaveBasis.jl")
include("fft.jl")
include("orbitals.jl")
include("input_output.jl")

export create_supercell
export cell_to_supercell
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
export self_consistent_field, kwargs_scf_checkpoints
export ScfConvergenceEnergy, ScfConvergenceDensity, ScfConvergenceForce
export ScfSaveCheckpoints, ScfDefaultCallback, AdaptiveDiagtol
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
include("symmetry.jl")

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

export atomic_system, periodic_system  # Reexport from AtomsBase
export run_wannier90
export DFTKCalculator
include("external/atomsbase.jl")
include("external/stubs.jl")  # Function stubs for conditionally defined methods
include("external/wannier_shared.jl")
include("external/atoms_calculators.jl")

export compute_bands
export plot_bandstructure
export irrfbz_path
export save_bands
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
export phonon_modes
include("postprocess/phonon.jl")

# Workarounds
include("workarounds/dummy_inplace_fft.jl")
include("workarounds/forwarddiff_rules.jl")
include("workarounds/gpu_arrays.jl")

# Precompilation block with a basic workflow
@setup_workload begin
    # very artificial silicon ground state example
    a = 10.26
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si; psp=load_psp("hgh/lda/Si-q4"))
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
