# Simple test of all_phonon_modes function

using AtomsBuilder
using DFTK
using PseudoPotentialData
using LinearAlgebra

# Setup: same as phonons.jl - LDA silicon, 2x2x2 grid
pseudopotentials = PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth")
model  = model_DFT(bulk(:Si); pseudopotentials, functionals=LDA())

# Use a very small cutoff and small grid for fast testing
println("Running SCF...")
basis  = PlaneWaveBasis(model; Ecut=3, kgrid=[2, 2, 2])
scfres = self_consistent_field(basis, tol=1e-6)

println("SCF converged")
println("Number of symmetries: ", length(basis.symmetries))
println("Number of k-points: ", length(basis.kpoints))

# Just compute a single phonon mode to test the infrastructure
println("\nTesting single phonon mode computation...")
ph_modes = DFTK.phonon_modes(scfres; q=[0.0, 0.0, 0.0])
println("Computed phonon frequencies: ", ph_modes.frequencies[1:3])
println("Success!")

