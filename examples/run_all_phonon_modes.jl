# Minimal example showing how to use all_phonon_modes

using DFTK
using AtomsBuilder  
using PseudoPotentialData

# Setup: Silicon with LDA, 2x2x2 k-grid (same as phonons.jl example)
pseudopotentials = PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth")
model = model_DFT(bulk(:Si); pseudopotentials, functionals=LDA())
basis = PlaneWaveBasis(model; Ecut=10, kgrid=[2, 2, 2])

println("Performing SCF calculation...")
scfres = self_consistent_field(basis, tol=1e-12)
println("✓ SCF converged")

println("\nComputing phonon modes for all q-points using symmetries...")
all_modes = DFTK.all_phonon_modes(scfres)

println("\nResults:")
println("  Total q-points: $(length(all_modes))")
println("\nPhonon modes at each q-point:")
for (i, mode_data) in enumerate(all_modes)
    q = mode_data.q
    freqs = mode_data.modes.frequencies
    println("  q[$i] = $(round.(q, digits=3))")
    println("    First 3 frequencies: $(round.(freqs[1:3], digits=4))")
end

println("\n" * "="^70)
println("Verification: all_phonon_modes(scfres) ≈ all_phonon_modes(unfold_bz(scfres))")
println("="^70)

scfres_unfold = DFTK.unfold_bz(scfres)
all_modes_unfold = DFTK.all_phonon_modes(scfres_unfold)

max_freq_diff = 0.0
for i = 1:length(all_modes)
    freq_diff = maximum(abs.(all_modes[i].modes.frequencies - 
                              all_modes_unfold[i].modes.frequencies))
    max_freq_diff = max(max_freq_diff, freq_diff)
end

println("Maximum frequency difference: $max_freq_diff")
if max_freq_diff < 1e-6
    println("✓ TEST PASSED: Frequencies match to high precision!")
else
    println("✗ TEST FAILED: Frequencies differ by more than 1e-6")
end
