@testitem "Energy from orbital eigenvalues"  setup=[TestCases] begin
using DFTK

function total_energy_from_eigenvalues(energies::Energies, ham::Hamiltonian,
                                       ρ, eigenvalues, occupation)
    # Orbital energies take care of electron-electron and electron-nuclear
    # interaction energies, but suffer from double counting for the Hartree
    # and from introducing a ∫ρ vxc contribution for the XC term, which is
    # not the XC energy.
    data = map(eigenvalues, occupation) do εk, occk
        sum(εk .* occk)
    end
    sum_eigenvalues = DFTK.weighted_ksum(ham.basis, data)

    # Keys for the nuclear-nuclear interaction and Entropy term
    sum_energies = 0.0
    for key in ("Ewald", "PspCorrection", "Entropy")
        haskey(energies, key) || continue
        sum_energies += energies[key]
    end

    # Correcting for Hartree double counting and XC energy
    @assert ham.basis.model.n_spin_components == 1
    ρtot = DFTK.total_density(ρ)
    i_xc = findfirst(t -> t isa DFTK.TermXc, ham.basis.terms)
    @assert !isnothing(i_xc)
    pot_xc = ham[1].operators[i_xc].potential
    energy_xcpot = sum(pot_xc .* ρtot) * ham.basis.dvol
    energy_correction = - energies.Hartree + energies.Xc - energy_xcpot

    sum_eigenvalues + sum_energies + energy_correction
end
function total_energy_from_eigenvalues(scfres::NamedTuple)
    total_energy_from_eigenvalues(scfres.energies, scfres.ham, scfres.ρ,
                                  scfres.eigenvalues, scfres.occupation)
end

silicon = TestCases.silicon
model   = model_PBE(silicon.lattice, silicon.atoms, silicon.positions, temperature=1e-2)
basis   = PlaneWaveBasis(model; Ecut=15, kgrid=(1, 2, 3), kshift=(0, 1/2, 0))
scfres  = self_consistent_field(basis; tol=1e-6)

etot_eigenvalues = total_energy_from_eigenvalues(scfres)
@test abs(etot_eigenvalues - scfres.energies.total) < 1e-5
end
