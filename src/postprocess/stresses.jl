using ForwardDiff
"""
Compute the stresses of an obtained SCF solution (derivatives of the energy
with respect to lattice), in fractional coordinates.
"""
@timing function compute_stresses(basis::PlaneWaveBasis, ψ, occ; kwargs...)
    # TODO optimize by only computing derivatives wrt 6 independent parameters
    # TODO this is a bit ugly (couples strongly to the structures) because we
    # don't have a good way to say "make the same model/pwbasis except for this"

    # compute the Hellmann-Feynman energy (with fixed ψ/occ/ρ)
    function HF_energy(lattice)
        T = eltype(lattice)
        # mimic Model constructor
        model = basis.model
        new_model = Model(lattice;
                          model.n_electrons,
                          model.atoms,
                          magnetic_moments=[], # not used because we give symmetries explicitly
                          terms=model.term_types,
                          model.temperature,
                          model.smearing,
                          model.spin_polarization,
                          model.symmetries)
        # XXX find a way to make this work
        new_basis = PlaneWaveBasis(new_model,
                                   0, # Ecut
                                   basis.kcoords,
                                   basis.ksymops,
                                   basis.symmetries;
                                   basis.fft_size,
                                   basis.kgrid,
                                   basis.kshift,
                                   basis.comm_kpts)
        ρ = DFTK.compute_density(new_basis, ψ, occ)
        energies, _ = energy_hamiltonian(new_basis, ψ, occupation; ρ=ρ)
        energies.total
    end
    ForwardDiff.gradient(HF_energy, basis.model.lattice)
end
function compute_stresses(scfres)
    compute_stresses(scfres.basis, scfres.ψ, scfres.occupation)
end
