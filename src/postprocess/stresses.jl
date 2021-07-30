using ForwardDiff
"""
Compute the stresses (= 1/Vol dE/d(M*lattice), taken at M=I) of an obtained SCF solution.
"""
@timing function compute_stresses(scfres)
    # TODO optimize by only computing derivatives wrt 6 independent parameters
    # compute the Hellmann-Feynman energy (with fixed ψ/occ/ρ)
    function HF_energy(lattice)
        T = eltype(lattice)
        basis = scfres.basis
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
        new_basis = PlaneWaveBasis(new_model,
                                   basis.Ecut, basis.fft_size, basis.variational,
                                   basis.kcoords_global, basis.ksymops_global,
                                   basis.kgrid, basis.kshift, basis.symmetries,
                                   basis.comm_kpts)
        ρ = DFTK.compute_density(new_basis, scfres.ψ, scfres.occupation)
        energies, _ = energy_hamiltonian(new_basis, scfres.ψ, scfres.occupation;
                                         ρ, scfres.eigenvalues, scfres.εF)
        energies.total
    end
    L = scfres.basis.model.lattice
    Ω = scfres.basis.model.unit_cell_volume
    stresses_not_symmetrized = ForwardDiff.gradient(M -> HF_energy((I+M) * L), zero(L)) / Ω
    symmetrize_stresses(L, scfres.basis.symmetries, stresses_not_symmetrized)
end
