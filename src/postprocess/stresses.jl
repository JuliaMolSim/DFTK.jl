"""
Compute the stresses (= 1/Vol dE/d(M*lattice), taken at M=I) of an obtained SCF solution.
"""
@timing function compute_stresses_cart(scfres)
    # TODO optimize by only computing derivatives wrt 6 independent parameters
    # compute the Hellmann-Feynman energy (with fixed ψ/occ/ρ)
    function HF_energy(lattice::AbstractMatrix{T}) where T
        basis = scfres.basis
        model = basis.model
        new_model = Model(lattice, model.atoms, model.positions;
                          model.n_electrons,
                          magnetic_moments=[], # not used because symmetries explicitly given
                          terms=model.term_types,
                          model.temperature,
                          model.smearing,
                          model.spin_polarization,
                          model.symmetries)
        new_basis = PlaneWaveBasis(new_model,
                                   basis.Ecut, basis.fft_size, basis.variational,
                                   basis.kcoords_global, basis.kweights_global,
                                   basis.kgrid, basis.kshift, basis.symmetries_respect_rgrid,
                                   basis.comm_kpts)
        ρ = DFTK.compute_density(new_basis, scfres.ψ, scfres.occupation)
        energies, _ = energy_hamiltonian(new_basis, scfres.ψ, scfres.occupation;
                                         ρ, scfres.eigenvalues, scfres.εF)
        energies.total
    end
    L = scfres.basis.model.lattice
    Ω = scfres.basis.model.unit_cell_volume
    stresses = ForwardDiff.gradient(M -> HF_energy((I+M) * L), zero(L)) / Ω
    symmetrize_stresses(scfres.basis, stresses)
end
