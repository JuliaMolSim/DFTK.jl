"""
Compute the stresses (= 1/Vol dE/d(M*lattice), taken at M=I) of an obtained SCF solution.
"""
@timing function compute_stresses_cart(scfres)
    # TODO optimize by only computing derivatives wrt 6 independent parameters
    # compute the Hellmann-Feynman energy (with fixed ψ/occ/ρ)
    function HF_energy(lattice::AbstractMatrix{T}) where {T}
        basis = scfres.basis
        new_model = Model(basis.model; lattice)
        new_basis = PlaneWaveBasis(new_model,
                                   basis.Ecut, basis.fft_size, basis.variational,
                                   basis.kgrid, basis.symmetries_respect_rgrid,
                                   basis.use_symmetries_for_kpoint_reduction,
                                   basis.comm_kpts, basis.architecture)
        ρ = compute_density(new_basis, scfres.ψ, scfres.occupation)
        energies = energy_hamiltonian(new_basis, scfres.ψ, scfres.occupation;
                                      ρ, scfres.eigenvalues, scfres.εF).energies
        energies.total
    end
    L = scfres.basis.model.lattice
    Ω = scfres.basis.model.unit_cell_volume
    stresses = ForwardDiff.gradient(M -> HF_energy((I+M) * L), zero(L)) / Ω
    symmetrize_stresses(scfres.basis, stresses)
end
