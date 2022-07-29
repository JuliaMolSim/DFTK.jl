"""
Compute the stresses (= 1/Vol dE/d(M*lattice), taken at M=I) of an obtained SCF solution.
"""
@timing function compute_stresses_cart(scfres)
    # TODO optimize by only computing derivatives wrt 6 independent parameters
    # compute the Hellmann-Feynman energy (with fixed ψ/occ/ρ)
    function HF_energy(lattice::AbstractMatrix{T}) where T
        # Update lattice (also changes floating-point type)
        new_basis = PlaneWaveBasis(scfres.basis, lattice)
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
