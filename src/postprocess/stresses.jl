@doc raw"""
Compute the stresses of an obtained SCF solution. The stress tensor
is given by
```math
\left( \begin{array}{ccc}
σ_{xx} σ_{xy} σ_{xz} \\
σ_{yx} σ_{yy} σ_{yz} \\
σ_{zx} σ_{zy} σ_{zz}
\end{array}
\right) = \frac{1}{|Ω|} \left. \frac{dE[ (I+ϵ) * L]}{dM}\right|_{ϵ=0}
```
where ``ϵ`` is the strain.
See [O. Nielsen, R. Martin Phys. Rev. B. **32**, 3792 (1985)](https://doi.org/10.1103/PhysRevB.32.3792)
for details. In Voigt notation one would use the vector
``[σ_{xx} σ_{yy} σ_{zz} σ_{zy} σ_{zx} σ_{yx}]``.
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
