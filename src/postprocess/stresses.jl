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
    L  = scfres.basis.model.lattice
    Ω  = scfres.basis.model.unit_cell_volume
    stresses = 1/Ω * ForwardDiff.gradient(zeros(eltype(L), 6)) do M
        D = [1+M[1]   M[6]   M[5];  # Lattice distortion matrix
               M[6] 1+M[2]   M[4];
               M[5]   M[4] 1+M[3]]
        HF_energy(D * L)
    end
    symmetrize_stresses(scfres.basis, voigt_to_full(stresses))
end
function voigt_to_full(v::AbstractVector{T}) where {T}
    [v[1]       v[6]/T(2)  v[5]/T(2);
     v[6]/T(2)  v[2]       v[4]/T(2);
     v[5]/T(2)  v[4]/T(2)  v[3]     ]
end
function full_to_voigt(A::AbstractMatrix{T}) where {T}
    [A[1, 1], A[2, 2], A[3, 3],
     A[3, 2] + A[2, 3],
     A[3, 1] + A[1, 3],
     A[1, 2] + A[2, 1]]
end
