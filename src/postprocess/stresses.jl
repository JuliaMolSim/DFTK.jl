@doc raw"""
Compute the stresses of an obtained SCF solution. The stress tensor
is given by
```math
\left( \begin{array}{ccc}
σ_{xx} σ_{xy} σ_{xz} \\
σ_{yx} σ_{yy} σ_{yz} \\
σ_{zx} σ_{zy} σ_{zz}
\end{array}
\right) = \frac{1}{|Ω|} \left. \frac{dE[ (I+ϵ) * L]}{dϵ}\right|_{ϵ=0}
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
        (; energies) = energy(new_basis, scfres.ψ, scfres.occupation;
                              ρ, scfres.eigenvalues, scfres.εF)
        energies.total
    end
    L  = scfres.basis.model.lattice
    Ω  = scfres.basis.model.unit_cell_volume

    # Define f(ϵ) = E[ (I+ϵ) * L]. Since the strain is symmetric (same as σ) it has only
    # 6 free components which we collect as in Voigt notation
    #    M = [ϵ_{xx} ϵ_{yy} ϵ_{zz} ϵ_{zy} ϵ_{zx} ϵ_{yx}]
    # Then
    function HF_energy_voigt(M)
        D = [1+M[1]   M[6]   M[5];  # Lattice distortion matrix
               M[6] 1+M[2]   M[4];
               M[5]   M[4] 1+M[3]]
        HF_energy(D * L)
    end
    # The derivative of this function wrt. M is by the chain rule
    #    [df/dϵ_{xx}, df/dϵ_{yy}, df/dϵ_{zz},
    #     df/dϵ_{zy}+df/dϵ_{yz}, df/dϵ_{zx}+df/dϵ_{xz}, df/dϵ_{yx}+df/dϵ_{xy}]
    # Therefore
    stress_voigt = 1/Ω * ForwardDiff.gradient(HF_energy_voigt, zeros(eltype(L), 6))
    symmetrize_stresses(scfres.basis, voigt_to_full(stress_voigt))
end
function voigt_to_full(v::AbstractVector{T}) where {T}
    @SArray[v[1]       v[6]/T(2)  v[5]/T(2);
            v[6]/T(2)  v[2]       v[4]/T(2);
            v[5]/T(2)  v[4]/T(2)  v[3]     ]
end
function full_to_voigt(ε::AbstractMatrix{T}) where {T}
    @SVector[ε[1, 1], ε[2, 2], ε[3, 3],
             ε[3, 2] + ε[2, 3],
             ε[3, 1] + ε[1, 3],
             ε[1, 2] + ε[2, 1]]
end
