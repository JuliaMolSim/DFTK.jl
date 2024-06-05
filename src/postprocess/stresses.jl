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
where ``ϵ`` is the strain tensor.
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

    # Note that both strain and stress are symmetric, therefore we only do
    # AD with respect to the 6 free Voigt strain components. Note, that the
    # conversion from Voigt strain to 3x3 strain adds the ones on the diagonal
    stress_voigt = 1/Ω * ForwardDiff.gradient(
        v -> HF_energy(voigt_strain_to_full(v) * L), zeros(eltype(L), 6)
    )::Vector{eltype(L)}
    symmetrize_stresses(scfres.basis, voigt_stress_to_full(stress_voigt))
end
function voigt_stress_to_full(v::AbstractVector{T}) where {T}
    @SArray[v[1] v[6] v[5];
            v[6] v[2] v[4];
            v[5] v[4] v[3]]
end
function full_stress_to_voigt(σ::AbstractMatrix{T}) where {T}
    @SVector[σ[1, 1], σ[2, 2], σ[3, 3],
             (σ[3, 2] + σ[2, 3]) / T(2),
             (σ[3, 1] + σ[1, 3]) / T(2),
             (σ[1, 2] + σ[2, 1]) / T(2)]
end
function voigt_strain_to_full(v::AbstractVector{T}) where {T}
    @SArray[1 + v[1]           v[6]/T(2)       v[5]/T(2);
                v[6]/T(2)  1 + v[2]            v[4]/T(2);
                v[5]/T(2)      v[4]/T(2)   1 + v[3]     ]
end
function full_strain_to_voigt(ε::AbstractVector{T}) where {T}
    @SVector[ε[1, 1] - 1, ε[2, 2] - 1, ε[3, 3] - 1,
             ε[3, 2] + ε[2, 3],
             ε[3, 1] + ε[1, 3],
             ε[1, 2] + ε[2, 1]]
end
