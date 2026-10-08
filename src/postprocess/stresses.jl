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

!!! info "Stresses are not always symmetric with respect to the physical structure"
    We ensure that the returned stresses are the derivatve of the obtained DFT
    energy with respect to the lattice *within the discretisation* encoded in
    the [`PlaneWaveBasis`](@ref). Note, that as a result we are unable to make sure
    that stresses always keep the symmetries of the physical structure, simply because
    the discretised problem (encoded in the `basis`) may be unable to represent numerically
    all physical symmetries, i.e. `basis.symmetries` may be only a subset
    of `model.symmetries`. Use `symmetrize_stresses(basis, stresses; basis.model.symmetries)`
    to explicitly make sure that the `stresses` returned by this function keep the symmetry
    of the physical model. See also the discussion in [`compute_forces`](@ref).
"""
@timing function compute_stresses_cart(basis, ψ, occupation; eigenvalues, εF, symmetries=basis.symmetries, ρ=nothing)
    # compute the Hellmann-Feynman energy (with fixed ψ/occ/ρ)
    function HF_energy(lattice)
        new_model = Model(basis.model; lattice)
        new_basis = PlaneWaveBasis(basis; model=new_model)
        if isnothing(ρ)
            ρ_strained = compute_density(new_basis, ψ, occupation; symmetries)
        else
            # TODO: horrible hack to avoid symmetrizing with fully reduced IBZ
            # TODO: will not work for τ and USPP/PAW
            # TODO: fix this
            ρ_strained = ρ .* (basis.dvol / new_basis.dvol)
        end
        τ = nothing
        # if any(needs_τ, basis.terms)
        #     τ = compute_kinetic_energy_density(new_basis, ψ, occupation) # TODO symmetries
        # end
        (; energies) = energy(new_basis, ψ, occupation;
                              ρ=ρ_strained, τ, eigenvalues, εF)
        energies.total
    end
    L  = basis.model.lattice
    Ω  = basis.model.unit_cell_volume

    # Note that both strain and stress are symmetric, therefore we only do
    # AD with respect to the 6 free Voigt strain components. Note, that the
    # conversion from Voigt strain to 3x3 strain adds the ones on the diagonal
    f = v -> HF_energy(voigt_strain_to_full(v) * L)
    x = zeros(eltype(L), 6)
    # Use chunk size of 1 to limit memory usage
    config = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{1}())
    stress_voigt = 1/Ω * ForwardDiff.gradient(f, x, config)::Vector{eltype(L)}
    symmetrize_stresses(basis, voigt_stress_to_full(stress_voigt); symmetries)
end
@timing function compute_stresses_cart(scfres; symmetries=scfres.basis.symmetries)
    compute_stresses_cart(scfres.basis, scfres.ψ, scfres.occupation;
                          scfres.eigenvalues, scfres.εF, symmetries, scfres.ρ)
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
function full_strain_to_voigt(ε::AbstractMatrix{T}) where {T}
    @SVector[ε[1, 1] - 1, ε[2, 2] - 1, ε[3, 3] - 1,
             ε[3, 2] + ε[2, 3],
             ε[3, 1] + ε[1, 3],
             ε[1, 2] + ε[2, 1]]
end

function transformation_to_voigt(A::AbstractMatrix{T}) where {T}
    stack(1:6; dims=1) do i
        η = zeros(T, 6)
        η[i] = one(T)
        ϵ = voigt_strain_to_full(η)
        ϵ_transformed = A \ ϵ * A
        full_strain_to_voigt(ϵ_transformed)
    end
end


# Internal function used to not only compute stresses, but also symmetrise with respect
# to a custom set of symmetries. Currently only used in the DFTK calculator; if it becomes
# broadly useful, we should perhaps promote this to a proper API function.
function _compute_stresses_cart_symmetrized(scfres; symmetries)
    stresses = compute_stresses_cart(scfres)
    symmetrize_stresses(scfres.basis, stresses; symmetries)
end
