## Local potentials. Can be provided from external potentials, or from `model.atoms`.

# A local potential term. Must have the field `potential_values`, storing the
# potential in real space on the grid. If the potential is different in the α and β
# components then it should be a 4d-array with the last axis running over the
# two spin components.
abstract type TermLocalPotential <: Term end

@timing "ene_ops: local" function ene_ops(term::TermLocalPotential,
                                          basis::PlaneWaveBasis{T}, ψ, occupation;
                                          kwargs...) where {T}
    potview(data, spin) = ndims(data) == 4 ? (@view data[:, :, :, spin]) : data
    ops = [RealSpaceMultiplication(basis, kpt, potview(term.potential_values, kpt.spin))
           for kpt in basis.kpoints]
    if :ρ in keys(kwargs)
        E = sum(total_density(kwargs[:ρ]) .* term.potential_values) * basis.dvol
    else
        E = T(Inf)
    end

    (; E, ops)
end

## External potentials

struct TermExternal <: TermLocalPotential
    potential_values::AbstractArray
end

"""
External potential from an analytic function `V` (in cartesian coordinates).
No low-pass filtering is performed.
"""
struct ExternalFromReal
    potential::Function
end

function (external::ExternalFromReal)(basis::PlaneWaveBasis{T}) where {T}
    pot_real = external.potential.(r_vectors_cart(basis))
    TermExternal(convert_dual.(T, pot_real))
end

"""
External potential from the (unnormalized) Fourier coefficients `V(G)`
G is passed in cartesian coordinates
"""
struct ExternalFromFourier
    potential::Function
end
function (external::ExternalFromFourier)(basis::PlaneWaveBasis{T}) where {T}
    unit_cell_volume = basis.model.unit_cell_volume
    pot_fourier = map(G_vectors_cart(basis)) do G
        convert_dual(complex(T), external.potential(G) / sqrt(unit_cell_volume))
    end
    enforce_real!(basis, pot_fourier)  # Symmetrize Fourier coeffs to have real iFFT
    TermExternal(irfft(basis, pot_fourier))
end



## Atomic local potential

struct TermAtomicLocal{AT} <: TermLocalPotential
    potential_values::AT
end

"""
Atomic local potential defined by `model.atoms`.
"""
struct AtomicLocal end
function compute_local_potential(basis::PlaneWaveBasis{T}; positions=basis.model.positions) where {T}
    # pot_fourier is <e_G|V|e_G'> expanded in a basis of e_{G-G'}
    # Since V is a sum of radial functions located at atomic
    # positions, this involves a form factor (`local_potential_fourier`)
    # and a structure factor e^{-i G·r}
    model = basis.model
    G_cart = to_cpu(G_vectors_cart(basis))
    # TODO Bring G_cart on the CPU for compatibility with the pseudopotentials which
    #      are not isbits ... might be able to solve this by restructuring the loop


    # Pre-compute the form factors at unique values of |G| to speed up
    # the potential Fourier transform (by a lot). Using a hash map gives O(1)
    # lookup.
    form_factors = IdDict{Tuple{Int,T},T}()  # IdDict for Dual compatability
    for G in G_cart
        q = norm(G)
        for (igroup, group) in enumerate(model.atom_groups)
            if !haskey(form_factors, (igroup, q))
                element = model.atoms[first(group)]
                form_factors[(igroup, q)] = local_potential_fourier(element, q)
            end
        end
    end

    Gs = to_cpu(G_vectors(basis))  # TODO Again for GPU compatibility
    pot_fourier = map(enumerate(Gs)) do (iG, G)
        q = norm(G_cart[iG])
        pot = sum(enumerate(model.atom_groups)) do (igroup, group)
            structure_factor = sum(r -> cis2pi(-dot(G, r)), @view positions[group])
            form_factors[(igroup, q)] * structure_factor
        end
        pot / sqrt(model.unit_cell_volume)
    end

    enforce_real!(basis, pot_fourier)  # Symmetrize Fourier coeffs to have real iFFT
    pot_real = irfft(basis, to_device(basis.architecture, pot_fourier))

    pot_real
end
(::AtomicLocal)(basis::PlaneWaveBasis) = TermAtomicLocal(compute_local_potential(basis))

function compute_δV(::TermAtomicLocal, basis::PlaneWaveBasis{T}, γ, τ;
                    q=zero(Vec3{T}), positions=basis.model.positions) where {T}
    displacement = zero.(positions)
    displacement[τ] = setindex(displacement[τ], one(T), γ)
    ForwardDiff.derivative(zero(T)) do ε
        positions = ε*displacement .+ positions
        compute_local_potential(basis; positions)
    end
end

@timing "forces: local" function compute_forces(term::TermAtomicLocal, basis::PlaneWaveBasis{TT},
                                                ψ, occupation; ρ, kwargs...) where {TT}
    T = promote_type(TT, real(eltype(ψ[1])))
    model = basis.model
    ρ_fourier = fft(basis, total_density(ρ))

    # forces = ∫ρδV
    n_atoms = length(model.positions)
    forces = [zero(Vec3{T}) for _ in 1:n_atoms]
    for τ in 1:n_atoms
        for γ in 1:3
            δV_τγ = compute_δV(term, basis, γ, τ)
            forces[τ] = setindex(forces[τ], -real(sum(conj(ρ_fourier) .* fft(basis, δV_τγ))), γ)
        end
    end
    forces
end

# function barrier to work around various type instabilities
function _force_local_internal(basis, ρ_fourier, form_factors, r)
    T = real(eltype(ρ_fourier))
    f = zero(Vec3{T})
    for (iG, G) in enumerate(G_vectors(basis))
        f -= real(conj(ρ_fourier[iG])
                  .* form_factors[iG]
                  .* cis2pi(-dot(G, r))
                  .* (-2T(π)) .* G .* im
                  ./ sqrt(basis.model.unit_cell_volume))
    end
    f
end
