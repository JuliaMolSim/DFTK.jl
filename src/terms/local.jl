## Local potentials. Can be provided from external potentials, or from `model.atoms`.

# a local potential term. Must have the field `potential_values`, storing the
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

    (E=E, ops=ops)
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
    force_real!(basis, pot_fourier)  # Symmetrize Fourier coeffs to have real iFFT
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
function (::AtomicLocal)(basis::PlaneWaveBasis{T}) where {T}
    model = basis.model
    # pot_fourier is <e_G|V|e_G'> expanded in a basis of e_{G-G'}
    # Since V is a sum of radial functions located at atomic
    # positions, this involves a form factor (`local_potential_fourier`)
    # and a structure factor e^{-i G·r}

    pot_fourier = map(G_vectors(basis)) do G
        pot = sum(model.atom_groups) do group
            element = model.atoms[first(group)]
            form_factor::T = local_potential_fourier(element, norm(model.recip_lattice * G))
            form_factor * sum(cis2pi(-dot(G, r)) for r in @view model.positions[group])
        end
        pot / sqrt(model.unit_cell_volume)
    end
    force_real!(basis, pot_fourier)  # Symmetrize Fourier coeffs to have real iFFT

    pot_real = irfft(basis, pot_fourier)
    TermAtomicLocal(pot_real)
end

@timing "forces: local" function compute_forces(::TermAtomicLocal, basis::PlaneWaveBasis{TT},
                                                ψ, occupation; ρ, kwargs...) where {TT}
    T = promote_type(TT, real(eltype(ψ[1])))
    model = basis.model
    recip_lattice = model.recip_lattice
    ρ_fourier = fft(basis, total_density(ρ))

    # energy = sum of form_factor(G) * struct_factor(G) * rho(G)
    # where struct_factor(G) = e^{-i G·r}
    forces = [zero(Vec3{T}) for _ in 1:length(model.positions)]
    for group in model.atom_groups
        element = model.atoms[first(group)]
        form_factors = [Complex{T}(local_potential_fourier(element, norm(G)))
                        for G in G_vectors_cart(basis)]
        for idx in group
            r = model.positions[idx]
            forces[idx] = _force_local_internal(basis, ρ_fourier, form_factors, r)
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
