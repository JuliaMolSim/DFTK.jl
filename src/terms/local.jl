## Local potentials. Can be provided from external potentials, or from `model.atoms`.

# a local potential term. Must have the field `potential_values`, storing the
# potential in real space on the grid. If the potential is different in the α and β
# components then it should be a 4d-array with the last axis running over the
# two spin components.
abstract type TermLocalPotential <: Term end

@timing "ene_ops: local" function ene_ops(term::TermLocalPotential,
                                          basis::PlaneWaveBasis{T}, ψ, occ; kwargs...) where {T}
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

function (external::ExternalFromReal)(basis::PlaneWaveBasis)
    TermExternal(external.potential.(r_vectors_cart(basis)))
end

"""
External potential from the (unnormalized) Fourier coefficients `V(G)`
G is passed in cartesian coordinates
"""
struct ExternalFromFourier
    potential::Function
end
function (external::ExternalFromFourier)(basis::PlaneWaveBasis)
    unit_cell_volume = basis.model.unit_cell_volume
    pot_fourier = [complex(external.potential(G) / sqrt(unit_cell_volume))
                   for G in G_vectors_cart(basis)]
    pot_real = G_to_r(basis, pot_fourier)
    TermExternal(real(pot_real))
end



## Atomic local potential

struct TermAtomicLocal <: TermLocalPotential
    potential_values::AbstractArray
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
    # and a structure factor e^{-i Gr}

    pot_fourier = zeros(Complex{T}, basis.fft_size)
    for (iG, G) in enumerate(G_vectors(basis))
        pot = zero(T)
        for (elem, positions) in model.atoms
            form_factor::T = local_potential_fourier(elem, norm(model.recip_lattice * G))
            for r in positions
                pot += cis(-2T(π) * dot(G, r)) * form_factor
            end
        end
        pot_fourier[iG] = pot / sqrt(model.unit_cell_volume)
    end

    pot_real = G_to_r(basis, pot_fourier)
    TermAtomicLocal(real(pot_real))
end

@timing "forces: local" function compute_forces(::TermAtomicLocal,
                                                basis::PlaneWaveBasis{T},
                                                ψ, occ; ρ, kwargs...) where {T}
    atoms = basis.model.atoms
    recip_lattice = basis.model.recip_lattice
    unit_cell_volume = basis.model.unit_cell_volume
    ρ_fourier = r_to_G(basis, total_density(ρ))

    # energy = sum of form_factor(G) * struct_factor(G) * rho(G)
    # where struct_factor(G) = cis(-2π G⋅r)
    forces = [zeros(Vec3{T}, length(positions)) for (el, positions) in atoms]
    for (iel, (el, positions)) in enumerate(atoms)
        form_factors = [Complex{T}(local_potential_fourier(el, norm(recip_lattice * G)))
                        for G in G_vectors(basis)]

        for (ir, r) in enumerate(positions)
            forces[iel][ir] = _force_local_internal(basis, ρ_fourier, form_factors, r)
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
                  .* cis(-2T(π) * dot(G, r))
                  .* (-2T(π)) .* G .* im
                  ./ sqrt(basis.model.unit_cell_volume))
    end
    f
end
