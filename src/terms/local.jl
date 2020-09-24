## Local potentials. Can be provided from external potentials, or from `model.atoms`.

# a local potential term. Must have the field `potential`, storing the
# potential in real space on the grid
abstract type TermLocalPotential <: Term end

function ene_ops(term::TermLocalPotential, ψ, occ; kwargs...)
    basis = term.basis
    T = eltype(basis)
    ops = [RealSpaceMultiplication(basis, kpoint, term.potential) for kpoint in basis.kpoints]
    if :ρ in keys(kwargs)
        dVol = basis.model.unit_cell_volume / prod(basis.fft_size)
        E = dVol * sum(kwargs[:ρ].real .* term.potential)
    else
        E = T(Inf)
    end

    (E=E, ops=ops)
end

## External potentials

struct TermExternal <: TermLocalPotential
    basis::PlaneWaveBasis
    potential::AbstractArray
end

"""
External potential from an analytic function `V` (in cartesian coordinates).
No low-pass filtering is performed.
"""
struct ExternalFromReal{T <: Function}
    V::T
end

function (external::ExternalFromReal)(basis::PlaneWaveBasis{T}) where {T}
    potential = external.V.(r_vectors_cart(basis))
    TermExternal(basis, potential)
end

"""
External potential from the (unnormalized) Fourier coefficients `V(G)`
G is passed in cartesian coordinates
"""
struct ExternalFromFourier{T <: Function}
    V::T
end
function (external::ExternalFromFourier)(basis::PlaneWaveBasis)
    unit_cell_volume = basis.model.unit_cell_volume
    pot_fourier = [complex(external.V(G) / sqrt(unit_cell_volume))
                   for G in G_vectors_cart(basis)]
    pot_real = G_to_r(basis, pot_fourier)
    TermExternal(basis, real(pot_real))
end



## Atomic local potential

struct TermAtomicLocal <: TermLocalPotential
    basis::PlaneWaveBasis
    potential::AbstractArray
end

"""
Atomic local potential defined by `model.atoms`.
"""
struct AtomicLocal end
function (E::AtomicLocal)(basis::PlaneWaveBasis{T}) where {T}
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
    TermAtomicLocal(basis, real(pot_real))
end

@timing "forces_local" function forces(term::TermAtomicLocal, ψ, occ; ρ, kwargs...)
    T = eltype(term.basis)
    atoms = term.basis.model.atoms
    recip_lattice = term.basis.model.recip_lattice
    unit_cell_volume = term.basis.model.unit_cell_volume

    # energy = sum of form_factor(G) * struct_factor(G) * rho(G)
    # where struct_factor(G) = cis(-2π G⋅r)
    forces = [zeros(Vec3{T}, length(positions)) for (el, positions) in atoms]
    for (iel, (el, positions)) in enumerate(atoms)
        form_factors = [Complex{T}(local_potential_fourier(el, norm(recip_lattice * G)))
                        for G in G_vectors(term.basis)]

        for (ir, r) in enumerate(positions)
            forces[iel][ir] = _force_local_internal(term.basis, ρ.fourier, form_factors, r)
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
