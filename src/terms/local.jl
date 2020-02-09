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
term_name(term::TermExternal) = "External"

"""
External potential from an analytic function `V` (in non-reduced coordinates).
No low-pass filtering is performed.
"""
struct ExternalFromReal{T <: Function}
    V::T
end

function (E::ExternalFromReal)(basis::PlaneWaveBasis{T}) where {T}
    potential = [E.V(basis.model.lattice * r) for r in r_vectors(basis)]
    TermExternal(basis, potential)
end

"""
External potential from the (unnormalized) Fourier coefficients `V(G)`
"""
struct ExternalFromFourier{T <: Function}
    V::T
end
function (E::ExternalFromFourier)(basis::PlaneWaveBasis)
    pot_fourier = [complex(E.V(basis.model.recip_lattice * G) / sqrt(basis.model.unit_cell_volume))
                   for G in G_vectors(basis)]
    pot_real = G_to_r(basis, pot_fourier)
    TermExternal(basis, real(pot_real))
end



## Atomic local potential

struct TermAtomicLocal <: TermLocalPotential
    basis::PlaneWaveBasis
    potential::AbstractArray
end
term_name(term::TermAtomicLocal) = "Atomic local"

"""
Atomic local potential defined by `model.atoms`.
"""
struct AtomicLocal end
function (E::AtomicLocal)(basis::PlaneWaveBasis{T}) where {T}
    model = basis.model

    # TODO doc
    # Fourier coefficient of potential for an element el at position r (G in cartesian coordinates)
    pot(el, r, G) = Complex{T}(
        local_potential_fourier(el, norm(G))
        * cis(-dot(G, model.lattice * r)))
    pot(G) = sum(pot(elem, r, G)
                 for (elem, positions) in model.atoms
                 for r in positions)

    pot_fourier = [pot(basis.model.recip_lattice * G) / sqrt(basis.model.unit_cell_volume)
                   for G in G_vectors(basis)]
    pot_real = G_to_r(basis, pot_fourier)
    TermAtomicLocal(basis, real(pot_real))
end

function forces(term::TermAtomicLocal, ψ, occ; ρ, kwargs...)
    T = eltype(term.basis)
    atoms = term.basis.model.atoms
    f = [zeros(Vec3{T}, length(positions)) for (el, positions) in atoms]
    # energy = sum of form_factor(G) * struct_factor(G) * rho(G)
    # where struct_factor(G) = cis(-2π G⋅r)
    for (iel, (el, positions)) in enumerate(atoms)
        form_factors = [Complex{T}(local_potential_fourier(el, norm(term.basis.model.recip_lattice * G)))
                        for G in G_vectors(term.basis)]

        for (ir, r) in enumerate(positions)
            f[iel][ir] = -real(sum(conj(ρ.fourier[iG]) .*
                                   form_factors[iG] .*
                                   cis(-2T(π) * dot(G, r)) .*
                                   (-2T(π)) .* G .* im ./
                                   sqrt(term.basis.model.unit_cell_volume)
                                   for (iG, G) in enumerate(G_vectors(term.basis))))
        end
    end
    f
end
