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
    potential = [external.V(basis.model.lattice * r) for r in r_vectors(basis)]
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
    pot_fourier = [complex(external.V(basis.model.recip_lattice * G) / sqrt(unit_cell_volume))
                   for G in G_vectors(basis)]
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

    #
    # pot_fourier is <e_G|V|e_G'> expanded in a basis of e_{G-G'}
    #

    # TODO doc
    # Fourier coefficient of potential for an element el at position r
    # (G in cartesian coordinates)
    pot(el, r, Gcart) = Complex{T}(local_potential_fourier(el, norm(Gcart))
                               * cis(-dot(Gcart, model.lattice * r)))
    pot(Gcart) = sum(pot(elem, r, Gcart)
                     for (elem, positions) in model.atoms
                     for r in positions)

    pot_fourier = [pot(model.recip_lattice * G) / sqrt(model.unit_cell_volume)
                   for G in G_vectors(basis)]
    pot_real = G_to_r(basis, pot_fourier)
    TermAtomicLocal(basis, real(pot_real))
end

function forces(term::TermAtomicLocal, ψ, occ; ρ, kwargs...)
    @timeit to "local" begin
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
                forces[iel][ir] = -real(sum(conj(ρ.fourier[iG])
                                        .* form_factors[iG]
                                        .* cis(-2T(π) * dot(G, r))
                                        .* (-2T(π)) .* G .* im
                                        ./ sqrt(unit_cell_volume)
                                       for (iG, G) in enumerate(G_vectors(term.basis))))
            end
        end
        forces
    end
end
