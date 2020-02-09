"""
Hartree term: for a decaying potential V the energy would be

1/2 ∫ρ(x)ρ(y)V(x-y) dxdy

with the integral on x in the unit cell and of y in the whole space.
For the Coulomb potential with periodic boundary conditions, this is rather

1/2 ∫ρ(x)ρ(y) G(x-y) dx dy

where G is the Green's function of the periodic Laplacian with zero
mean (-Δ G = sum_{R} 4π δ_R, integral of G zero on a unit cell).
"""
struct Hartree end
(H::Hartree)(basis) = TermHartree(basis)

struct TermHartree <: Term
    basis::PlaneWaveBasis
    # Fourier coefficients of the Green's function of the periodic Poisson equation
    poisson_green_coeffs
end
function TermHartree(basis::PlaneWaveBasis{T}) where T
    # Solving the Poisson equation ΔV = -4π ρ in Fourier space
    # is multiplying elementwise by 4π / |G|^2.
    poisson_green_coeffs = 4T(π) ./ [sum(abs2, basis.model.recip_lattice * G)
                                     for G in G_vectors(basis)]
    # Zero the DC component (i.e. assume a compensating charge background)
    poisson_green_coeffs[1] = 0
    TermHartree(basis, poisson_green_coeffs)
end
term_name(term::TermHartree) = "Hartree"

function ene_ops(term::TermHartree, ψ, occ; ρ, kwargs...)
    basis = term.basis
    T = eltype(basis)
    pot_fourier = term.poisson_green_coeffs .* ρ.fourier
    potential = real(G_to_r(basis, pot_fourier)) # TODO optimize this
    E = real(dot(pot_fourier, ρ.fourier) / 2)

    ops = [RealSpaceMultiplication(basis, kpoint, potential) for kpoint in basis.kpoints]
    (E=E, ops=ops)
end
