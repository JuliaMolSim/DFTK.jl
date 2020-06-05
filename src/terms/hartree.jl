"""
Hartree term: for a decaying potential V the energy would be

1/2 ∫ρ(x)ρ(y)V(x-y) dxdy

with the integral on x in the unit cell and of y in the whole space.
For the Coulomb potential with periodic boundary conditions, this is rather

1/2 ∫ρ(x)ρ(y) G(x-y) dx dy

where G is the Green's function of the periodic Laplacian with zero
mean (-Δ G = sum_{R} 4π δ_R, integral of G zero on a unit cell).
"""
struct Hartree
    scaling_factor::Real  # to scale by an arbitrary factor (useful for exploration)
end
Hartree(; scaling_factor=1) = Hartree(scaling_factor)
(hartree::Hartree)(basis) = TermHartree(basis, hartree.scaling_factor)

struct TermHartree <: Term
    basis::PlaneWaveBasis
    scaling_factor::Real
    # Fourier coefficients of the Green's function of the periodic Poisson equation
    poisson_green_coeffs
end
function TermHartree(basis::PlaneWaveBasis{T}, scaling_factor) where T
    # Solving the Poisson equation ΔV = -4π ρ in Fourier space
    # is multiplying elementwise by 4π / |G|^2.
    poisson_green_coeffs = 4T(π) ./ [sum(abs2, basis.model.recip_lattice * G)
                                     for G in G_vectors(basis)]

    # Zero the DC component (i.e. assume a compensating charge background)
    poisson_green_coeffs[1] = 0
    TermHartree(basis, scaling_factor, scaling_factor .* poisson_green_coeffs)
end

function ene_ops(term::TermHartree, ψ, occ; ρ, kwargs...)
    basis = term.basis
    T = eltype(basis)
    pot_fourier = term.poisson_green_coeffs .* ρ.fourier
    potential = real(G_to_r(basis, pot_fourier))  # TODO optimize this
    E = real(dot(pot_fourier, ρ.fourier) / 2)

    ops = [RealSpaceMultiplication(basis, kpoint, potential) for kpoint in basis.kpoints]
    (E=E, ops=ops)
end

function compute_kernel(term::TermHartree; kwargs...)
    vc_G = term.poisson_green_coeffs
    G_to_r_matrix(term.basis) * Diagonal(vec(vc_G)) * r_to_G_matrix(term.basis)
end
apply_kernel(term::TermHartree, dρ; kwargs...) = term.poisson_green_coeffs .* dρ.fourier
