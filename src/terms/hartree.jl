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
(hartree::Hartree)(basis)   = TermHartree(basis, hartree.scaling_factor)
function Base.show(io::IO, hartree::Hartree)
    fac = isone(hartree.scaling_factor) ? "" : ", scaling_factor=$scaling_factor"
    print(io, "Hartree($fac)")
end

struct TermHartree <: TermNonlinear
    scaling_factor::Real  # scaling factor, absorbed into poisson_green_coeffs
    # Fourier coefficients of the Green's function of the periodic Poisson equation
    poisson_green_coeffs::AbstractArray
end
function TermHartree(basis::PlaneWaveBasis{T}, scaling_factor) where {T}
    model = basis.model

    # Solving the Poisson equation ΔV = -4π ρ in Fourier space
    # is multiplying elementwise by 4π / |G|^2.
    poisson_green_coeffs = 4T(π) ./ [sum(abs2, G) for G in G_vectors_cart(basis)]
    assert_consistent_electrostatics(model)
    poisson_green_coeffs[1] = 0  # Compensating charge background => Zero DC
    force_real!(basis, poisson_green_coeffs)  # Symmetrize Fourier coeffs to have real iFFT

    TermHartree(T(scaling_factor), T(scaling_factor) .* poisson_green_coeffs)
end

@timing "ene_ops: hartree" function ene_ops(term::TermHartree, basis::PlaneWaveBasis{T},
                                            ψ, occupation; ρ, kwargs...) where {T}
    ρtot_fourier = fft(basis, total_density(ρ))
    pot_fourier = term.poisson_green_coeffs .* ρtot_fourier
    pot_real = irfft(basis, pot_fourier)
    E = real(dot(pot_fourier, ρtot_fourier) / 2)

    ops = [RealSpaceMultiplication(basis, kpt, pot_real) for kpt in basis.kpoints]
    (E=E, ops=ops)
end

function compute_kernel(term::TermHartree, basis::PlaneWaveBasis; kwargs...)
    vc_G = term.poisson_green_coeffs
    # Note that `real` here: if omitted, will result in high-frequency noise of even FFT grids
    K = real(ifft_matrix(basis) * Diagonal(vec(vc_G)) * fft_matrix(basis))
    basis.model.n_spin_components == 1 ? K : [K K; K K]
end

function apply_kernel(term::TermHartree, basis::PlaneWaveBasis, δρ; kwargs...)
    δV = zero(δρ)
    δρtot = total_density(δρ)
    # note broadcast here: δV is 4D, and all its spin components get the same potential
    δV .= irfft(basis, term.poisson_green_coeffs .* fft(basis, δρtot))
end
