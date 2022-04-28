using SpecialFunctions
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
function TermHartree(basis::PlaneWaveBasis{T}, scaling_factor) where T
    model = basis.model

    # Solving the Poisson equation ΔV = -4π ρ in Fourier space
    # is multiplying elementwise by 4π / |G|^2.
    poisson_green_coeffs = 4T(π) ./ [sum(abs2, G) for G in G_vectors_cart(basis)]
    if !isempty(model.atoms)
        # Assume positive charge from nuclei is exactly compensated by the electrons
        sum_charges = sum(charge_ionic, model.atoms)
        @assert sum_charges == model.n_electrons
    end
    poisson_green_coeffs[1] = 0  # Compensating charge background => Zero DC

    TermHartree(T(scaling_factor), T(scaling_factor) .* poisson_green_coeffs)
end

# a gaussian of exponent α and integral M
ρref_real(r::T, M=1, α=1) where {T} = M * exp(-T(1)/2 * (α*r)^2) / ((2T(π))^(T(3)/2)) * α^3
# solution of -ΔVref = 4π ρref
function Vref_real(r::T, M=1, α=1) where {T}
    r == 0 && return M * 2 / sqrt(T(pi)) * α / sqrt(2)
    M * erf(α/sqrt(2)*r)/r
end

one_hot(i) = Vec3{Bool}(j == i for j=1:3)
∂f∂α(f, α, r) = ForwardDiff.derivative(ε -> f(r + ε * one_hot(α)), zero(eltype(r)))

function get_center(basis, ρ)
    sumρ = sum(ρ)
    center = map(1:3) do α
        rα = [r[α] for r in r_vectors_cart(basis)]
        sum(rα .* ρ) / sumρ
    end
    Vec3(center...)
end
function get_integral(basis, ρ)
    sum(ρ) * basis.dvol
end
function get_dipole(α, center, basis, ρ)
    rα = [r[α]-center[α] for r in r_vectors_cart(basis)]
    dot(rα, ρ) * basis.dvol
end
function get_variance(center, basis, ρ)
    rr = [sum(abs2, r-center) for r in r_vectors_cart(basis)]
    dot(rr, ρ) * basis.dvol
end

@timing "ene_ops: hartree" function ene_ops(term::TermHartree, basis::PlaneWaveBasis{T},
                                            ψ, occ; ρ, kwargs...) where {T}
    model = basis.model
    ρ_real = total_density(ρ)
    ρ_fourier = r_to_G(basis, ρ_real)
    pot_fourier = term.poisson_green_coeffs .* ρ_fourier
    pot_real = G_to_r(basis, pot_fourier)

    # For isolated systems, the above does not compute a good potential (eg it assumes zero DC component)
    # We correct it by solving -Δ V = 4πρ in two steps: we split ρ into ρref and ρ-ρref,
    # where ρref is a gaussian has the same total charge as ρ.
    # We compute the first potential in real space (explicitly since ρref is known),
    # and the second (short-range) potential in Fourier space
    # Compared to the usual scheme (ρref = 0), this results in a correction potential
    # equal to Vref computed in real space minus Vref computed in Fourier space
    if any(.!model.periodic)
        @assert all(.!model.periodic)
        # determine center and width from density
        # Strictly speaking, these computations should result in extra terms to guarantee energy/ham consistency
        center = get_center(basis, ρ_real)
        ρref_11 = [ρref_real(norm(r - center), model.n_electrons, 1) for r in r_vectors_cart(basis)]
        spread_11 = get_variance(center, basis, ρref_11)
        spread_ρ = get_variance(center, basis, ρ_real)
        α = 1/sqrt(spread_ρ/spread_11)
        # α = 1

        ρrad_fun(r) = ρref_real(norm(r - center), model.n_electrons, α)
        ρrad = ρrad_fun.(r_vectors_cart(basis))


        # at this point we've determined a gaussian of same width and center as the original.
        # Now we get the dipole moments of ρ and match them
        ρders = [∂f∂α.(ρrad_fun, α, r_vectors_cart(basis)) for α=1:3]

        coeffs_ders = map(1:3) do α
            get_dipole(α, center, basis, ρ_real) / get_dipole(α, center, basis, ρders[α])
        end
        ρref = ρrad + sum([coeffs_ders[α]*ρders[α] for α=1:3])

        # compute corresponding solution of -ΔVref = 4π ρref
        Vref_rad_fun(r) = Vref_real(norm(r - center), model.n_electrons, α)
        Vref_rad = Vref_rad_fun.(r_vectors_cart(basis))
        Vref_ders = [∂f∂α.(Vref_rad_fun, α, r_vectors_cart(basis)) for α=1:3]
        Vref = Vref_rad + sum([coeffs_ders[α]*Vref_ders[α] for α=1:3])

        # TODO possibly optimize FFTs here
        Vcorr_real = Vref - G_to_r(basis, term.poisson_green_coeffs .* r_to_G(basis, ρref))
        Vcorr_fourier = r_to_G(basis, Vcorr_real)
        pot_real .+= Vcorr_real
        pot_fourier .+= Vcorr_fourier
    end

    E = real(dot(pot_fourier, ρ_fourier) / 2)

    ops = [RealSpaceMultiplication(basis, kpt, pot_real) for kpt in basis.kpoints]
    (E=E, ops=ops)
end

function compute_kernel(term::TermHartree, basis::PlaneWaveBasis; kwargs...)
    vc_G = term.poisson_green_coeffs
    # Note that `real` here: if omitted, will result in high-frequency noise of even FFT grids
    K = real(G_to_r_matrix(basis) * Diagonal(vec(vc_G)) * r_to_G_matrix(basis))
    basis.model.n_spin_components == 1 ? K : [K K; K K]
end

function apply_kernel(term::TermHartree, basis::PlaneWaveBasis, δρ; kwargs...)
    δV = zero(δρ)
    δρtot = total_density(δρ)
    # note broadcast here: δV is 4D, and all its spin components get the same potential
    δV .= G_to_r(basis, term.poisson_green_coeffs .* r_to_G(basis, δρtot))
end
