# To gracefully treat the combination of strain and electric fields,
# we write all internal DFTK calculations in terms of the
# "fully reduced" polarization discussed in the Appendix of
# [Wu et al., Phys. Rev. B 72, 035105 (2005)].
#
# The fully reduced electronic polarization is:
#   Pel_α({uₙₖ}) = -i/2π Σₙₖ wₖfₙₖ <uₙₖ|∂k_α uₙₖ>
# and the fully reduced ionic polarization is: (Rₛ in reduced coordinates)
#   Pion_α = Σₛ Zₛ Rₛ_α.
# The Cartesian polarization is:
#   Pcart = L/Ω P
#
# At nonzero field, the {uₙₖ} minimize the electric enthalpy functional:
#   F({uₙₖ}, E) = Energy({uₙₖ}) - E ⋅ Ptot({uₙₖ}).
# See:
# - perturbative treatment around E=0: Nunes and Gonze, Phys. Rev. B 63, 155107 (2001)
# - nonzero E: Souza et al., Phys. Rev. Lett. 89, 117602 (2002)
#              Umari and Pasquarello, Phys. Rev. Lett. 89, 157602 (2002)
# The reduced polarization Ptot is Pel + Pion above and the
# reduced electric field E is defined such that Ecart = L⁻ᵀ E.
# It holds that E ⋅ Ptot = Ω Ecart ⋅ Pcart; the latter is the usual
# literature definition, but the reduced variants are more natural when
# combining electric fields and strain perturbations.
#
# # Analytic k-derivative
# The k-derivative of uₙₖ can be evaluated analytically via a Sternheimer solve.
# The phase of the k-derivative projected onto the occupied subspace,
# i.e. precisely the term required to evaluate the polarization, is arbitrary in the
# Sternheimer resolution (usually set to zero by projection onto the unoccupied subspace).
# Hence, this approach cannot compute the polarization itself, but it can be used for
# polarization derivatives and for electric field perturbations.
#
# Polarization derivative ∂P_α/∂θ:
# ∂Pel_α({uₙₖ})/∂θ = -i/2π * Σₙₖ wₖfₙₖ (<∂θ uₙₖ|∂k_α uₙₖ> + <uₙₖ|∂k_α∂θ uₙₖ>)
#                 (by parts)
#                 = -i/2π * Σₙₖ wₖfₙₖ (<∂θ uₙₖ|∂k_α uₙₖ> + ∂k_α <uₙₖ|∂θ uₙₖ> - <∂k_α uₙₖ|∂θ uₙₖ>)
#                 (<uₙₖ|∂θ uₙₖ> = 0 from Sternheimer for insulators)
#                 = -i/2π * Σₙₖ wₖfₙₖ (<∂θ uₙₖ|∂k_α uₙₖ> - <∂k_α uₙₖ|∂θ uₙₖ>)
#                 = -i/2π * Σₙₖ wₖfₙₖ (<∂θ uₙₖ|∂k_α uₙₖ> - conj(<∂ θuₙₖ|∂k_α uₙₖ>))
#                 =  1/2π * Σₙₖ wₖfₙₖ 2 Im <∂θ uₙₖ|∂k_α uₙₖ>.
#
# And for an electric field perturbation in direction α the DFPT rhs is
# 1/wₖfₙₖ ∂²F/∂<uₙₖ|∂E_α = -1/wₖfₙₖ ∂Pel_α/∂<uₙₖ| = i/2π ∂k_α uₙₖ.
# So the steps are:
# 1. Compute ∂k_α uₙₖ as response to ∂H/∂k_α (Sternheimer).
# 2. Compute DFPT response to i/2π ∂k_α uₙₖ.
#
# # Numerical k-derivative
# The k-derivative can be evaluated numerically with finite differences, which
# provides access to the polarization itself.
#
# Pick a direction α, denote the corresponding coordinate k∥ and the other two
# directions k⟂. We also consider only occupied bands of occupation fₙₖ = f
# and no symmetries so wₖ = 1/Nk⟂ 1/Nk∥
# Pel_α({uₙₖ}) = -f/2π Σₙₖ wₖfₙₖ <uₙₖ|∂k_α uₙₖ>
#             = -if/2π 1/Nk⟂ Σ_k⟂ Σ_k∥ <uₙₖ|1/Nk∥ ∂k∥ uₙₖ>
#             = -if/2π 1/Nk⟂ Σ_k⟂ Σ_k∥ Tr( U(k)' 1/Nk∥ ∂k∥ U(k))
# Approximate: (the Tr is purely imaginary, so keep only the Im part of the FD approximation)
# Σ_k∥ Tr( U(k)' 1/Nk∥ ∂k∥ U(k)) ≈ Σ_k∥ i Im Tr( U(k)' U(k+Δk∥) - U(k)' U(k))
#                                = Σ_k∥ i Im Tr( U(k)' U(k+Δk∥) - I)
#                                (replace x - 1 ≈ log(x))
#                                ≈ Σ_k∥ i Im Tr log U(k)' U(k+Δk∥)
#                                (Tr log = log det)
#                                = Σ_k∥ i Im log det U(k)' U(k+Δk∥)
# So the full formula is:
# Pel_α({uₙₖ}) = f/2π 1/Nk⟂ Σ_k⟂ Σ_k∥ Im log det U(k)' U(k+Δk∥)
# which is invariant under rotation of any U.
#
# See: King-Smith and Vanderbilt, Phys. Rev. B 47, 1651(R) (1993)
# TODO: numerical k-derivatives are not yet implemented but will soon be!

@doc raw"""
Compute the k-derivative of the Hamiltonian H_k in direction δk.
The passed δk must be in reduced coordinates!
Returns the contraction of the k-derivative with the wavefunctions u_{nk}.
"""
function compute_Hk_derivative(basis::PlaneWaveBasis, ψ, δk; ρ=nothing, τ=nothing)
    basis.use_symmetries_for_kpoint_reduction && error(
        "k-derivatives do not yet support symmetry-reduced k-point grids. " *
        "Pass `use_symmetries_for_kpoint_reduction=false` when constructing " *
        "the PlaneWaveBasis to use the full grid.")
    function apply_kshift(ε::T) where {T}
        kcoords = map(basis.kcoords_global) do kcoord
            kcoord + ε * δk
        end
        kgrid_shifted = ExplicitKpoints(kcoords, convert.(T, basis.kweights))
        model = Model(basis.model; lattice=convert.(T, basis.model.lattice))
        basis_shifted = PlaneWaveBasis(model; basis.Ecut, kgrid=kgrid_shifted,
                                       basis.variational, basis.fft_size,
                                       basis.symmetries_respect_rgrid,
                                       basis.use_symmetries_for_kpoint_reduction,
                                       basis.comm_kpts, basis.architecture)
        ρ_new = isnothing(ρ) ? nothing : convert.(T, ρ)
        τ_new = isnothing(τ) ? nothing : convert.(T, τ)
        ham = Hamiltonian(basis_shifted; ρ=ρ_new, τ=τ_new)
        ham * ψ
    end
    ForwardDiff.derivative(apply_kshift, zero(eltype(basis)))
end

@doc raw"""
Compute the derivative of the (periodic part `u` of the) orbitals `ψ` with
respect to the wavevector change `δk` in reduced coordinates.
"""
function compute_k_derivative(ham, ψ, occupation, εF, eigenvalues, δk;
                              ρ=nothing, τ=nothing, occupation_threshold, tol,
                              bandtolalg=BandtolBalanced(ham.basis, ψ, occupation; occupation_threshold),
                              verbose=true)
    check_full_or_zero_occupation(ham.basis.model, occupation; occupation_threshold)
    verbose && mpi_master(ham.basis.comm_kpts) && println("Computing k-derivative in direction $δk")
    δHkψ = compute_Hk_derivative(ham.basis, ψ, δk; ρ, τ)
    apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, δHkψ;
                occupation_threshold, bandtolalg, tol).δψ
end

@doc raw"""
Compute the derivative of the (periodic part `u` of the) orbitals `ψ` with
respect to `k` in all 3 reduced directions.
"""
function compute_k_derivatives(ham, ψ, occupation, εF, eigenvalues; kwargs...)
    map(1:3) do α
        δk = zeros(eltype(ham.basis), 3)
        δk[α] = 1
        compute_k_derivative(ham, ψ, occupation, εF, eigenvalues, δk; kwargs...)
    end
end
function compute_k_derivatives(scfres; tol, verbose=true)
    compute_k_derivatives(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF, scfres.eigenvalues;
                          scfres.ρ, scfres.τ, scfres.occupation_threshold, tol, verbose)
end

"""
Compute the response of the electronic polarization to a perturbation `δψ`
of the orbitals. This also requires the k-derivatives `δψδks`,
which can be obtained with [`compute_k_derivatives`](@ref).
"""
function electronic_polarization_derivative(basis::PlaneWaveBasis{T}, occupation,
                                            δψ, δψδks) where {T}
    length(δψδks) == 3 || error("Expected `δψδks` of length 3, got length $(length(δψδks))")
    δP = Vec3{T}(map(1:3) do α
        δPα = map(occupation, δψ, δψδks[α]) do occk, δψk, δψkδkα
            1/T(π) * sum(occk .* imag.(columnwise_dots(δψk, δψkδkα)))
        end
        weighted_ksum(basis, δPα)
    end)
    # TODO: implement symmetrization once we support k-point reduction
    δP
end
