# Berry phase electronic polarization in direction α:
#   Pel,cart_α({uₙₖ}) = -2i/(2π)^3 Σₙ ∫ <uₙₖ|∂kcart_α uₙₖ>
#                    = -i/Ω * Σₙₖ wₖfₙₖ <uₙₖ|∂kcart_α uₙₖ>,
# where the derivative ∂kcart_α is computed in Cartesian coordinates.
# In reduced coordinates, k = 1/2π L^T kcart hence
#   Pel,cart_α({uₙₖ}) = Σ_β L_αβ / Ω * [ -i/2π Σₙₖ wₖfₙₖ <uₙₖ|∂k_α uₙₖ> ]
#
# To gracefully treat the combination of strain and electric fields,
# we write all internal DFTK calculations in terms of the
# "fully reduced" polarization discussed in the Appendix of
# [Wu et al., Phys. Rev. B 72, 035105 (2005)].
# The reduced electronic polarization is:
#   Pel_α({uₙₖ}) = -i/2π Σₙₖ wₖfₙₖ <uₙₖ|∂k_α uₙₖ>
# and the reduced ionic polarization is: (Rₛ in reduced coordinates)
#   Pion_α = Σₛ Zₛ Rₛ_α.
# The Cartesian polarization is:
#   Pcart = L/Ω P
#
# TODO: references to Nunes-Gonze, Souza-Vanderbilt, Umari-Pasquarello
# At nonzero field, the {uₙₖ} minimize the electric enthalpy functional:
#   F({uₙₖ}, E) = Energy({uₙₖ}) - E ⋅ Ptot({uₙₖ}).
# The reduced polarization Ptot is defined above and the
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
# The DFPT rhs for an electric field perturbation in direction α is
# 1/wₖfₙₖ ∂²F/∂<uₙₖ|∂E_α = -1/wₖfₙₖ ∂Pel_α/∂<uₙₖ| = i/2π ∂k_α uₙₖ.
# So the steps are:
# 1. Compute ∂k_α|uₙₖ> as response to ∂H/∂k_α (Sternheimer).
# 2. Compute DFPT response to i/2π ∂k_α uₙₖ.
# 3. Evaluate required derivative of quantity wrt. uₙₖ with AD
#
# For the special case of a polarization derivative ∂P_α/∂θ, step 3 is replaced by:
# ∂Pel_α({uₙₖ})/∂θ = -i/2π * Σₙₖ wₖfₙₖ (<∂θ uₙₖ|∂k_α uₙₖ> + <uₙₖ|∂k_α∂θ uₙₖ>)
#                 (by parts wrt k because the sum approximates an integral over k)
#                 = -i/2π * Σₙₖ wₖfₙₖ (<∂θ uₙₖ|∂k_α uₙₖ> - <∂k_α uₙₖ|∂θ uₙₖ>)
#                 = -i/2π * Σₙₖ wₖfₙₖ (<∂θ uₙₖ|∂k_α uₙₖ> - conj(<∂ θuₙₖ|∂k_α uₙₖ>))
#                 =  1/2π * Σₙₖ wₖfₙₖ 2 Im <∂θ uₙₖ|∂k_α uₙₖ>.
# TODO: the integration by part is likely not even necessary for this argument
# TODO: replace by <uₙₖ|∂k_α∂θ uₙₖ> = ∂k_α <uₙₖ|∂θ uₙₖ> - <∂k_α uₙₖ|∂θ uₙₖ>
#       with the former being zero by the gauge choice in the Sternheimer solver.
#
# # Numerical k-derivative
# The k-derivative can be evaluated numerically with finite differences, which
# provides access to the polarization itself as well as its derivatives.
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
# TODO: reference to KS-V

abstract type PolarizationMethod end
@kwdef struct LinearResponsePolarization <: PolarizationMethod
    verbose::Bool = true
    # TODO: not sure this is the best place to specify it?
    tol::Float64 = 1e-10
end

function electronic_polarization(basis, ψ, occupation, ::LinearResponsePolarization;
                                 kwargs...)
    @warn "electronic_polarization(::LinearResponsePolarization) always returns zero"
    zero(Vec3{eltype(basis)})
end
"""
Strip the dual part (with tag `Tg`) of all SCF quantities. Works for both primal and dual inputs.
"""
function _primal_scf_quantities(Tg, basis, ψ, occupation, εF, eigenvalues, ρ, τ)
    basis0 = eltype(basis) <: Dual ? construct_value(basis) : basis
    ψ0 = map(ψ) do ψk
        ForwardDiff.value.(Tg, real.(ψk)) .+ im .* ForwardDiff.value.(Tg, imag.(ψk))
    end
    occupation0  = [ForwardDiff.value.(Tg, occk) for occk in occupation]
    εF0          = ForwardDiff.value(Tg, εF)
    eigenvalues0 = [ForwardDiff.value.(Tg, εk) for εk in eigenvalues]
    ρ0           = isnothing(ρ) ? nothing : ForwardDiff.value.(Tg, ρ)
    τ0           = isnothing(τ) ? nothing : ForwardDiff.value.(Tg, τ)
    (; basis0, ψ0, occupation0, εF0, eigenvalues0, ρ0, τ0)
end

@timing function electronic_polarization(basis::PlaneWaveBasis{T}, ψ, occupation,
                                 lrpol::LinearResponsePolarization;
                                 εF, eigenvalues, occupation_threshold,
                                 ρ=nothing, τ=nothing) where {Tg,V,T<:Dual{Tg,V}}
    (; basis0, ψ0, occupation0, εF0, eigenvalues0, ρ0, τ0) =
        _primal_scf_quantities(Tg, basis, ψ, occupation, εF, eigenvalues, ρ, τ)

    # The polarization is only well-defined for insulators!
    _check_full_occupations(basis0.model, occupation0; occupation_threshold)

    ham = Hamiltonian(basis0; ψ=ψ0, occupation=occupation0, ρ=ρ0, τ=τ0)
    bandtolalg = BandtolBalanced(basis0, ψ0, occupation0; occupation_threshold)

    dψ_dk = map(1:3) do α
        lrpol.verbose && mpi_master(basis0.comm_kpts) && println("Computing dunk/dk_$α")
        δk = zeros(V, 3)
        δk[α] = one(V)
        compute_dψdk(basis0, ψ0, occupation0, εF0, eigenvalues0, δk; ρ=ρ0, τ=τ0,
                     occupation_threshold, lrpol.tol, ham, bandtolalg)
    end
    P = zero(Vec3{V})
    dP = ntuple(ForwardDiff.npartials(T)) do n
        δψ = [ForwardDiff.partials.(Tg, ψk, n) for ψk in ψ]
        δP = Vec3{V}(map(1:3) do α
            δPk = map(occupation0, δψ, dψ_dk[α]) do occk, δψk, dψk_dk
                1/V(π) * sum(occk .* imag.(columnwise_dots(δψk, dψk_dk)))
            end
            weighted_ksum(basis0, δPk)
        end)
        # TODO: missing symmetrization
        δP
    end
    map(P, dP...) do Pα, dPα...
        ForwardDiff.Dual{Tg}(Pα, dPα)
    end
end

function electronic_polarization(scfres, method::PolarizationMethod)
    electronic_polarization(scfres.basis, scfres.ψ, scfres.occupation, method;
                            scfres.εF, scfres.eigenvalues, scfres.occupation_threshold,
                            scfres.ρ, scfres.τ)
end

function ionic_polarization(model::Model{T}) where {T}
    Pion = sum(zip(model.positions, model.atoms)) do (pos, atom)
        pos * charge_ionic(atom)
    end
    # Fix a choice of branch: within [-0.5, 0.5]
    # TODO: this doesn't fix 0.5 itself?
    rem.(Pion, one(T), RoundNearest)
end

function total_polarization(basis::PlaneWaveBasis, ψ, occupation,
                            method::PolarizationMethod; kwargs...)
    ( electronic_polarization(basis, ψ, occupation, method; kwargs...)
    + ionic_polarization(basis.model))
end
function total_polarization(scfres, method::PolarizationMethod)
    electronic_polarization(scfres, method) + ionic_polarization(scfres.basis.model)
end

function total_polarization_cart(basis::PlaneWaveBasis, ψ, occupation,
                                 method::PolarizationMethod; kwargs...)
    model = basis.model
    1/model.unit_cell_volume * model.lattice * total_polarization(basis, ψ, occupation,
                                                                  method; kwargs...)
end
function total_polarization_cart(scfres, method::PolarizationMethod)
    model = scfres.basis.model
    1/model.unit_cell_volume * model.lattice * total_polarization(scfres, method)
end

@doc raw"""
Compute the Jacobian-vector product of H_k u_{nk} wrt. k in direction δk.
The passed δk must be in reduced coordinates!
"""
function compute_δHψδk(basis::PlaneWaveBasis, ψ, δk; ρ=nothing, τ=nothing)
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
        # TODO: what would happen if we were to recompute ρ and τ?
        #       in principle the k derivatives in τ should not change the result, should test this!
        ρ_new = isnothing(ρ) ? nothing : convert.(T, ρ)
        τ_new = isnothing(τ) ? nothing : convert.(T, τ)
        ham = Hamiltonian(basis_shifted; ρ=ρ_new, τ=τ_new)
        ham * ψ
    end
    ForwardDiff.derivative(apply_kshift, zero(eltype(basis)))
end

@doc raw"""
Compute the derivative of the (periodic part `u` of the) orbitals `ψ` with
respect to `k` in the direction `δk` in reduced coordinates.
"""
function compute_δψδk(ham, ψ, occupation, εF, eigenvalues, δk;
                      ρ=nothing, τ=nothing, occupation_threshold, tol,
                      bandtolalg=BandtolBalanced(ham.basis, ψ, occupation; occupation_threshold),
                      verbose=true)
    check_full_or_zero_occupation(ham.basis.model, occupation; occupation_threshold)
    verbose && mpi_master(ham.basis.comm_kpts) && println("Computing dψ/dk in direction $δk")
    δHψδk = compute_δHψδk(ham.basis, ψ, δk; ρ, τ)
    apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, δHψδk;
                occupation_threshold, bandtolalg, tol).δψ
end

@doc raw"""
Compute the derivative of the (periodic part `u` of the) orbitals `ψ` with
respect to `k` in all 3 reduced directions.
"""
function compute_δψδks(ham, ψ, occupation, εF, eigenvalues; kwargs...)
    map(1:3) do α
        δk = zeros(eltype(ham.basis), 3)
        δk[α] = 1
        compute_δψδk(ham, ψ, occupation, εF, eigenvalues, δk; kwargs...)
    end
end
function compute_δψδks(scfres; tol, verbose=true)
    compute_δψδks(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF, scfres.eigenvalues;
                  scfres.ρ, scfres.τ, scfres.occupation_threshold, tol, verbose)
end

"""
Compute the response of the electronic polarization to a perturbation `δψ`
of the orbitals. This also requires the k-derivatives `δψδks`,
which can be obtained with [`compute_δψδks`](@ref).
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
    # TODO: missing symmetrization
    δP
end

# TODO: probably going away!
@kwdef struct BerryPhasePolarization <: PolarizationMethod
end

# TODO: make a version that only computes the polarization in one direction
# and a wrapper that computes it in all 3 directions.

@timing function electronic_polarization(basis::PlaneWaveBasis{T}, ψ, occupation,
                                 ::BerryPhasePolarization; occupation_threshold, kwargs...) where {T}
    if mpi_nprocs(basis.comm_kpts) > 1
        error("Berry phase currently does not support MPI parallelization over k-points.")
    end
    if basis.n_irreducible_kpoints != length(basis.kgrid)
        # TODO: make electronic_polarization fit for symmetry-reduced k-point grids
        error("Berry phase does not yet support symmetry-reduced k-point grids. " *
              "Pass `use_symmetries_for_kpoint_reduction=false` when constructing " *
              "the PlaneWaveBasis to use the full Monkhorst-Pack grid.")
    end
    if !(basis.kgrid isa MonkhorstPack)
        error("Berry phase requires a Monkhorst-Pack k-point grid.")
    end
    if basis.kgrid.kshift != [0, 0, 0]
        error("Berry phase requires an unshifted k-point grid.")
    end
    kgrid = basis.kgrid.kgrid_size::Vec3{Int}

    model = basis.model
    # All k-points must have the same number of occupied orbitals
    filled_occ = filled_occupation(model)
    nbands = div(model.n_electrons, model.n_spin_components * filled_occ)
    for occk in occupation
        # TODO: these checks will require more care with MPI
        all(occk[1:nbands] .>= filled_occ - occupation_threshold) || error(
            "Berry phase requires nbands ($nbands) occupied orbitals, got: $occk")
        all(occk[nbands+1:end] .<= occupation_threshold) || error(
            "Berry phase requires extra bands (beyond $nbands) to have zero occupation, got: $occk")
    end

    P = map(1:3) do α
        Δk = [α == 1, α == 2, α == 3] ./ kgrid
        # TODO: not gonna work for spin...
        index_perp(k_integer) = (α == 1 ? k_integer[2] : k_integer[1],
                                 α == 3 ? k_integer[2] : k_integer[3])

        string_products = ones(complex(T), index_perp(kgrid)...)
        for (kpt, ψk) in zip(basis.kpoints, ψ)
            ikpt_adj, ΔG = find_equivalent_kpt(basis, kpt.coordinate + Δk, kpt.spin)
            kpt_adj = basis.kpoints[ikpt_adj]
            phase_shift = map(r -> cis2pi(-dot(ΔG, r)), r_vectors(basis))

            overlap = zeros(complex(T), nbands, nbands)
            # TODO: switch to Fourier space overlap cause the real space overlap
            # is a lot more expensive and unnecessary!
            ψnk_real = [ifft(basis, kpt, ψk[:, i]) .* phase_shift for i = 1:nbands]
            ψnk_adj_real = [ifft(basis, kpt_adj, ψ[ikpt_adj][:, i]) for i = 1:nbands]

            @views for j = 1:nbands
                for i = 1:nbands
                    overlap[i, j] = dot(ψnk_real[i], ψnk_adj_real[j]) * basis.dvol
                end
            end
            # TODO: quite awful heh
            kcoord_integer = round.(Int, (kpt.coordinate .+ 1) .% 1.0 .* kgrid)
            string_products[index_perp(kcoord_integer .+ 1)...] *= det(overlap)
        end

        string_phases = angle.(string_products)
        # The phase of each string is only defined up to 2π, and we now have to
        # average them. A naive average would only be defined modulo 2π/Nk⟂,
        # while we expect the Berry phase to be defined modulo 2π. We thus
        # fix the phase of each string from a neighboring string before averaging.
        # Align first row to its first entry, then each column from the first row.
        # TODO: how will this work with spin?
        align_phases!(string_phases[1,:])
        for i in axes(string_phases, 2)
            align_phases!(string_phases[:,i])
        end

        @show string_phases * filled_occ / 2π

        total_phase = sum(string_phases) / length(string_phases)
        # Fix a choice of branch: phase between -π and π
        rem2pi(total_phase, RoundNearest) * filled_occ / 2π
    end
    Vec3(P)
end

function align_phases!(phases)
    prev_phase = first(phases)
    for i in eachindex(phases)
        if rem2pi(phases[i] - prev_phase, RoundNearest) > π/2
            # TODO: is this printed reasonably?
            @warn "Large phase difference between neighboring k-point strings,
                   consider using a denser k-mesh."
        end
        prev_phase = phases[i] = prev_phase + rem2pi(phases[i] - prev_phase, RoundNearest)
    end
    if rem2pi(first(phases) - prev_phase, RoundNearest) > π/2
        @warn "Large phase difference between neighboring k-point strings,
               consider using a denser k-mesh."
    end
end
