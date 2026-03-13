@doc raw"""
Compute the independent-particle susceptibility. Will blow up for large systems.
For non-spin-polarized calculations the matrix dimension is
`prod(basis.fft_size)` × `prod(basis.fft_size)` and
for collinear spin-polarized cases it is
`2prod(basis.fft_size)` × `2prod(basis.fft_size)`.
In this case the matrix has effectively 4 blocks, which are:
```math
\left(\begin{array}{cc}
    (χ_0)_{αα}  & (χ_0)_{αβ} \\
    (χ_0)_{βα}  & (χ_0)_{ββ}
\end{array}\right)
```
"""
function compute_χ0(ham;
                    temperature=ham.basis.model.temperature,
                    smearing=ham.basis.model.smearing)
    # We're after χ0(r,r') such that δρ = ∫ χ0(r,r') δV(r') dr'
    # where (up to normalizations)
    # ρ = ∑_nk f(εnk - εF) |ψnk|^2
    # ∑_nk f(εnk - εF) = N_el
    # Everything is summed on k so we omit it for notational simplicity

    # We differentiate wrt a variation δV of the external potential
    # δρ = ∑_n (f'n δεn |ψn|^2 + 2Re fn ψn* δψn - f'n δεF |ψn|^2
    # with fn = f(εnk - εF), f'n = f'(εnk - εF)
    # δN_el = 0 = ∑_n f'n (δεn - δεF)

    # Now we use from first order perturbation theory
    # δεn = <ψn|δV|ψn>
    # δψn = ∑_{m != n} <ψm|δV|ψn> |ψm> / (εn-εm)

    # for δεF we get, with DOS = -∑_n f'n and LDOS = -∑_n f'n |ψn|^2
    # δεF = 1/DOS ∫ δV(r) LDOS(r)dr

    # for δρ we note ρnm = ψn* ψm, and we get
    # δρ = LDOS δεF + ∑_n f'n <ρn|δV> ρn + ∑_{n,m != n} 2Re fn ρnm <ρmn|δV> / (εn-εm)
    # δρ = LDOS δεF + ∑_n f'n <ρn|δV> ρn + ∑_{n,m != n} (fn-fm)/(εn-εm) ρnm <ρnm|δV>
    # The last two terms merge with the convention that (f(x)-f(x))/(x-x) = f'(x) into
    # δρ = LDOS δεF + ∑_{n,m} (fn-fm) ρnm <ρmn|δV> / (εn-εm)
    # Therefore the kernel is LDOS(r) LDOS(r') / DOS + ∑_{n,m} (fn-fm)/(εn-εm) ρnm(r) ρmn(r')
    basis = ham.basis
    filled_occ = filled_occupation(basis.model)
    n_spin   = basis.model.n_spin_components
    n_fft    = prod(basis.fft_size)
    fermialg = default_fermialg(smearing)

    length(basis.model.symmetries) == 1 || error("Disable symmetries for computing χ0")

    EVs = [eigen(Hermitian(Array(Hk))) for Hk in ham.blocks]
    Es = [EV.values for EV in EVs]
    Vs = [EV.vectors for EV in EVs]
    T  = eltype(basis)
    occupation, εF = compute_occupation(basis, Es, fermialg; temperature, tol_n_elec=10eps(T))

    χ0 = zeros_like(G_vectors(basis), T, n_spin * n_fft, n_spin * n_fft)
    for (ik, kpt) in enumerate(basis.kpoints)
        # The sum-over-states terms of χ0 are diagonal in the spin blocks (no αβ / βα terms)
        # so the spin of the kpt selects the block we are in
        spinrange = kpt.spin == 1 ? (1:n_fft) : (n_fft+1:2n_fft)
        χ0σσ = @view χ0[spinrange, spinrange]

        N = length(G_vectors(basis, basis.kpoints[ik]))
        @assert N < 10_000
        E = Es[ik]
        V = Vs[ik]
        Vr = cat(ifft.(Ref(basis), Ref(kpt), eachcol(V))..., dims=4)
        Vr = reshape(Vr, n_fft, N)
        for m = 1:N, n = 1:N
            enred = (E[n] - εF) / temperature
            @assert occupation[ik][n] ≈ filled_occ * Smearing.occupation(smearing, enred)
            ddiff = Smearing.occupation_divided_difference
            ratio = filled_occ * ddiff(smearing, E[m], E[n], εF, temperature)
            # dvol because inner products have a dvol in them
            # so that the dual gets one : |f> -> <dvol f|
            # can take the real part here because the nm term is complex conjugate of mn
            # TODO optimize this a bit... use symmetry nm, reduce allocs, etc.
            factor = basis.kweights[ik] * ratio * basis.dvol

            @views χ0σσ .+= factor .* real(conj((Vr[:, m] .* Vr[:, m]'))
                                           .*   (Vr[:, n] .* Vr[:, n]'))
        end
    end
    mpi_sum!(χ0, basis.comm_kpts)

    # Add variation wrt εF (which is not diagonal wrt. spin)
    if !is_effective_insulator(basis, Es, εF; temperature, smearing)
        dos  = compute_dos(εF, basis, Es)
        ldos = compute_ldos(εF, basis, Es, Vs)
        χ0 .+= vec(ldos) .* vec(ldos)' .* basis.dvol ./ sum(dos)
    end
    χ0
end


# make ldiv! act as a given function
struct FunctionPreconditioner{T}
    precondition!::T  # precondition!(y, x) applies f to x and puts it into y
end
LinearAlgebra.ldiv!(y::T, P::FunctionPreconditioner, x) where {T} = P.precondition!(y, x)::T
LinearAlgebra.ldiv!(P::FunctionPreconditioner, x) = (x .= P.precondition!(similar(x), x))
precondprep!(P::FunctionPreconditioner, ::Any) = P

struct MaskedOperator{T}
    masked_product!::T  # masked_product!(Ax, x; mask)
end
@timing mul_masked!(Ax, M::MaskedOperator, x; mask) = M.masked_product!(Ax, x; mask)

# Solves (1-P) (H-ε) (1-P) δψ = - (1-P) rhs
# where 1-P is the projector on the orthogonal of ψk
# The solver simultaneously solves for multiple right-hand sides, i.e.:
# (1-P) (H-ε) (1-P) δψ[:, n] = - (1-P) rhs[:, n] for all columns n.
# /!\ It is assumed (and not checked) that ψk'Hk*ψk = Diagonal(εk) (extra states
# included).
function sternheimer_solver(Hk, ψk, ε, rhs;
                            callback=identity,
                            ψk_extra=zeros_like(ψk), εk_extra=zeros_like(ε),
                            Hψk_extra=zeros_like(ψk), tol=1e-9,
                            miniter=1, maxiter=100, δψk0=nothing)
    basis  = Hk.basis
    kpoint = Hk.kpoint

    # Note: to maintain clearer mathematical formulas, all commements assume the problem
    #       is solved band by band, with ψkn = ψk[:, n]. In practice, the problem is solved
    #       for all bands simultaneously for performance reasons.

    # We use a Schur decomposition of the orthogonal of the occupied states
    # into a part where we have the partially converged, non-occupied bands
    # (which are Rayleigh-Ritz wrt to Hk) and the rest.

    # Projectors:
    # projector onto the computed and converged states
    P(ϕ) = ψk * (ψk' * ϕ)
    # projector onto the computed but nonconverged states
    P_extra(ϕ) = ψk_extra * (ψk_extra' * ϕ)
    # projector onto the computed (converged and unconverged) states
    P_computed(ϕ) = P(ϕ) + P_extra(ϕ)
    # Q = 1-P is the projector onto the orthogonal of converged states
    Q(ϕ) = ϕ - P(ϕ)

    # R = 1-P_computed is the projector onto the orthogonal of computed states
    # Implement allocation light, in-place version for performance
    function R!(Rϕ, ϕ)
        mul!(Rϕ, ψk, ψk' * ϕ)  # P(ϕ)
        mul!(Rϕ, ψk_extra, ψk_extra' * ϕ, -1, -1)  # -P(ϕ) - P_extra(ϕ)
        Rϕ .+= ϕ  # R = ϕ -P(ϕ) - P_extra(ϕ)
    end
    R(ϕ) = R!(similar(ϕ), ϕ)

    # We put things into the form
    # δψkn = ψk_extra * αkn + δψknᴿ ∈ Ran(Q)
    # where δψknᴿ ∈ Ran(R).
    # Note that, if ψk_extra = [], then 1-P = 1-P_computed and
    # δψkn = δψknᴿ is obtained by inverting the full Sternheimer
    # equations in Ran(Q) = Ran(R)
    #
    # This can be summarized as the following:
    #
    # <---- P ----><------------ Q = 1-P -----------------
    #              <-- P_extra -->
    # <--------P_computed -------><-- R = 1-P_computed ---
    # |-----------|--------------|------------------------
    # 1     N_occupied  N_occupied + N_extra

    # Define the operator H-ε by its action on a set of vectors ϕ: 
    # (H-ε)ϕ = Hϕ - ϕ * Diagonal(ε). Application can be restriced to a
    #given active range of columns of x, define by a mask.
    function H(ϕ; mask=1:size(ϕ, 2))
        Hϕ = Hk * ϕ[:, mask]
        mul!(Hϕ, ϕ[:, mask], Diagonal(ε[mask]), -1, 1)  # Hϕ - ϕ * Diagonal(ε)
    end

    # 1) solve for δψknᴿ
    # ----------------------------
    # writing αkn as a function of δψknᴿ, we get that δψknᴿ
    # solves the system (in Ran(1-P_computed))
    #
    # R * (H - ε) * (1 - M * (H - ε)) * R * δψknᴿ = R * (1 - M) * b
    #
    # where M = ψk_extra * (ψk_extra' (H-ε) ψk_extra)^{-1} * ψk_extra'
    # is defined above and b is the projection of -rhs onto Ran(Q).
    #
    # ψk_extra are not converged but have been Rayleigh-Ritzed (they are NOT
    # eigenvectors of H) so H(ψk_extra) = ψk_extra' (Hk-ε) ψk_extra should be a
    # real diagonal matrix with εk_extra - ε_l on the diagonal (ε_l is a scalar).
    #
    # When solving for multiple RHS at once, ε is a vector, and the above becomes
    # a 3-tensor: ψmk_extra' (H - ε_l) ψnk_extra = δmn (εk_extra_n - ε_l)
    # Applying M * b becomes:
    # ψkm_extra * 1/(εk_extra_n - ε_l) * ψkn_extra' * b[:, l]
    # 1/(εk_extra_n - ε_l) is stored in matrix form as inv_ψk_exHψk_ex[n, l],
    # and multiplies ψkn_extra' * b[:, l] elementwise when applying M * b.
    inv_ψk_exHψk_ex = 1 ./(real.(εk_extra) .- ε')

    b = -Q(rhs)
    bb = R(b -  Hψk_extra * (inv_ψk_exHψk_ex .* ψk_extra'b))  # R * (1-M) * b

    # Implementation of: R * (H - ε) * (1 - M * (H - ε)) * R * ϕ
    @views function RAR!(RARϕ, ϕ; mask)
        R!(RARϕ[:, mask], ϕ[:, mask])
        HRϕ = H(RARϕ; mask)
        # Schur complement of (1-P) (H-ε) (1-P)
        # with the splitting Ran(1-P) = Ran(P_extra) ⊕ Ran(R)
        # HRϕ[:, n] -= Hψk_extra[:, n] * [1/(εk_extra_n - ε_l)] .* Hψk_extra[:, n]' Rϕ[:, l]
        mul!(HRϕ, Hψk_extra, inv_ψk_exHψk_ex[:, mask] .* Hψk_extra'RARϕ[:, mask], -1, 1)
        R!(RARϕ[:, mask], HRϕ)
    end
    A = MaskedOperator(RAR!)
    precon = PreconditionerTPA(basis, kpoint)
    # First column of ψk as there is no natural kinetic energy.
    # We take care of the (rare) cases when ψk is empty.
    precondprep!(precon, size(ψk, 2) ≥ 1 ? repeat(ψk[:, 1], 1, size(bb, 2)) : nothing)
    @timing function R_ldiv!(x, y)
        R!(x, precon \ R(y))
    end
    xinit = isnothing(δψk0) ? zero(bb) : R(δψk0)
    cg_res = cg!(xinit, A, bb; precon=FunctionPreconditioner(R_ldiv!), tol, proj! =R!,
                callback=info -> callback(merge(info, (; basis, kpoint))),
                miniter, maxiter)
    δψkᴿ = cg_res.x

    # 2) solve for αk now that we know δψkᴿ
    # We again do this for all right-hand sides at once, such that
    # αk[m, l] = 1/(εk_extra_n - ε_l) δnm (ψk_extra[:, n]' 
    #                                      * (b[:, l] - (H - ε_l) * δψkᴿ[:, l])
    # Note that αk is an empty array if there is no extra bands.
    αk = inv_ψk_exHψk_ex .* (ψk_extra' * (b - H(δψkᴿ)))

    δψk = ψk_extra * αk + δψkᴿ

    (; δψk, cg_res.n_iter, cg_res.residual_norms, cg_res.converged, cg_res, tol)
end

# Apply the four-point polarizability operator χ0_4P = -Ω^-1
# Returns (δψ, δocc, δεF) corresponding to a change in *total* Hamiltonian δH
# We start from
# P = f(H-εF) = ∑_n fn |ψn><ψn|, tr(P) = N
# where P is the density matrix, f the occupation function.
# Charge conservation yields δεF as follows:
# δεn = <ψn|δH|ψn>
# 0 = ∑_n fn' (δεn - δεF) determines δεF
# where fn' = f'((εn-εF)/T)/T.

# Then <ψm|δP|ψn> = (fm-fn)/(εm-εn) <ψm|δH|ψn>,
# except for the diagonal which is
# <ψn|δP|ψn> = (fn'-δεF) δεn.

# We want to represent δP with a tuple (δψ, δf). We do *not* impose that
# δψ is orthogonal at finite temperature. A formal differentiation yields
# δP = ∑_n fn (|δψn><ψn| + |ψn><δψn|) + δfn |ψn><ψn|.
# Identifying with <ψm|δP|ψn> we get for the off-diagonal terms
# (fm-fn)/(εm-εn) <ψm|δH|ψn> = fm <δψm|ψn> + fn <ψm|δψn>.
# For the diagonal terms, n==m and we obtain
# 0 = ∑_n Re (fn <ψn|δψn>) + δfn,
# so that a gauge choice has to be made here. We choose to set <ψn|δψn> = 0 and
# δfn = fn' (δεn - δεF) ensures the summation to 0 with the definition of δεF as
# above.

# We therefore need to compute all the δfn: this is done with compute_δocc!.
# Regarding the δψ, they are computed with compute_δψ! as follows. We refer to
# the paper https://arxiv.org/abs/2210.04512 for more details.

# We split the computation of δψn in two contributions:
# for the already-computed states, we add an explicit contribution
# for the empty states, we solve a Sternheimer equation
# (H-εn) δψn = - P_{ψ^⟂} δH ψn

# The off-diagonal explicit term needs a careful consideration of stability.
# Let <ψm|δψn> = αmn <ψm|δH|ψn>. αmn has to satisfy
# fn αmn + fm αnm = ratio = (fn-fm)/(εn-εm)   (*)
# The usual way is to impose orthogonality (=> αmn=-αnm),
# but this means that αmn = 1/(εm-εn), which is unstable
# Instead, we minimize αmn^2 + αnm^2 under the linear constraint (*), which leads to
# αmn = ratio * fn / (fn^2 + fm^2)
# fn αmn = ratio * fn^2 / (fn^2 + fm^2)

# This formula is very nice
# - It gives a vanishing contribution fn αmn for empty states
#   (note that α itself blows up, but it's compensated by fn)
# - In the case where fn=1/0 or fm=0 we recover the same formulas
#   as the ones with orthogonality
# - When n=m it gives the correct contribution
# - It does not blow up for degenerate states
function compute_αmn(fm, fn, ratio)
    ratio == 0 && return ratio
    ratio * fn / (fn^2 + fm^2)
end

function is_effective_insulator(basis::PlaneWaveBasis, eigenvalues, εF::T;
                                atol=eps(T),
                                smearing=basis.model.smearing,
                                temperature=basis.model.temperature) where {T}
    if iszero(temperature) || smearing isa Smearing.None
        return true
    else
        min_enred = minimum(eigenvalues) do εk
            minimum(εnk -> abs(εnk - εF) / temperature, εk)
        end
        min_enred = mpi_min(min_enred, basis.comm_kpts)

        # This is the largest possible value the occupation has in the
        # orbital just above the Fermi level
        max_occupation = Smearing.occupation(smearing, min_enred)
        return max_occupation < atol
    end
end

"""
Compute the derivatives of the occupations (and of the Fermi level).
The derivatives of the occupations are in-place stored in δocc.
The tuple (; δocc, δεF) is returned. It is assumed the passed `δocc`
are initialised to zero.
"""
function compute_δocc!(δocc, basis::PlaneWaveBasis{T}, ψ, εF, ε, δHψ, δtemperature) where {T}
    model = basis.model
    temperature = model.temperature
    smearing = model.smearing
    filled_occ = filled_occupation(model)

    # compute the derivative of
    # occ[k][n] = filled_occ*occupation((εnk-εF)/T)
    δεF = zero(T)
    if !is_effective_insulator(basis, ε, εF; smearing, temperature)
        # First compute δocc without self-consistent Fermi δεF.
        D = zero(T)
        for ik = 1:length(basis.kpoints), (n, εnk) in enumerate(ε[ik])
            δεnk = real(dot(ψ[ik][:, n], δHψ[ik][:, n]))
            εnkred = (εnk - εF) / temperature
            δεnkred = δεnk/temperature - εnkred*δtemperature/temperature
            fpnk = filled_occ * Smearing.occupation_derivative(smearing, εnkred)
            δocc[ik][n] = fpnk * δεnkred
            D -= fpnk * basis.kweights[ik] / temperature  # while we're at it, accumulate the total DOS D
        end
        D = mpi_sum(D, basis.comm_kpts)

        if isnothing(model.εF)  # εF === nothing means that Fermi level is fixed by model
            # Compute δεF from δ ∑ occ = 0…
            δocc_tot = mpi_sum(sum(basis.kweights .* sum.(δocc)), basis.comm_kpts)
            δεF = -δocc_tot / D

            # … and add the corresponding contribution to δocc
            for ik = 1:length(basis.kpoints), (n, εnk) in enumerate(ε[ik])
                fpnk = filled_occ * Smearing.occupation_derivative(smearing, (εnk - εF) / temperature)
                δocc[ik][n] -= fpnk * δεF / temperature
            end
        end
    end

    (; δocc, δεF)
end

"""
Perform in-place computations of the derivatives of the wave functions by solving
a Sternheimer equation for each `k`-points. It is assumed the passed `δψ` are initialised
to zero. `bandtol_minus_q` is an array of arrays of tolerances for each band such that
`bandtol_minus_q[ik][n]` leads to the actual tolerance value when solving for `δψ`
(which notably is the variation of variation of `ψ[k_to_k_minus_q[ik]]`).
Note that for phonon calculations, `δHψ[ik]` is ``δH·ψ_{k-q}``, expressed
in `basis.kpoints[ik]` from which `δψ` is computed (but expressed in `basis.kpoints[ik] - q`).
"""
function compute_δψ!(δψ, basis::PlaneWaveBasis{T}, H, ψ, εF, ε, δHψ, ε_minus_q=ε;
                     ψ_extra=[zeros_like(ψk, size(ψk,1), 0) for ψk in ψ],
                     q=zero(Vec3{T}), bandtol_minus_q, δψ0=nothing, kwargs_sternheimer...) where {T}
    # We solve the Sternheimer equation for all columns n at once
    #   (H_k - ε_{n,k-q}) δψ_{n,k} = - (1 - P_{k}) δHψ_{n, k-q},
    # where P_{k} is the projector on ψ_{k} and with the conventions:
    # * δψ_{k} is the variation of ψ_{k-q}, which implies (for ℬ_{k} the `basis.kpoints`)
    #     δψ_{k-q} ∈ ℬ_{k-q} and δHψ_{k-q} ∈ ℬ_{k};
    # * δHψ[ik] = δH ψ_{k-q};
    # * ε_minus_q[ik] = ε_{·, k-q}.
    temperature = basis.model.temperature
    smearing = basis.model.smearing
    filled_occ = filled_occupation(basis.model)
    @assert !haskey(kwargs_sternheimer, :tol)

    # Reporting
    residual_norms = [Vector{T}() for _ in 1:length(ψ)]
    n_iter = [Vector{Int}() for _ in 1:length(ψ)]
    converged = true

    # Compute δψk for each k-point
    for ik = 1:length(ψ)
        Hk   = H[ik]
        ψk   = ψ[ik]
        εk   = ε[ik]
        δψk  = δψ[ik]
        tolk_minus_q = bandtol_minus_q[ik]
        εk_minus_q   = ε_minus_q[ik]
        @assert length(εk_minus_q) == length(tolk_minus_q)
        sizehint!(residual_norms[ik], length(εk_minus_q))
        sizehint!(n_iter[ik], length(εk_minus_q))

        ψk_extra  = ψ_extra[ik]
        @timing "Prepare extra bands" begin
            Hψk_extra = Hk * ψk_extra
            εk_extra  = diag(real.(ψk_extra' * Hψk_extra))
        end
        α = zeros_like(εk, length(εk), length(εk_minus_q))
        for n = 1:length(εk_minus_q)
            fnk_minus_q = filled_occ * Smearing.occupation(smearing, (εk_minus_q[n]-εF) / temperature)

            # Explicit contributions (nonzero only for temperature > 0)
            for m = 1:length(εk)
                # The n == m contribution in compute_δρ is obtained through δoccupation, see
                # the explanation above; except if we perform phonon calculations.
                iszero(q) && (m == n) && continue
                fmk = filled_occ * Smearing.occupation(smearing, (εk[m]-εF) / temperature)
                ddiff = Smearing.occupation_divided_difference
                ratio = filled_occ * ddiff(smearing, εk[m], εk_minus_q[n], εF, temperature)
                α[m, n] = compute_αmn(fmk, fnk_minus_q, ratio)  # fnk_minus_q * αmn + fmk * αnm = ratio
            end
        end
        # Array operations for GPU efficiency
        dot_prods = ψk' * δHψ[ik]
        dot_prods .*= to_device(basis.architecture, α)
        mul!(δψk, ψk, dot_prods, 1, 1)

        # Sternheimer contribution, for all columns of δψk at once.
        εk_minus_q_device = to_device(basis.architecture, εk_minus_q)
        δψk0 = isnothing(δψ0) ? nothing : δψ0[ik]
        res = sternheimer_solver(Hk, ψk, εk_minus_q_device, δHψ[ik];
                                 ψk_extra, εk_extra, Hψk_extra,
                                 tol=tolk_minus_q, δψk0, kwargs_sternheimer...)
        !res.converged && @warn("Sternheimer CG not converged", res.tol, res.residual_norms)

        δψk .+= res.δψk
        append!(residual_norms[ik], res.residual_norms)
        push!(n_iter[ik], res.n_iter)
        converged = converged && res.converged
    end

    (; δψ, n_iter, residual_norms, converged)
end


"""
Compute the orbital and occupation changes as a result of applying the ``χ_0`` superoperator
to the Hamiltonian change `δH` represented by the matrix-vector products `δHψ`. 
"""
@views @timing function apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, δHψ;
                                    δtemperature=zero(eltype(ham.basis)),
                                    occupation_threshold, q=zero(Vec3{eltype(ham.basis)}),
                                    bandtolalg, tol=1e-9, δψ0=nothing, kwargs_sternheimer...)
    basis = ham.basis
    k_to_k_minus_q = k_to_kpq_permutation(basis, -q)

    # We first select orbitals with occupation number higher than
    # occupation_threshold for which we compute the associated response δψn,
    # the others being discarded to ψ_extra.
    # We then use the extra information we have from these additional bands,
    # non-necessarily converged, to split the Sternheimer_solver with a Schur
    # complement.
    occupation = [to_cpu(oc) for oc in occupation]
    (mask_occ, mask_extra) = occupied_empty_masks(occupation, occupation_threshold)

    ψ_occ   = [ψ[ik][:, maskk] for (ik, maskk) in enumerate(mask_occ)]
    ψ_extra = [ψ[ik][:, maskk] for (ik, maskk) in enumerate(mask_extra)]
    ε_occ   = [eigenvalues[ik][maskk] for (ik, maskk) in enumerate(mask_occ)]
    δHψ_minus_q_occ = [δHψ[ik][:, mask_occ[k_to_k_minus_q[ik]]]
                       for ik = 1:length(basis.kpoints)]
    # Only needed for phonon calculations.
    ε_minus_q_occ  = [eigenvalues[k_to_k_minus_q[ik]][mask_occ[k_to_k_minus_q[ik]]]
                      for ik = 1:length(basis.kpoints)]

    # Return band tolerances in the k-point order of basis.kpoints, ψ and occupations
    # which we need to reorder to obtain k-q. Note that only tolerances for the
    # occupied bands are returned.
    bandtol_occ = determine_band_tolerances(bandtolalg, tol)
    bandtol_minus_q_occ = [bandtol_occ[k_to_k_minus_q[ik]] for ik in 1:length(basis.kpoints)]
    @assert bandtolalg.occupation_threshold == occupation_threshold

    # First we compute δoccupation. We only need to do this for the actually occupied
    # orbitals. So we make a fresh array padded with zeros, but only alter the elements
    # corresponding to the occupied orbitals. (Note both compute_δocc! and compute_δψ!
    # assume that the first array argument has already been initialised to zero).
    # For phonon calculations when q ≠ 0, we do not use δoccupation, and compute directly
    # the full perturbation δψ.
    δoccupation = zero.(occupation)
    if iszero(q)
        δocc_occ = [δoccupation[ik][maskk] for (ik, maskk) in enumerate(mask_occ)]
        (; δεF) = compute_δocc!(δocc_occ, basis, ψ_occ, εF, ε_occ, δHψ_minus_q_occ, δtemperature)
    else
        # When δH is not periodic, δH ψnk is a Bloch wave at k+q and ψnk at k,
        # so that δεnk = <ψnk|δH|ψnk> = 0 and there is no occupation shift
        @assert δtemperature == 0 # TODO think about this
        δεF = zero(εF)
    end

    # Then we compute δψ (again in-place into a zero-padded array) with elements of
    # `basis.kpoints` that are equivalent to `k+q`.
    δψ = zero.(δHψ)
    δψ_occ = [δψ[ik][:, maskk] for (ik, maskk) in enumerate(mask_occ[k_to_k_minus_q])]

    if isnothing(δψ0)
        δψ0_occ = nothing
    else
        δψ0_occ = [δψ0[ik][:, mask_occ[k_to_k_minus_q[ik]]] for ik = 1:length(basis.kpoints)]
    end
    res = compute_δψ!(δψ_occ, basis, ham.blocks, ψ_occ, εF, ε_occ, δHψ_minus_q_occ, ε_minus_q_occ;
                      ψ_extra, q, bandtol_minus_q=bandtol_minus_q_occ, δψ0=δψ0_occ,
                      kwargs_sternheimer...)
    (; δψ, δoccupation, δεF, res.n_iter, res.residual_norms, res.converged)
end

"""
Get the density variation `δρ` corresponding to a potential variation `δV`.

Parameters:
- `bandtolalg` and `tol`:
  The resulting density variation is computed targeting an accuracy of `δρ` of
  `tol`. For this the tolerances when solving the Sternheimer equations
  are chosen by the `bandtolalg` algorithm, by default the balanced strategy of
  [arxiv 2505.02319](https://arxiv.org/pdf/2505.02319).
- `miniter`: Minimal number of CG iterations per k and band for Sternheimer
- `maxiter`: Maximal number of CG iterations per k and band for Sternheimer
"""
function apply_χ0(ham, ψ, occupation, εF::T, eigenvalues, δV::AbstractArray{TδV};
                  δtemperature=zero(eltype(ham.basis)),
                  occupation_threshold=default_occupation_threshold(TδV),
                  q=zero(Vec3{eltype(ham.basis)}),
                  bandtolalg=BandtolBalanced(ham.basis, ψ, occupation; occupation_threshold),
                  kwargs_sternheimer...) where {T, TδV}
    basis = ham.basis

    # Make δV respect the basis symmetry group, since we won't be able
    # to compute perturbations that don't anyway
    δV = symmetrize_ρ(basis, δV)

    # Normalize δV to avoid numerical trouble; theoretically should not be necessary,
    # but it simplifies the interaction with the Sternheimer linear solver
    # (it makes the rhs be order 1 even if δV is small)
    normδH = norm(δV)
    normδH < eps(T) && return (; δρ=zero(δV), normδH)
    δV ./= normδH

    if bandtolalg isa BandtolGuaranteed
        # This is the ||K v|| term of arxiv 2505.02319, see also the discussion
        # of the determine_band_tolerances(::BandtolGuaranteed, ...) below.
        bandtolalg = (1/normδH) * bandtolalg
    end

    # For phonon calculations, assemble
    #   δHψ_k = δV_{q} · ψ_{k-q}.
    δHψ = multiply_ψ_by_blochwave(basis, ψ, δV, q)
    res = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, δHψ;
                      δtemperature, occupation_threshold, q, bandtolalg,
                      kwargs_sternheimer...)

    δρ = compute_δρ(basis, ψ, res.δψ, occupation, res.δoccupation; occupation_threshold, q)
    δρ = δρ * normδH
    (; δρ, normδH, res...)
end

function apply_χ0(scfres, δV; kwargs...)
    apply_χ0(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF, scfres.eigenvalues, δV;
             scfres.occupation_threshold, kwargs...)
end


struct BandtolGuaranteed{T,AT}
    bandtol_factors::AT  # Factors by which density tol is multiplied to yield CG tolerances
    bandtol_min::T       # Minimal tolerance
    bandtol_max::T       # Maximal tolerance
    occupation_threshold::T
end

"""
    BandtolGuaranteed(basis, ψ, occupation; occupation_threshold, bandtol_min, bandtol_max)

Guaranteed (grt) algorithm for adaptively choosing the Sternheimer tolerance as discussed in
[arxiv 2505.02319](https://arxiv.org/pdf/2505.02319). Chooses the convergence thresholds
for the Sternheimer solver of each band adaptively such that the resulting density
response is reliably accurate a value of `tol_density` (passed when calling
`determine_band_tolerances` Compared to [`BandtolBalanced`](@ref) less efficient,
but more accurate approach.
"""
function BandtolGuaranteed(args...; kwargs...)
    construct_bandtol(BandtolGuaranteed, args...; kwargs...)
end

struct BandtolBalanced{T,AT}
    bandtol_factors::AT  # Factors by which density tol multiplied to yield CG tolerances
    bandtol_min::T       # Minimal tolerance
    bandtol_max::T       # Maximal tolerance
    occupation_threshold::T
end

"""
    BandtolBalanced(basis, ψ, occupation; occupation_threshold, bandtol_min, bandtol_max)

Balanced (bal) algorithm for adaptively choosing the Sternheimer tolerance as discussed in
[arxiv 2505.02319](https://arxiv.org/pdf/2505.02319). Chooses the convergence thresholds
for the Sternheimer solver adaptively, such that the density response is roughly accurate
to `tol_density` (passed when calling `determine_band_tolerances`.
Compared to [`BandtolGuaranteed`](@ref) usually more efficient,
but sometimes `tol_density` is not fully achieved.
"""
function BandtolBalanced(args...; kwargs...)
    construct_bandtol(BandtolBalanced, args...; kwargs...)
end

function Base.:*(α::Number, t::Bandtol) where {Bandtol <: Union{BandtolGuaranteed,BandtolBalanced}}
    Bandtol(map(facs -> α .* facs, t.bandtol_factors),
            t.bandtol_min, t.bandtol_max, t.occupation_threshold)
end

function construct_bandtol(Bandtol::Type, basis::PlaneWaveBasis, ψ, occupation::AbstractVector{<:AbstractVector{T}};
                           occupation_threshold, bandtol_min=eps(T)/2, bandtol_max=typemax(T)) where {T}
    Ω  = basis.model.unit_cell_volume
    Ng = prod(basis.fft_size)
    Nk = length(basis.kpoints)
    occupation = [to_cpu(oc) for oc in occupation]
    mask_occ = occupied_empty_masks(occupation, occupation_threshold).mask_occ

    # Including k-points the expression (3.11) in 2505.02319 becomes
    #   with Φk = (ψ_{1,k} … ψ_{n,k})_k  (Concatenation of all occupied orbitals for this k)
    #        wk = kweights[ik]
    #        Nocck = Number of occupied orbitals for k
    #        Yk = [f_{1k} z_{1k} … f_{nk} z_{nk}]_k  (Concatenation of all solutions)
    # error ≤ 2 ‖K v_i‖ ∑_k wk ‖Re(F⁻¹ Φk)‖_{2,∞} ‖Re(F⁻¹ Yk)‖_{1,2} sqrt(Nocck)
    #       ≤ 2 ‖K v_i‖ ∑_k wk ‖Re(F⁻¹ Φk)‖_{2,∞} w^{-1} max_{n} f_{nk} ‖z_{nk}‖ sqrt(Nocck)
    # Distributing the error equally across all k-points leads to (with w = sqrt(Ω / Ng))
    #   ‖z_{nk}‖ ≤ sqrt(Ω / Ng) / (‖K v_i‖ sqrt(Nocck) ‖Re(F⁻¹ Φk)‖_{2,∞} * 2f_{nk} Nk wk)
    # If we bound ‖Re(F⁻¹ Φk)‖_{2,∞} from below this is sqrt(Nocc / Ω).
    # See also section SM6 and Table SM4 in 2505.02319.
    #
    # Note that the kernel term ||K v_i|| of 2505.02319 is dropped here as it purely arises
    # from the rescaling of the RHS performed in apply_χ0 above. Consequently the function
    # apply_χ0 also takes care of introducing this term if `BandtolGuaranteed` is employed.
    #

    bandtol_factors = map(1:Nk) do ik
        orbital_term = adaptive_bandtol_orbital_term_(Bandtol, basis, basis.kpoints[ik],
                                                      ψ[ik], mask_occ[ik])
        map(mask_occ[ik]) do n
            (  (sqrt(Ω/Ng) / sqrt(length(mask_occ[ik])) / orbital_term)
             / (2 * occupation[ik][n] * Nk * basis.kweights[ik]))
        end
    end

    Bandtol(bandtol_factors, bandtol_min, bandtol_max, T(occupation_threshold))
end
function construct_bandtol(Bandtol::Type, scfres::NamedTuple; kwargs...)
    construct_bandtol(Bandtol, scfres.basis, scfres.ψ, scfres.occupation;
                      scfres.occupation_threshold, kwargs...)
end

function adaptive_bandtol_orbital_term_(::Type{BandtolGuaranteed}, basis, kpt, ψk, mask_k)
    # Orbital term ‖F⁻¹ Φk‖_{2,∞} for thik k-point
    # Note that compared to the paper we deliberately do not take the real part,
    # since taking the real part represents an additional approximation
    # (thus making the strategy less guaranteed)
    row_sums_squared = sum(mask_k) do n
        ψnk_real = @views ifft(basis, kpt, ψk[:, n])
        abs2.(ψnk_real)
    end
    sqrt(maximum(row_sums_squared))
end
function adaptive_bandtol_orbital_term_(::Type{BandtolBalanced}, basis, kpt, ψ, mask_k)
    sqrt(length(mask_k)) / sqrt(basis.model.unit_cell_volume) # Lower bound of above
end

function determine_band_tolerances(alg, density_tol)
    map(alg.bandtol_factors) do fac_k
        clamp.(fac_k .* density_tol, alg.bandtol_min, alg.bandtol_max)
    end
end
