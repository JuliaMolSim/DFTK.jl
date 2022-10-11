using LinearMaps
using IterativeSolvers
using ProgressMeter

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
function compute_χ0(ham; temperature=ham.basis.model.temperature)
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
    model = basis.model
    filled_occ = filled_occupation(model)
    n_spin   = basis.model.n_spin_components
    fft_size = basis.fft_size
    n_fft    = prod(fft_size)

    length(model.symmetries) == 1 || error("Disable symmetries completely for computing χ0")

    EVs = [eigen(Hermitian(Array(Hk))) for Hk in ham.blocks]
    Es = [EV.values for EV in EVs]
    Vs = [EV.vectors for EV in EVs]
    occ, εF = compute_occupation(basis, Es; temperature)

    χ0 = zeros(eltype(basis), n_spin * n_fft, n_spin * n_fft)
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
        @showprogress "Computing χ0 for k-point $ik/$(length(basis.kpoints)) ..." for m = 1:N, n = 1:N
            enred = (E[n] - εF) / temperature
            @assert occ[ik][n] ≈ filled_occ * Smearing.occupation(model.smearing, enred)
            ddiff = Smearing.occupation_divided_difference
            ratio = filled_occ * ddiff(model.smearing, E[m], E[n], εF, temperature)
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
    if temperature > 0
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

# Solves (1-P) (H-εn) (1-P) δψn = - (1-P) rhs
# where 1-P is the projector on the orthogonal of ψk
# n is used for the preconditioning with ψk[:,n] and the optional callback
# /!\ It is assumed (and not checked) that ψk'Hk*ψk = Diagonal(εk) (extra states
# included).
function sternheimer_solver(Hk, ψk, εnk, rhs, n; callback=info->nothing,
                            ψk_extra=zeros(size(ψk,1), 0), εk_extra=zeros(0),
                            Hψk_extra=zeros(size(ψk,1), 0),
                            abstol=1e-9, reltol=0, verbose=false)
    basis = Hk.basis
    kpoint = Hk.kpoint

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
    R(ϕ) = ϕ - P_computed(ϕ)

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

    # ψk_extra are not converged but have been Rayleigh-Ritzed (they are NOT
    # eigenvectors of H) so H(ψk_extra) = ψk_extra' (Hk-εn) ψk_extra should be a
    # real diagonal matrix.
    H(ϕ) = Hk * ϕ - εnk * ϕ
    ψk_exHψk_ex = Diagonal(real.(εk_extra .- εnk))

    # 1) solve for δψknᴿ
    # ----------------------------
    # writing αkn as a function of δψknᴿ, we get that δψknᴿ
    # solves the system (in Ran(1-P_computed))
    #
    # R * (H - εn) * (1 - M * (H - εn)) * R * δψknᴿ = R * (1 - M) * b
    #
    # where M = ψk_extra * (ψk_extra' (H-εn) ψk_extra)^{-1} * ψk_extra'
    # is defined above and b is the projection of -rhs onto Ran(Q).
    #
    b = -Q(rhs)
    bb = R(b -  Hψk_extra * (ψk_exHψk_ex \ ψk_extra'b))
    function RAR(ϕ)
        Rϕ = R(ϕ)
        # Schur complement of (1-P) (H-εn) (1-P)
        # with the splitting Ran(1-P) = Ran(P_extra) ⊕ Ran(R)
        R(H(Rϕ) - Hψk_extra * (ψk_exHψk_ex \ Hψk_extra'Rϕ))
    end
    precon = PreconditionerTPA(basis, kpoint)
    precondprep!(precon, ψk[:, n])
    function R_ldiv!(x, y)
        x .= R(precon \ R(y))
    end
    J = LinearMap{eltype(ψk)}(RAR, size(Hk, 1))
    δψknᴿ, ch = cg(J, bb; Pl=FunctionPreconditioner(R_ldiv!), abstol, reltol,
                   verbose, log=true)
    info = (; basis=basis, kpoint=kpoint, ch=ch, n=n)
    callback(info)

    # 2) solve for αkn now that we know δψknᴿ
    # Note that αkn is an empty array if there is no extra bands.
    αkn = ψk_exHψk_ex \ ψk_extra' * (b - H(δψknᴿ))

    δψkn = ψk_extra * αkn + δψknᴿ
end

# Apply the four-point polarizability operator χ0_4P = -Ω^-1
# Returns δψ corresponding to a change in *total* Hamiltonian δH
# We start from
# P = f(H-εF), tr(P) = N
# where P is the density matrix, f the occupation function
# δεn = <ψn|δV|ψn>
# 0 = ∑_n fn' (δεn - δεF) determines δεF
# where fn' = f'((εn-εF)/T)/T

# Then <ψm|δP|ψn> = (fm-fn)/(εm-εn) <ψm|δH|ψn>
# Except for the diagonal which is
# <ψn|δP|ψn> = (fn'-δεF) <ψn|δH|ψn>

# We want to represent this with a δψ. We do *not* impose that
# δψ is orthogonal at finite temperature.
# We get
# δP = ∑_k fk (|δψk><ψk| + |ψk><δψk|)
# Identifying with <ψm|δP|ψn> we get for the diagonal terms
# <ψn|δψn> fn = fn'<ψn|δH-δεF|ψn>
# and for the off-diagonal
# (fm-fn)/(εm-εn) <ψm|δH|ψn> = fm <δψm|ψn> + fn <ψm|δψn>

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

@views @timing function apply_χ0_4P(ham, ψ, occ, εF, eigenvalues, δHψ;
                                    occupation_threshold, kwargs_sternheimer...)
    basis  = ham.basis
    model = basis.model
    temperature = model.temperature
    filled_occ = filled_occupation(model)
    T = eltype(basis)
    Nk = length(basis.kpoints)

    # We first select orbitals with occupation number higher than
    # occupation_threshold for which we compute the associated response δψn,
    # the others being discarded to ψ_extra / ε_extra.
    # We then use the extra information we have from these additional bands,
    # non-necessarily converged, to split the sternheimer_solver with a Schur
    # complement.

    mask_occ   = map(occk -> isless.(occupation_threshold, occk), occ)
    mask_extra = map(occk -> (!isless).(occupation_threshold, occk), occ)

    ψ_occ   = [ψ[ik][:, maskk] for (ik, maskk) in enumerate(mask_occ)]
    ψ_extra = [ψ[ik][:, maskk] for (ik, maskk) in enumerate(mask_extra)]

    ε_occ   = [eigenvalues[ik][maskk] for (ik, maskk) in enumerate(mask_occ)]
    ε_extra = [eigenvalues[ik][maskk] for (ik, maskk) in enumerate(mask_extra)]

    occ_occ = [occ[ik][maskk] for (ik, maskk) in enumerate(mask_occ)]

    # First compute δεF
    δεF = zero(T)
    δocc = [zero(occ_occ[ik]) for ik = 1:Nk]  # = fn' * (δεn - δεF)
    smearing = model.smearing
    if temperature > 0
        # First compute δocc without self-consistent Fermi δεF
        D = zero(T)
        for ik = 1:Nk, (n, εnk) in enumerate(ε_occ[ik])
            enred = (εnk - εF) / temperature
            δεnk = real(dot(ψ_occ[ik][:, n], δHψ[ik][:, n]))
            fpnk = filled_occ * Smearing.occupation_derivative(smearing, enred) / temperature
            δocc[ik][n] = δεnk * fpnk
            D += fpnk * basis.kweights[ik]
        end
        # compute δεF
        D = mpi_sum(D, basis.comm_kpts)  # equal to minus the total DOS
        δocc_tot = mpi_sum(sum(basis.kweights .* sum.(δocc)), basis.comm_kpts)
        δεF = isnothing(model.εF) ? zero(δεF) : δocc_tot / D # no δεF when Fermi level is fixed
        # recompute δocc
        for ik = 1:Nk, (n, εnk) in enumerate(ε_occ[ik])
            enred = (εnk - εF) / temperature
            fpnk = filled_occ * Smearing.occupation_derivative(smearing, enred) / temperature
            δocc[ik][n] -= fpnk * δεF
        end
    end

    # compute δψnk band per band
    δψ = zero.(ψ)
    for ik = 1:Nk
        ψk = ψ_occ[ik]
        δψk = δψ[ik]
        Hψk_extra = ham.blocks[ik] * ψ_extra[ik]

        εk = ε_occ[ik]
        for n = 1:length(εk)
            fnk = filled_occ * Smearing.occupation(smearing, (εk[n]-εF) / temperature)

            # explicit contributions (nonzero only for temperature > 0)
            for m = 1:length(εk)
                fmk = filled_occ * Smearing.occupation(smearing, (εk[m]-εF) / temperature)
                ddiff = Smearing.occupation_divided_difference
                ratio = filled_occ * ddiff(smearing, εk[m], εk[n], εF, temperature)
                αmn = compute_αmn(fmk, fnk, ratio)  # fnk * αmn + fmk * αnm = ratio
                δψk[:, n] .+= ψk[:, m] .* αmn .* (dot(ψk[:, m], δHψ[ik][:, n]) * (n != m))
            end

            # Sternheimer contribution
            δψk[:, n] .+= sternheimer_solver(ham.blocks[ik], ψk, εk[n], δHψ[ik][:, n], n;
                                             ψk_extra=ψ_extra[ik], εk_extra=ε_extra[ik],
                                             Hψk_extra, kwargs_sternheimer...)
        end
    end

    # pad δoccupation
    δoccupation = zero.(occ)
    for (ik, maskk) in enumerate(mask_occ)
        δoccupation[ik][maskk] .= δocc[ik]
    end

    # keeping zeros for extra bands to keep the output δψ with the same size
    # than the input ψ
    (; δψ, δoccupation, δεF)
end

"""
Get the density variation δρ corresponding to a total potential variation δV.
"""
function apply_χ0(ham, ψ, occupation, εF, eigenvalues, δV;
                  occupation_threshold=default_occupation_threshold(),
                  kwargs_sternheimer...)

    basis = ham.basis

    # Make δV respect the basis symmetry group, since we won't be able
    # to compute perturbations that don't anyway
    δV = symmetrize_ρ(basis, δV)

    # Normalize δV to avoid numerical trouble; theoretically should
    # not be necessary, but it simplifies the interaction with the
    # Sternheimer linear solver (it makes the rhs be order 1 even if
    # δV is small)
    normδV = norm(δV)
    normδV < eps(typeof(εF)) && return zero(δV)
    δV ./= normδV

    δHψ = [DFTK.RealSpaceMultiplication(basis, kpt, @views δV[:, :, :, kpt.spin]) * ψ[ik]
           for (ik, kpt) in enumerate(basis.kpoints)]
    δψ, δoccupation, δεF = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, δHψ;
                                       occupation_threshold, kwargs_sternheimer...)
    δρ = DFTK.compute_δρ(basis, ψ, δψ, occupation, δoccupation)
    δρ * normδV
end

function apply_χ0(scfres, δV; kwargs_sternheimer...)
    apply_χ0(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF,
             scfres.eigenvalues, δV; scfres.occupation_threshold,
             kwargs_sternheimer...)
end
