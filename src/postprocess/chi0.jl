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
    occ, εF = compute_occupation(basis, Es, temperature=temperature)

    χ0 = zeros(eltype(basis), n_spin * n_fft, n_spin * n_fft)
    for (ik, kpt) in enumerate(basis.kpoints)
        # The sum-over-states terms of χ0 are diagonal in the spin blocks (no αβ / βα terms)
        # so the spin of the kpt selects the block we are in
        spinrange = kpt.spin == 1 ? (1:n_fft) : (n_fft+1:2n_fft)
        χ0σσ = @view χ0[spinrange, spinrange]

        N = length(G_vectors(basis.kpoints[ik]))
        @assert N < 10_000
        E = Es[ik]
        V = Vs[ik]
        Vr = cat(G_to_r.(Ref(basis), Ref(kpt), eachcol(V))..., dims=4)
        Vr = reshape(Vr, n_fft, N)
        @showprogress "Computing χ0 for kpoint $ik/$(length(basis.kpoints)) ..." for m = 1:N, n = 1:N
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
struct FunctionPreconditioner
    precondition!  # precondition!(y, x) applies f to x and puts it into y
end
LinearAlgebra.ldiv!(y::T, P::FunctionPreconditioner, x) where {T} = P.precondition!(y, x)::T
LinearAlgebra.ldiv!(P::FunctionPreconditioner, x) = (x .= P.precondition!(similar(x), x))

# Solves Q (H-εn) Q δψn = -Q rhs
# where Q is the projector on the orthogonal of ψk
@timing_seq function sternheimer_solver(Hk, ψk, ψnk, εnk, rhs; tol_cg=1e-6, verbose=false)
    basis = Hk.basis
    kpoint = Hk.kpoint

    # we err on the side of caution here by applying Q *a lot*
    # there are optimizations to be made here
    Q(ϕ) = ϕ - ψk * (ψk' * ϕ)
    function QHQ(ϕ)
        Qϕ = Q(ϕ)
        Q(Hk * Qϕ - εnk * Qϕ)
    end
    precon = PreconditionerTPA(basis, kpoint)
    precondprep!(precon, ψnk)
    function f_ldiv!(x, y)
        x .= Q(precon \ Q(y))
    end
    J = LinearMap{eltype(ψk)}(QHQ, size(Hk, 1))
    # tol_cg should not be too tight, and in particular not be
    # too far below the error in the ψ. Otherwise Q and H
    # don't commute enough, and an oversolving of the linear
    # system can lead to spurious solutions
    rhs = Q(rhs)
    δψnk = cg(J, rhs, Pl=FunctionPreconditioner(f_ldiv!),
              reltol=0, abstol=tol_cg, verbose=verbose)
    δψnk
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
# Let <ψm|δψn> = αmn <δψm|δH|ψn>. αmn has to satisfy
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

@views @timing function apply_χ0_4P(ham, ψ, occ, εF, eigenvalues, δHψ; kwargs_sternheimer...)
    basis  = ham.basis
    model = basis.model
    temperature = model.temperature
    filled_occ = filled_occupation(model)
    T      = eltype(basis)
    Nk = length(basis.kpoints)

    # First compute δεF
    δεF = zero(T)
    δocc = [zero(occ[ik]) for ik = 1:Nk]  # = fn' * (δεn - δεF)
    if temperature > 0
        # First compute δocc without self-consistent Fermi δεF
        D = zero(T)
        for ik = 1:Nk
            for (n, εnk) in enumerate(eigenvalues[ik])
                enred = (εnk - εF) / temperature
                δεnk = real(dot(ψ[ik][:, n], δHψ[ik][:, n]))
                fpnk = (filled_occ 
                        * Smearing.occupation_derivative(model.smearing, enred)
                        / temperature)
                δocc[ik][n] = δεnk * fpnk
                D += fpnk * basis.kweights[ik]
            end
        end
        # compute δεF
        D = mpi_sum(D, basis.comm_kpts)  # equal to minus the total DOS
        δocc_tot = mpi_sum(sum(basis.kweights .* sum.(δocc)), basis.comm_kpts)
        δεF = δocc_tot / D
    end

    # compute δψnk band per band
    δψ = zero.(ψ)
    for ik = 1:Nk
        ψk = ψ[ik]
        δψk = δψ[ik]

        εk = eigenvalues[ik]
        for n = 1:length(εk)
            fnk = filled_occ * Smearing.occupation(model.smearing, (εk[n]-εF) / temperature)

            # explicit contributions
            for m = 1:length(εk)
                fmk = filled_occ * Smearing.occupation(model.smearing, (εk[m]-εF) / temperature)
                ddiff = Smearing.occupation_divided_difference
                ratio = filled_occ * ddiff(model.smearing, εk[m], εk[n], εF, temperature)
                αmn = compute_αmn(fmk, fnk, ratio)  # fnk * αmn + fmk * αnm = ratio
                δψk[:, n] .+= ψk[:, m] .* αmn .* (dot(ψk[:, m], δHψ[ik][:, n]) - (n == m) * δεF)
            end

            # Sternheimer contribution
            # TODO unoccupied orbitals may have a very bad sternheimer solve, be careful about this
            δψk[:, n] .-= sternheimer_solver(ham.blocks[ik], ψk, ψk[:, n], εk[n], δHψ[ik][:, n];
                                             kwargs_sternheimer...)
        end
    end
    δψ
end

"""
Get the density variation δρ corresponding to a total potential variation δV.

Note: This function assumes that all bands contained in `ψ` and `eigenvalues` are
sufficiently converged. By default the `self_consistent_field` routine of `DFTK`
returns `3` extra bands, which are not converged by the eigensolver
(see `n_ep_extra` parameter). These should be discarded before using this function.
"""
function apply_χ0(ham, ψ, εF, eigenvalues, δV; kwargs_sternheimer...)
    basis = ham.basis
    model = basis.model
    occ = [filled_occupation(model) *
           Smearing.occupation.(model.smearing, (eigenvalues[ik] .- εF) ./ model.temperature)
           for ik = 1:length(basis.kpoints)]

    # Normalize δV to avoid numerical trouble; theoretically should
    # not be necessary, but it simplifies the interaction with the
    # Sternheimer linear solver (it makes the rhs be order 1 even if
    # δV is small)
    normδV = norm(δV)
    normδV < eps(typeof(εF)) && return zero(δV)

    # Make δV respect the full model symmetry group, since it's
    # invalid to consider perturbations that don't (technically it
    # could be made to only respect basis.symmetries, but symmetrizing wrt
    # the model symmetry group means that χ0 is unaffected by the
    # use_symmetry kwarg of basis, which is nice)
    δV = symmetrize(basis, δV) / normδV

    δHψ = [DFTK.RealSpaceMultiplication(basis, kpt, @views δV[:, :, :, kpt.spin]) * ψ[ik]
           for (ik, kpt) in enumerate(basis.kpoints)]
    δψ = apply_χ0_4P(ham, ψ, occ, εF, eigenvalues, δHψ; kwargs_sternheimer...)
    δρ = DFTK.compute_δρ(basis, ψ, δψ, occ)
    δρ * normδV
end
