using LinearMaps
using IterativeSolvers
using ForwardDiff
using ProgressMeter

struct FunctionPreconditioner
    f_ldiv!
end
LinearAlgebra.ldiv!(y, P::FunctionPreconditioner, x) = P.f_ldiv!(y, x)
LinearAlgebra.ldiv!(P::FunctionPreconditioner, x) = P.f_ldiv!(copy(x), x)
# (Base.:\)(P::FunctionPreconditioner, x) = P.f_ldiv!(copy(x), x)

"""
Compute the independent-particle susceptibility. Will blow up for large systems.
Drop all non-diagonal terms with (f(εn)-f(εm))/(εn-εm) factor less than `droptol`.
"""
function compute_χ0(ham; droptol=0, temperature=ham.basis.model.temperature)
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
    fft_size = basis.fft_size
    @assert model.spin_polarization == :none
    filled_occ = filled_occupation(model)
    dVol = basis.model.unit_cell_volume / prod(basis.fft_size)

    EVs = [eigen(Hermitian(Array(Hk))) for Hk in ham.blocks]
    Es = [EV.values for EV in EVs]
    Vs = [EV.vectors for EV in EVs]
    occ, εF = find_occupation(basis, Es, temperature=temperature)

    χ0 = zeros(eltype(basis), prod(fft_size), prod(fft_size))
    for ik = 1:length(basis.kpoints)
        if length(basis.ksymops[ik]) != 1
            error("Kpoint symmetry not supported")
        end
        N = length(G_vectors(basis.kpoints[ik]))
        @assert N < 10_000
        E = Es[ik]
        V = Vs[ik]
        Vr = cat(G_to_r.(Ref(basis), Ref(basis.kpoints[ik]), eachcol(V))..., dims=4)
        Vr = reshape(Vr, prod(fft_size), N)
        @showprogress "Computing χ0 for kpoint $ik/$(length(basis.kpoints)) ..." for m = 1:N, n = 1:N
            enred = (E[n] - εF) / temperature
            @assert occ[ik][n] ≈ filled_occ * Smearing.occupation(model.smearing, enred)
            ddiff = Smearing.occupation_divided_difference
            ratio = filled_occ * ddiff(model.smearing, E[m], E[n], εF, temperature)
            (n != m) && (abs(ratio) < droptol) && continue
            # dVol because inner products have a dVol so that |f> becomes <dVol f|
            # can take the real part here because the nm term is complex conjugate of mn
            # TODO optimize this a bit... use symmetry nm, reduce allocs, etc.
            factor = basis.kweights[ik] * ratio * dVol
            @views χ0 .+= factor .* real(conj((Vr[:, m] .* Vr[:, m]'))
                                            .*   (Vr[:, n] .* Vr[:, n]'))
        end
    end

    # Add variation wrt εF
    if temperature > 0
        ldos = vec(LDOS(εF, basis, Es, Vs))
        dos = DOS(εF, basis, Es)
        χ0 .+= (ldos .* ldos') .* dVol ./ dos
    end

    χ0
end

"""
Returns the change in density δρ for a given δV. Drop all non-diagonal terms with
(f(εn)-f(εm))/(εn-εm) factor less than `droptol`. If `sternheimer_contribution`
is false, only compute excitations inside the provided orbitals.
"""
function apply_χ0(ham, δV, ψ, εF, eigenvalues; droptol=0,
                  sternheimer_contribution=true, temperature=ham.basis.model.temperature)
    if droptol > 0 && sternheimer_contribution == true
        error("Droptol cannot be positive if sternheimer contribution is to be computed.")
    end
    # δρ = ∑_nk (f'n δεn |ψn|^2 + 2Re fn ψn* δψn - f'n δεF |ψn|^2
    basis = ham.basis
    T = eltype(basis)
    model = basis.model
    fft_size = basis.fft_size
    @assert model.spin_polarization == :none
    filled_occ = filled_occupation(model)
    dVol = basis.model.unit_cell_volume / prod(basis.fft_size)

    δρ_four = zeros(complex(T), size(δV))
    for ik = 1:length(basis.kpoints)
        δρk = zero(δV)
        for n = 1:size(ψ[ik], 2)
            ψnk = @view ψ[ik][:, n]
            ψnk_real = G_to_r(basis, basis.kpoints[ik], ψnk)
            εnk = eigenvalues[ik][n]

            # 2Re fn ψn* δψn
            # we split δψn into its component on the computed and uncomputed states:
            # δψn = P ψn + Q ψn
            # we compute Pψn explicitly by sum over states, and
            # Q δψn by solving the Sternheimer equation
            # (H-εn) Q δψn = -Q δV ψn
            # where Q = sum_n |ψn><ψn|

            # explicit contributions
            for m = 1:size(ψ[ik], 2)
                εmk = eigenvalues[ik][m]
                ddiff = Smearing.occupation_divided_difference
                ratio = filled_occ * ddiff(model.smearing, εmk, εnk, εF, temperature)
                (n != m) && (abs(ratio) < droptol) && continue
                ψmk = @view ψ[ik][:, m]
                ψmk_real = G_to_r(basis, basis.kpoints[ik], ψmk)
                # ∑_{n,m != n} (fn-fm)/(εn-εm) ρnm <ρmn|δV>
                ρnm = conj(ψnk_real) .* ψmk_real
                weight = dVol*dot(ρnm, δV)
                δρk .+= real(ratio .* weight .* ρnm)
            end

            # Sternheimer contributions.
            !(sternheimer_contribution) && continue
            fnk = filled_occ * Smearing.occupation(model.smearing, (εnk-εF) / temperature)
            abs(fnk) < eps(T) && continue
            # compute δψn by solving Q (H-εn) Q δψn = -Q δV ψn
            # we err on the side of caution here by applying Q a lot,
            # there are optimizations to be made here
            Q(ϕ) = ϕ - ψ[ik] * (ψ[ik]' * ϕ)
            rhs = - Q(r_to_G(basis, basis.kpoints[ik], δV .* ψnk_real))
            function QHQ(ϕ)
                Qϕ = Q(ϕ)
                Q(ham.blocks[ik] * Qϕ - εnk * Qϕ)
            end
            precon = PreconditionerTPA(basis, basis.kpoints[ik])
            precondprep!(precon, ψnk)
            function f_ldiv!(x, y)
                ldiv!(x, precon, Q(y))
                x .= Q(x)
            end
            J = LinearMap{eltype(ψ[ik])}(QHQ, size(ham.blocks[ik], 1))
            δψnk = cg(J, rhs, Pl=FunctionPreconditioner(f_ldiv!))  # TODO tweak tolerances here
            δψnk_real = G_to_r(basis, basis.kpoints[ik], δψnk)
            δρk .+= 2 .* fnk .* real(conj(ψnk_real) .* δψnk_real)
        end
        _symmetrize_ρ!(δρ_four, r_to_G(basis, complex(δρk)), basis, basis.ksymops[ik], G_vectors(basis))
    end
    δρ = real(G_to_r(basis, δρ_four))
    count = sum(length(basis.ksymops[ik]) for ik in 1:length(basis.kpoints))
    δρ ./= count

    # Add variation wrt εF
    if temperature > 0
        ldos = LDOS(εF, basis, eigenvalues, ψ, temperature=temperature)
        ldos_unsym = LDOS(εF, basis, eigenvalues, ψ, temperature=temperature, disable_symmetrization=true)
        dos = DOS(εF, basis, eigenvalues, temperature=temperature)
        δρ .+= ldos .* dot(ldos_unsym, δV) .* dVol ./ dos
    end

    δρ
end
