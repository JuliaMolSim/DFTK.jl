using ForwardDiff
using ProgressMeter


"""
Compute the independent-particle susceptibility. Will blow up for large systems
"""
function compute_χ0(ham; diagonal_only=false)
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

    # for δεF we get, with DOS = ∑_n f'n and LDOS = ∑_n f'n |ψn|^2
    # δεF = 1/DOS ∫ δV(r) LDOS(r)dr

    # for δρ we note ρnm = ψn* ψm, and we get
    # δρ = LDOS δεF + ∑_n f'n <ρn|δV> ρn + ∑_{n,m != n} 2Re fn ρnm <ρmn|δV> / (εn-εm)
    # δρ = LDOS δεF + ∑_n f'n <ρn|δV> ρn + ∑_{n,m != n} (fn-fm)/(εn-εm) ρnm <ρmn|δV>
    # The last two terms merge with the convention that (f(x)-f(x))/(x-x) = f'(x) into
    # δρ = LDOS δεF + ∑_{n,m} (fn-fm) ρnm <ρmn|δV> / (εn-εm)
    # Therefore the kernel is LDOS(r) LDOS(r') / DOS + ∑_{n,m} (fn-fm)/(εn-εm) ρnm(r) ρmn(r')
    basis = ham.basis
    model = basis.model
    fft_size = basis.fft_size
    @assert model.spin_polarisation == :none
    filled_occ = filled_occupation(model)
    dVol = basis.model.unit_cell_volume / prod(basis.fft_size)

    EVs = [eigen(Hermitian(Array(Hk))) for Hk in ham.blocks]
    Es = [EV.values for EV in EVs]
    Vs = [EV.vectors for EV in EVs]
    occ, εF = find_occupation(basis, Es)

    χ0 = zeros(eltype(basis), prod(fft_size), prod(fft_size))
    for ik = 1:length(basis.kpoints)
        if length(basis.ksymops[ik]) != 1
            error("Kpoint symmetry not supported")
        end
        N = length(G_vectors(basis.kpoints[ik]))
        @assert N < 10_000
        E = Es[ik]
        V = Vs[ik]
        Vr = hcat(G_to_r.(Ref(basis), Ref(basis.kpoints[ik]), eachcol(V))...)
        Vr = reshape(Vr, prod(fft_size), N)
        @showprogress "Computing χ0 for kpoint $ik/$(length(basis.kpoints)) ..." for m = 1:N, n = 1:N
            diagonal_only && (n != m) && continue
            enred = (E[n] - εF) / model.temperature
            @assert occ[ik][n] ≈ filled_occ * Smearing.occupation(model.smearing, enred)
            factor = filled_occ * Smearing.occupation_divided_difference(model.smearing, E[m], E[n], εF, model.temperature)
            # dVol because inner products have a dVol so that |f> becomes <dVol f|
            # can take the real part here because the nm term is complex conjugate of mn
            # TODO optimize this a bit... use symmetry nm, reduce allocs, etc.
            @views χ0 .+= basis.kweights[ik].*real(conj((Vr[:, m] .* Vr[:, m]')) .* (Vr[:, n] .* Vr[:, n]')) .* factor .* dVol
        end
    end

    # Add variation wrt εF
    if model.temperature > 0
        ldos = vec(LDOS(εF, basis, Es, Vs))
        dos = DOS(εF, basis, Es)
        χ0 .+= (ldos .* ldos') .* dVol ./ dos
    end

    χ0
end

"""
Returns the change in density δρ for a given δV
"""
function apply_chi0(ham, δV, ψ, occupation, εF, eigenvalues; diagonal_only=false)
    # δρ = ∑_nk (f'n δεn |ψn|^2 + 2Re fn ψn* δψn - f'n δεF |ψn|^2
    basis = ham.basis
    model = basis.model
    fft_size = basis.fft_size
    @assert model.spin_polarisation == :none
    filled_occ = filled_occupation(model)
    dVol = basis.model.unit_cell_volume / prod(basis.fft_size)

    δρ = zero(δV)
    for ik = 1:length(basis.kpoints)
        if length(basis.ksymops[ik]) != 1
            error("Kpoint symmetry not supported")
        end
        for iband = 1:size(ψ[ik], 2)
            ψnk = @views ψ[ik][:, iband]
            ψnk_real = G_to_r(basis, basis.kpoints[ik], ψnk)
            εnk = eigenvalues[ik][iband]
            fnk = occupation[ik][iband]
            @assert fnk ≈ filled_occ * Smearing.occupation(model.smearing, (εnk - εF)/model.temperature)

            if model.temperature != 0
                # f'n δεn |ψn|^2
                fpnk = (filled_occ * 1 / model.temperature *
                       Smearing.occupation_derivative(model.smearing, (εnk-εF) / model.temperature))
                δεnk = dot(abs2.(ψnk_real), δV) * dVol
                δρ .+= basis.kweights[ik] .* fpnk .* δεnk .* abs2.(ψnk_real)
            end

            diagonal_only && continue

            # 2Re fn ψn* δψn
            # we split δψn into its component on the computed and uncomputed states:
            # δψn = P ψn + Q ψn
            # we compute Pψn explicitly by sum over states, and
            # Q δψn by solving the Sternheimer equation
            # (H-εn) Q δψn = -Q δV ψn
            # where Q = sum_n |ψn><ψn|
            fnk == 0 && continue

            ## TODO
        end
    end

    if model.temperature > 0
        ldos = LDOS(εF, basis, eigenvalues, ψ)
        dos = DOS(εF, basis, eigenvalues)
        δρ .+= ldos .* dot(ldos, δV) .* dVol ./ dos
    end

    δρ
end
