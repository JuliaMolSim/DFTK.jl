"""
Perturbation function to compute solutions on finer grids
"""
function perturbation(basis, kcoords, ksymops, scfres, Ecut_fine, compute_forces=false)

    Nk = length(basis.kpoints)

    # coarse grid
    occ = scfres.occupation
    egval = scfres.eigenvalues
    ψ = scfres.ψ
    ρ = scfres.ρ
    H = scfres.ham

    # fine grid
    basis_fine = PlaneWaveBasis(basis.model, Ecut_fine, kcoords, ksymops)

    # interpolate to fine grid and build the new density & hamiltonian
    # idcs_fine[ik] is the list of basis vector indices in basis_fine
    ψ_fine = DFTK.interpolate_blochwave(ψ, basis, basis_fine)
    ρ_fine = compute_density(basis_fine, ψ_fine, occ)
    H_fine = Hamiltonian(basis_fine, ψ=ψ_fine, occ=occ; ρ=ρ_fine)
    idcs_fine, idcs_fine_cplmt = DFTK.grid_interpolation_indices(basis, basis_fine)
    Hψ_fine = mul!(deepcopy(ψ_fine), H_fine, ψ_fine)

    # average of the local part of the potential of the Hamiltonian on the fine
    # grid
    avg_local_pot = mean(DFTK.total_local_potential(H_fine))

    # adding the average on the nonlocal part of the potential depending on the k point
    total_pot_avg = []
    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)
        non_local_op = [op for op in H_fine.blocks[ik].operators
                        if (op isa DFTK.NonlocalOperator)][1]
        avg_non_local_op = diag(Matrix(non_local_op))

        # compute potential average if used in the perturbation
        if avg
            total_pot_avgk = avg_local_pot .+ avg_non_local_op
        else
            total_pot_avgk = 0*avg_non_local_op
        end
        push!(total_pot_avg, total_pot_avgk)
    end

    # first order perturbation of the eigenvectors
    ψ1_fine = empty(ψ_fine)

    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)

        # kinetic components
        kin = [sum(abs2, basis.model.recip_lattice * (G + kpt_fine.coordinate))
              for G in G_vectors(kpt_fine)] ./ 2

        # occupied bands
        egvalk = egval[ik]
        occ_bands = [n for n in 1:length(egvalk) if occ[ik][n] != 0.0]

        # residual on the fine grid
        λψ_fine = similar(ψ_fine[ik])
        λψ_fine .= 0
        for n in occ_bands
            λψ_fine[:, n] = ψ_fine[ik][:, n] * egvalk[n]
        end
        r_fine = Hψ_fine[ik] - λψ_fine
        # this residual is different from interpolating r from the coarse grid
        # which would only have components <= Ecut

        # first order correction to the occupied eigenvectors ψ one by one
        # the perturbation lives only on the orthogonal of the coarse grid
        # we shift also with the mean potential
        # ψ1 = -(-Δ|orth + <W> - λ)^{-1} * r
        ψ1k_fine = deepcopy(ψ_fine[ik])

        for n in occ_bands
            ψ1k_fine[:, n] .= 0
            ψ1k_fine[idcs_fine_cplmt[ik], n] .=
            - 1 ./ (kin[idcs_fine_cplmt[ik]] .+ total_pot_avg[ik][idcs_fine_cplmt[ik]] .- egvalk[n]) .* r_fine[idcs_fine_cplmt[ik], n]
        end
        push!(ψ1_fine, ψ1k_fine)
    end

    # apply the perturbation and orthonormalize the occupied eigenvectors
    ψp_fine = ψ_fine .+ ψ1_fine

    for ik in 1:Nk
        occ_bands = [n for n in 1:length(egval[ik]) if occ[ik][n] != 0.0]
        Nb = length(occ_bands)
        ### QR orthonormalization
        # normalize eigenvectors before orthogonalization
        #  for n in 1:Nb
        #      ψp_fine[ik][:, n] /= norm(ψp_fine[ik][:, n])
        #  end
        #  ψp_fine[ik][:, occ_bands] .= Matrix(qr(ψp_fine[ik][:, occ_bands]).Q)
        ### Lowdin orthonormalization
        # overlap matrix
        S = zeros(ComplexF64, Nb, Nb)
        for i in occ_bands
            for j in occ_bands
                S[i,j] = dot(ψp_fine[ik][:, i], ψp_fine[ik][:, j])
            end
        end
        E, V = eigen(Hermitian(S))
        Sdiag = diagm(sqrt.(1.0./E))
        S = V * Sdiag * V^(-1)
        ψp_fine[ik][:, occ_bands] = ψp_fine[ik][:, occ_bands] * S
        S = zeros(ComplexF64, Nb, Nb)
        for i in occ_bands
            for j in occ_bands
                S[i,j] = dot(ψp_fine[ik][:, i], ψp_fine[ik][:, j])
            end
        end
        ### check orthonormalization
        @assert(norm(S - I) < 1e-12)
    end

    # compute perturbed density and Hamiltonian
    ρp_fine = compute_density(basis_fine, ψp_fine, occ)
    Hp_fine = Hamiltonian(basis_fine, ψ=ψp_fine, occ=occ; ρ=ρp_fine)

    # compute the eigenvalue perturbation λp = λ + λ1 + λ2
    # first order peturbation = 0
    egvalp1 = deepcopy(egval) # second order perturbation
    egvalp2 = deepcopy(egval) # third order perturbation
    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)
        occ_bands = [n for n in 1:length(egval[ik]) if occ[ik][n] != 0.0]

        # pre-allocated scratch arrays to compute the HamiltonianBlock
        T = eltype(basis)
        scratch = (
            ψ_reals=[zeros(complex(T), basis_fine.fft_size...) for tid = 1:Threads.nthreads()],
            Hψ_reals=[zeros(complex(T), basis_fine.fft_size...) for tid = 1:Threads.nthreads()]
        )

        # compute W * ψ1 where W is the total potential (ie H without kinetic)
        ops_no_kin = [op for op in H_fine.blocks[ik].operators
                      if !(op isa DFTK.FourierMultiplication)]
        H_fine_block_no_kin = HamiltonianBlock(basis_fine, kpt_fine, ops_no_kin, scratch)
        potψ1k = mul!(similar(ψ1_fine[ik]), H_fine_block_no_kin, ψ1_fine[ik])

        # perturbation of the eigenvalue (first order is 0)
        # λp = λ + 0 + <ψ|W|ψ1> + <ψ1|W-<W>|ψ1>
        for n in occ_bands
            egval1k = dot(ψ_fine[ik][:, n], potψ1k[:, n])
            egval2k = dot(ψ1_fine[ik][:, n], potψ1k[:, n])
            - dot(ψ1_fine[ik][:, n], diagm(total_pot_avg[ik])*ψ1_fine[ik][:, n])
            egvalp1[ik][n] += real(egval1k)
            egvalp2[ik][n] += real(egval1k) + real(egval2k)
        end
    end

    ### Rayleigh - Ritz method to compute eigenvalues from the perturbed
    # eigenvectors
    egvalp_rl = deepcopy(egval)
    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)
        occ_bands = [n for n in 1:length(egval[ik]) if occ[ik][n] != 0.0]
        Hψp_fine = mul!(similar(ψp_fine[ik]), H_fine.blocks[ik], ψp_fine[ik])
        ψpHψp_fine = ψp_fine[ik]'*Hψp_fine
        egvalp_rl[ik][occ_bands] .= real.(eigen(ψpHψp_fine).values[occ_bands])
    end

    # compute energies
    Ep_fine, Hp_fine = energy_hamiltonian(basis_fine, ψp_fine, occ; ρ=ρp_fine)

    # compute forces
    if compute_forces
        forcesp_fine = forces(basis_fine, ψp_fine, occ; ρ=ρp_fine)
    else
        forcesp_fine = 0
    end

    (Ep_fine, ψp_fine, ρp_fine, egvalp1, egvalp2, egvalp_rl, forcesp_fine)
end

"""
Perturbation for several values of the ratio α = Ecut_fine/Ecut
"""
function test_perturbation_ratio(Ecut, Ecut_ref, α_max, compute_forces)
    """
    Ecut: coarse grid Ecut
    Ecut_ref: Ecut for the reference solution
    α_max: max ratio
    compute_forces: if true, compute forces for the reference, coarse grid and
    fine grid (highly increase computation time)
    """

    ### reference solution
    println("---------------------------\nSolution for Ecut_ref = $(Ecut_ref)")
    basis_ref = PlaneWaveBasis(model, Ecut_ref, kcoords, ksymops)
    scfres_ref = self_consistent_field(basis_ref, tol=1e-12)
    Etot_ref = sum(values(scfres_ref.energies))
    egval_ref = scfres_ref.eigenvalues
    if compute_forces
        forces_ref = forces(scfres_ref)
    end

    ### solution on a coarse grid
    println("---------------------------\nSolution for Ecut = $(Ecut)")
    basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)
    scfres = self_consistent_field(basis, tol=1e-12)
    Etot = sum(values(scfres.energies))

    ### lists to save data for plotting
    α_list = vcat(collect(1:0.1:3), collect(3.5:0.5:α_max))
    Ep_list = []
    E_fine_list = []
    egvalp1_list = []
    egvalp2_list = []
    egvalp_rl_list = []
    egval_fine_list = []
    if compute_forces
        forcesp_list = []
        forces_fine_list = []
    end

    ### test perturbation for several ratio
    for α in α_list
        println("---------------------------\nEcut_fine = $(α) * Ecut")

        # full scf on basis_fine
        basis_fine = PlaneWaveBasis(model, α*Ecut, kcoords, ksymops)
        scfres_fine = self_consistent_field(basis_fine, tol=1e-12)
        push!(E_fine_list, sum(values(scfres_fine.energies)))
        push!(egval_fine_list, scfres_fine.eigenvalues)
        if compute_forces
            forces_fine = forces(scfres_fine)
            push!(forces_fine_list, forces_fine)
            display(forces_fine)
        end

        # perturbation
        Ep_fine, ψp_fine, ρp_fine, egvalp1, egvalp2, egvalp_rl, forcesp = perturbation(basis, kcoords, ksymops, scfres, α*Ecut, compute_forces)
        push!(Ep_list, sum(values(Ep_fine)))
        push!(egvalp1_list, deepcopy(egvalp1))
        push!(egvalp2_list, deepcopy(egvalp2))
        push!(egvalp_rl_list, deepcopy(egvalp_rl))
        if compute_forces
            push!(forcesp_list, forcesp)
            display(forcesp)
        end

    end

    ### Plotting results
    figure(figsize=(20,20))
    tit = "Average shift : $(avg)
    Ne = $(Ne), kpts = $(length(basis.kpoints)), Ecut_ref = $(Ecut_ref), Ecut = $(Ecut)
    kcoords = $(kcoords)"
    suptitle(tit)

    # plot energy relative error
    subplot(221)
    title("Relative energy error for α = Ecut_fine/Ecut")
    error_list = abs.((Ep_list .- Etot_ref)/Etot_ref)
    error_fine_list = abs.((E_fine_list .- Etot_ref)/Etot_ref)
    semilogy(α_list, error_list, "-+", label = "perturbation from Ecut = $(Ecut)")
    semilogy(α_list, error_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
    semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
    xlabel("α")
    legend()

    # plot eigenvalue relative error
    subplot(222)
    title("Relative error on the first egval[1][1] for α = Ecut_fine/Ecut")
    egvalp111 = [egvalp1_list[i][1][1] for i in 1:length(α_list)]
    egvalp211 = [egvalp2_list[i][1][1] for i in 1:length(α_list)]
    egvalp_rl11 = [egvalp_rl_list[i][1][1] for i in 1:length(α_list)]
    egval_fine11 = [egval_fine_list[i][1][1] for i in 1:length(α_list)]
    egval11_ref = egval_ref[1][1]
    error1_list = abs.((egvalp111 .- egval11_ref)/egval11_ref)
    error2_list = abs.((egvalp211 .- egval11_ref)/egval11_ref)
    error_rl_list = abs.((egvalp_rl11 .- egval11_ref)/egval11_ref)
    error_fine_list = abs.((egval_fine11 .- egval11_ref)/egval11_ref)
    semilogy(α_list, error1_list, "-+", label = "perturbation from Ecut = $(Ecut), order 2")
    semilogy(α_list, error2_list, "-+", label = "perturbation from Ecut = $(Ecut), order 3")
    semilogy(α_list, error_rl_list, "-+", label = "perturbation from Ecut = $(Ecut)\nwith Rayleigh coef")
    semilogy(α_list, error_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
    semilogy(α_list, [error1_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
    xlabel("α")
    legend()

    if compute_forces
        #  plot forces relative error, forces_ref should be 0
        subplot(223)
        title("Error on the norm of the forces for α = Ecut_fine/Ecut")
        error_list = norm.([forcesp - forces_ref for forcesp in forcesp_list])
        error_fine_list = norm.([forces_fine - forces_ref for forces_fine in forces_fine_list])
        semilogy(α_list, error_list, "-+", label = "perturbation from Ecut = $(Ecut)")
        semilogy(α_list, error_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
        semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
        legend()
    end

    savefig("first_order_perturbation_silicon_$(Ne)e_kpoints$(length(basis.kpoints))_Ecut_ref$(Ecut_ref)_Ecut$(Ecut)_avg$(avg).pdf")
end

"""
Perturbation for several values of the coarse Ecut
"""
function test_perturbation_coarsegrid(α, Ecut_min, Ecut_max)
    """
    α: ratio to compute the fine grid
    Ecut_min, Ecut_max: interval for the different coarse grid
    """

    ### reference solution
    Ecut_ref = 5*Ecut_max
    println("---------------------------\nSolution for Ecut_ref = $(Ecut_ref)")
    basis_ref = PlaneWaveBasis(model, Ecut_ref, kcoords, ksymops)
    scfres_ref = self_consistent_field(basis_ref, tol=1e-12)
    Etot_ref = sum(values(scfres_ref.energies))
    egval_ref = scfres_ref.eigenvalues

    Ecut_list = range(Ecut_min, Ecut_max, length=Int(Ecut_max/Ecut_min))
    Ep_list = []
    E_coarse_list = []
    for Ecut in Ecut_list
        println("---------------------------\nEcut = $(Ecut)")
        # full scf on coarse
        basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)
        scfres = self_consistent_field(basis, tol=1e-12)
        push!(E_coarse_list, sum(values(scfres.energies)))

        # perturbation
        Ep_fine, _ = perturbation(basis, kcoords, ksymops, scfres, α*Ecut)
        push!(Ep_list, sum(values(Ep_fine)))
    end

    ##### Plotting results
    figure(figsize=(20,10))
    tit = "Average shift : $(avg)
    Ne = $(Ne), kpts = $(length(kcoords)), Ecut_ref = $(Ecut_ref),
    kcoords = $(kcoords)"
    suptitle(tit)

    # size of the discretization grid
    N_list = sqrt.(2 .* Ecut_list)

    # plot energy error
    subplot(121)
    title("Difference with the reference energy Ecut = $(Ecut_ref)")
    error_list = abs.(Ep_list .- Etot_ref)
    error_coarse_list = abs.(E_coarse_list .- Etot_ref)
    semilogy(N_list, error_list, "-+", label = "perturbation")
    semilogy(N_list, error_coarse_list, "-+", label = "coarse grid")
    xlabel("Nc")
    legend()

    # plot energy relative error
    subplot(122)
    title("Relative error between perturbed and non-perturbed")
    error_list = abs.((Ep_list .- Etot_ref) ./ (E_coarse_list .- Etot_ref))
    loglog(N_list, error_list, "-+")
    xlabel("Nc")
    legend()

    # plot slope
    error_list_slope = error_list[end-9:end-5]
    Nc = N_list[end-9:end-5]
    data = DataFrame(X=log.(Nc), Y=log.(error_list_slope))
    ols = lm(@formula(Y ~ X), data)
    Nc_slope = N_list[end-11:end-3]
    slope = exp(coef(ols)[1]) .* Nc_slope .^ coef(ols1)[2]
    loglog(Nc_slope, 1.5 .* slope, "--", label = "slope -3.82")
    legend()

    savefig("first_order_perturbation_silicon_$(Ne)e_kpoints$(length(kcoords))_Ecut_ref$(Ecut_ref)_alpha$(α).pdf")

    ### Return results
    Ecut_list, N_list, Ep_list, E_coarse_list
end


