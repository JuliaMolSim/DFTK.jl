using DFTK
using PyPlot
import Statistics: mean

##### Setting the model
# Calculation parameters
kgrid = [1, 1, 1]       # k-Point grid
supercell = [1, 1, 1]   # Lattice supercell

# Setup silicon lattice
a = 10.263141334305942  # Silicon lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# Make a supercell if desired
pystruct = pymatgen_structure(lattice, atoms)
pystruct.make_supercell(supercell)
lattice = load_lattice(pystruct)
atoms = [Si => [s.frac_coords for s in pystruct.sites]]

# precize the number of electrons on build the model
Ne = 2
model = model_LDA(lattice, atoms; n_electrons=Ne)

# kgrid and ksymops
kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.lattice, model.atoms)
# modify kcoords to use different k points
#  kcoords = [[rand(), rand(), rand()]]
#  a specific kcoords for which we know it works
kcoords = [[0.27204337462860106, 0.4735127814871176, 0.6306195069419347]]
println(kcoords)

"""
Perturbation function to compute solutions on finer grids
"""
function perturbation(basis, kcoords, ksymops, scfres, Ecut_fine)

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

    # average of the local part of the potential of the Hamiltonian on the fine
    # grid
    avg_local_pot = mean(DFTK.total_local_potential(H_fine))

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
        for n in occ_bands
            λψ_fine[:, n] = ψ_fine[ik][:, n] * egvalk[n]
        end
        r_fine = H_fine.blocks[ik]*ψ_fine[ik] - λψ_fine
        # this residual is different from interpolating r from the coarse grid
        # which would only have components <= Ecut

        # first order correction to the occupied eigenvectors ψ one by one
        # the perturbation lives only on the orthogonal of the coarse grid
        # we shift also with the mean potential
        # ψ1 = -(-Δ|orth + W_mean - λ)^{-1} * r
        ψ1k_fine = copy(ψ_fine[ik])

        # average on the nonlocal part of the potential for the current kpoint
        non_local_op = [op for op in H_fine.blocks[ik].operators
                        if (op isa DFTK.NonlocalOperator)][1]
        avg_non_local_op = diag(Matrix(non_local_op))

        if avg
            total_pot_avg = avg_local_pot .+ avg_non_local_op
        else
            total_pot_avg = 0*avg_non_local_op
        end

        for n in occ_bands
            ψ1k_fine[:, n] .= 0
            ψ1k_fine[idcs_fine_cplmt[ik], n] .=
            - 1 ./ (kin[idcs_fine_cplmt[ik]] .+ total_pot_avg[idcs_fine_cplmt[ik]] .- egvalk[n]) .* r_fine[idcs_fine_cplmt[ik], n]
        end
        push!(ψ1_fine, ψ1k_fine)
    end

    # apply the perturbation and orthonormalize the occupied eigenvectors
    ψp_fine = ψ_fine .+ ψ1_fine

    for ik in 1:Nk
        occ_bands = [n for n in 1:length(egval[ik]) if occ[ik][n] != 0.0]
        Nb = length(occ_bands)
        # normalize eigenvectors before orthogonalization
        #  for n in 1:Nb
        #      ψp_fine[ik][:, n] /= norm(ψp_fine[ik][:, n])
        #  end
        ### QR orthonormalization
        #  ψp_fine[ik][:, occ_bands] .= Matrix(qr(ψp_fine[ik][:, occ_bands]).Q)
        ### Lowdin orthonormalization
        # overlap matrix
        S = zeros(ComplexF64, Nb, Nb)
        for i in 1:Nb
            for j in 1:Nb
                S[i,j] = dot(ψp_fine[ik][:, i], ψp_fine[ik][:, j])
            end
        end
        display(S)
        E, V = eigen(Hermitian(S))
        Sdiag = diagm(sqrt.(1.0./E))
        S = V * Sdiag * V^(-1)
        ψp_fine[ik][:, occ_bands] = ψp_fine[ik][:, occ_bands] * S
    end

    # compute perturbed density and Hamiltonian
    ρp_fine = compute_density(basis_fine, ψp_fine, occ)
    Hp_fine = Hamiltonian(basis_fine, ψ=ψp_fine, occ=occ; ρ=ρp_fine)

    # compute the eigenvalue perturbation
    egvalp1 = deepcopy(egval)
    egvalp2 = deepcopy(egval)
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

        # second order perturbation of the eigenvalue (first order is 0)
        # λp = λ + 0 + <ψ|W|ψ1>
        for n in occ_bands
            egval1k = dot(ψ_fine[ik][:, n], potψ1k[:, n])
            egval2k = dot(ψ1_fine[ik][:, n], potψ1k[:, n])
            egvalp1[ik][n] += real(egval1k)
            egvalp2[ik][n] += real(egval1k) + real(egval2k)
        end
    end

    # Rayleigh quotient from the perturbed eigenvectors
    #  λp = <ψp|H|ψp>
    #  egvalp_rl = deepcopy(egval)
    #  for (ik, kpt_fine) in enumerate(basis_fine.kpoints)
    #      occ_bands = [n for n in 1:length(egval[ik]) if occ[ik][n] != 0.0]
    #      Hψp_fine = mul!(similar(ψp_fine[ik]), H_fine.blocks[ik], ψp_fine[ik])
    #      for n in occ_bands
    #          egvalp_rl[ik][n] = real(dot(ψp_fine[ik][:, n], Hψp_fine[:, n]))
    #      end
    #  end
    # Rayleigh - Ritz method to compute eigenvalues from the perturbed
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
    forces_fine = norm(forces(basis_fine, ψp_fine, occ; ρ=ρp_fine))

    (Ep_fine, ψp_fine, ρp_fine, egvalp1, egvalp2, egvalp_rl, forces_fine)
end

################################# Calculations #################################

##### Reference solution on a fine grid
#  Ecut_ref = 100
#  println("---------------------------\nSolution for Ecut_ref = $(Ecut_ref)")
#  basis_ref = PlaneWaveBasis(model, Ecut_ref, kcoords, ksymops)
#  scfres_ref = self_consistent_field(basis_ref, tol=1e-12)
#  Etot_ref = sum(values(scfres_ref.energies))
#  egval_ref = scfres_ref.eigenvalues
#  forces_ref = norm(forces(scfres_ref))

##### Solution on a coarse grid
Ecut = 15
println("---------------------------\nSolution for Ecut = $(Ecut)")
basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)
scfres = self_consistent_field(basis, tol=1e-12)
Etot = sum(values(scfres.energies))
display(forces(scfres))

##### Perturbation for several values of the ratio α = Ecut_fine/Ecut
function test_perturbation(α_max)
    α_list = vcat(collect(1:0.2:3), collect(3.5:0.5:α_max))
    Ep_list = []
    E_fine_list = []
    egvalp1_list = []
    egvalp2_list = []
    egvalp_rl_list = []
    egval_fine_list = []
    forcesp_list = []
    forces_fine_list = []
    for α in α_list
        println("---------------------------\nEcut_fine = $(α) * Ecut")
        # full scf on basis_fine
        basis_fine = PlaneWaveBasis(model, α*Ecut, kcoords, ksymops)
        scfres_fine = self_consistent_field(basis_fine, tol=1e-12)
        push!(E_fine_list, sum(values(scfres_fine.energies)))
        push!(egval_fine_list, scfres_fine.eigenvalues)
        forces_fine = forces(scfres_fine)
        display(forces_fine)
        push!(forces_fine_list, norm(forces_fine))

        # perturbation
        Ep_fine, ψp_fine, ρp_fine, egvalp1, egvalp2, egvalp_rl, forcesp = perturbation(basis, kcoords, ksymops, scfres, α*Ecut)
        push!(Ep_list, sum(values(Ep_fine)))
        push!(egvalp1_list, deepcopy(egvalp1))
        push!(egvalp2_list, deepcopy(egvalp2))
        push!(egvalp_rl_list, deepcopy(egvalp_rl))
        push!(forcesp_list, norm(forcesp))
    end
    (α_list, Ep_list, E_fine_list, egvalp1_list, egvalp2_list, egvalp_rl_list, egval_fine_list, forcesp_list, forces_fine_list)
end

avg = true
α_list, Ep_list, E_fine_list, egvalp1_list, egvalp2_list, egvalp_rl_list, egval_fine_list, forcesp_list, forces_fine_list = test_perturbation(3)

##### Plotting results
figure(figsize=(20,20))
tit = "Average shift : $(avg)
Ne = $(Ne), kpts = $(length(basis.kpoints)), Ecut_ref = $(Ecut_ref), Ecut = $(Ecut)
kcoords = $(kcoords)"
suptitle(tit)

# plot energy
subplot(221)
title("Total energy for α = Ecut_fine/Ecut")
plot(α_list, Ep_list, "-+", label = "perturbation from Ecut = $(Ecut)")
plot(α_list, E_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
plot(α_list, [Etot_ref for α in α_list], "-+", label = "Ecut_ref = $(Ecut_ref)")
xlabel("α")
legend()

# plot energy relative error
subplot(222)
title("Relative energy error for α = Ecut_fine/Ecut")
error_list = abs.((Ep_list .- Etot_ref)/Etot_ref)
error_fine_list = abs.((E_fine_list .- Etot_ref)/Etot_ref)
semilogy(α_list, error_list, "-+", label = "perturbation from Ecut = $(Ecut)")
semilogy(α_list, error_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
xlabel("α")
legend()

# plot eigenvalue relative error
subplot(223)
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

# plot forces relative error, forces_ref should be 0
subplot(224)
title("Error on the norm of the forces for α = Ecut_fine/Ecut")
error_list = abs.(forcesp_list)
error_fine_list = abs.(forces_fine_list)
semilogy(α_list, error_list, "-+", label = "perturbation from Ecut = $(Ecut)")
semilogy(α_list, error_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
legend()

savefig("first_order_perturbation_silicon_$(Ne)e_kpoints$(length(basis.kpoints))_Ecut_ref$(Ecut_ref)_Ecut$(Ecut)_avg$(avg).pdf")
