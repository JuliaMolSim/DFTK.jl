using DFTK
using PyPlot

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

model = model_LDA(lattice, atoms; n_electrons=2)

"""
Perturbation function to compute solutions on finer grids
"""
function perturbation(basis, kgrid, scfres, Ecut_fine)

    Nk = length(basis.kpoints)
    basis_fine = PlaneWaveBasis(basis.model, Ecut_fine; kgrid=kgrid)

    # coarse grid
    occ = scfres.occupation
    egval = scfres.eigenvalues
    ψ = scfres.ψ
    ρ = scfres.ρ
    H = scfres.ham

    # interpolate to fine grid and build the new density & hamiltonian
    # idcs_fine[ik] is the list of basis vector indices in basis_fine
    ψ_fine = DFTK.interpolate_blochwave(ψ, basis, basis_fine)
    ρ_fine = compute_density(basis_fine, ψ_fine, occ)
    H_fine = Hamiltonian(basis_fine, ψ=ψ_fine, occ=occ; ρ=ρ_fine)
    idcs_fine, idcs_fine_cplmt = DFTK.grid_interpolation_indices(basis, basis_fine)

    # first order perturbation of the eigenvectors
    ψ1_fine = empty(ψ_fine)

    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)

        # kinetic components
        kin = [sum(abs2, basis.model.recip_lattice * (G + kpt_fine.coordinate))
              for G in G_vectors(kpt_fine)] ./ 2

        # occupied bands
        egvalk = egval[ik]
        occ_bands = [i for i in 1:length(egvalk) if occ[ik][i] != 0.0]

        # residual on the fine grid
        λψ_fine = similar(ψ_fine[ik])
        for n in occ_bands
            λψ_fine[:, n] = ψ_fine[ik][:, n] * egvalk[n]
        end
        r_fine = H_fine.blocks[ik]*ψ_fine[ik] - λψ_fine
        # this residual is different from interpolating r which would only have
        # components <= Ecut

        # first order correction to the occupied eigenvectors ψ one by one
        # the perturbation lives only on the orthogonal of the coarse grid
        # --> ψ1_fine = -(-Δ|orth - λ)^{-1} * r_fine
        ψ1k_fine = copy(ψ_fine[ik])
        for n in occ_bands
            ψ1k_fine[:, n] .= 0
            ψ1k_fine[idcs_fine_cplmt[ik], n] .=
            - 1 ./ (kin[idcs_fine_cplmt[ik]] .- egvalk[n]) .* r_fine[idcs_fine_cplmt[ik], n]
        end
        push!(ψ1_fine, ψ1k_fine)
    end

    # apply the perturbation and orthonormalize the occupied eigenvectors
    ψp_fine = ψ_fine .+ ψ1_fine
    for ik in 1:Nk
        occ_bands = [i for i in 1:length(egval[ik]) if occ[ik][i] != 0.0]
        ψp_fine[ik][:, occ_bands] .= Matrix(qr(ψp_fine[ik][:, occ_bands]).Q)
    end
    ρp_fine = compute_density(basis_fine, ψp_fine, occ)

    # compute and display energies
    Ep_fine, Hp_fine = energy_hamiltonian(basis_fine, ψp_fine, occ; ρ=ρp_fine)
    (Ep_fine, ψp_fine, ρp_fine)
end

################################# Calculations #################################

##### Reference solution on a fine grid
Ecut_ref = 100
println("---------------------------\nSolution for Ecut_ref = $(Ecut_ref)")
basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)
scfres_ref = self_consistent_field(basis_ref, tol=1e-12)
Etot_ref = sum(values(scfres_ref.energies))

##### Solution on a coarse grid
Ecut = 5
println("---------------------------\nSolution for Ecut = $(Ecut)")
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
scfres = self_consistent_field(basis, tol=1e-12)
Etot = sum(values(scfres.energies))

##### Perturbation for several values of the ratio α = Ecut_fine/Ecut
function test_perturbation(α_max)
    α_list = vcat(collect(1:0.2:3), collect(3.5:0.5:α_max))
    Ep_list = []
    E_fine_list = []
    for α in α_list
        println("---------------------------\nEcut_fine = $(α) * Ecut")
        # full scf on basis_fine
        basis_fine = PlaneWaveBasis(model, α*Ecut; kgrid=kgrid)
        scfres_fine = self_consistent_field(basis_fine, tol=1e-12)
        push!(E_fine_list, sum(values(scfres_fine.energies)))

        # perturbation
        Ep_fine, ψp_fine, ρp_fine = perturbation(basis, kgrid, scfres, α*Ecut)
        push!(Ep_list, sum(values(Ep_fine)))
    end
    (α_list, Ep_list, E_fine_list)
end
α_list, Ep_list, E_fine_list = test_perturbation(5)

##### Plotting results
figure(figsize=(20,10))
subplot(121)
title("Total energy for α = Ecut_fine/Ecut")
plot(α_list, Ep_list, "-+", label = "perturbation from Ecut = $(Ecut)")
plot(α_list, E_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
plot(α_list, [Etot_ref for α in α_list], "-+", label = "Ecut_ref = $(Ecut_ref)")
legend()
subplot(122)
title("Relative energy error for α = Ecut_fine/Ecut")
error_list = abs.((Ep_list .- Etot_ref)/Etot_ref)
error_fine_list = abs.((E_fine_list .- Etot_ref)/Etot_ref)
semilogy(α_list, error_list, "-+", label = "perturbation from Ecut = $(Ecut)")
plot(α_list, error_fine_list, "-+", label = "full solution for Ecut_fine = α * Ecut")
semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
legend()

savefig("first_order_perturbation_silicon_kpoints$(length(basis.kpoints))_Ecut_ref$(Ecut_ref)_Ecut$(Ecut).pdf")
