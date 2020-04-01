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
    vap = scfres.eigenvalues
    ψ = scfres.ψ
    ρ = scfres.ρ
    H = scfres.ham

    # interpolate to fine grid and build the new density & hamiltonian
    # idcs_fine[ik] is the list of basis vector indices in basis_fine
    ψ_fine, idcs_fine = DFTK.interpolate_blochwave(ψ, basis, basis_fine)
    ρ_fine = compute_density(basis_fine, ψ_fine, occ)
    H_fine = Hamiltonian(basis_fine, ψ=ψ_fine, occ=occ; ρ=ρ_fine)

    # build the complement indices of basis_fine that are not in basis
    idcs_fine_cplmt = [filter(id -> !(id in idcs_fine[ik]), 1:length(ψ_fine[ik][:,1]))
                       for ik in 1:Nk]

    # first order perturbation of the eigenvectors
    ψ1_fine = empty(ψ_fine)

    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)

        # kinetic components
        kin = [sum(abs2, basis.model.recip_lattice * (G + kpt_fine.coordinate))
              for G in G_vectors(kpt_fine)] ./ 2

        nband = size(ψ[ik], 2)

        # residual on the coarse grid
        Hψ = zeros(Complex{Float64}, size(ψ[ik]))
        mul!(Hψ, H.blocks[ik], ψ[ik])
        λψ = copy(ψ[ik])
        for n in 1:nband
            λψ[:,n] *= vap[ik][n]
        end
        r = Hψ - λψ

        # residual on the fine grid
        Hψ_fine = zeros(Complex{Float64}, size(ψ_fine[ik]))
        mul!(Hψ_fine, H_fine.blocks[ik], ψ_fine[ik])
        λψ_fine = copy(ψ_fine[ik])
        for n in 1:nband
            λψ_fine[:,n] *= vap[ik][n]
        end
        r_fine = Hψ_fine - λψ_fine
        # this residual is different from interpolating r which would only have
        # components <= Ecut

        # we apply the first order perturbation to the eigenvectors ψ one by one
        # ψ1_fine = -(-Δ|orth - λ)^{-1} * r_fine
        ψ1k_fine = similar(ψ_fine[ik], length(G_vectors(kpt_fine)), nband)
        ψ1k_fine .= 0
        for n in 1:nband
            ψ1k_fine[idcs_fine_cplmt[ik], n] .=
            - 1 ./ (kin[idcs_fine_cplmt[ik]] .- vap[ik][n]) .* r_fine[idcs_fine_cplmt[ik], n]
        end
        push!(ψ1_fine, ψ1k_fine)
    end

    # apply the perturbation and renormalize to compute density
    ψp_fine = ψ_fine .+ ψ1_fine
    for ik in 1:Nk
        for n in 1:size(ψ[ik],2)
            ψp_fine[ik][:,n] ./= norm(ψp_fine[ik][:,n])
        end
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
Ecut = 15
println("---------------------------\nSolution for Ecut = $(Ecut)")
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
scfres = self_consistent_field(basis, tol=1e-12)
Etot = sum(values(scfres.energies))

##### Perturbation for several values of the ratio α = Ecut_fine/Ecut
function test_perturbation(α_max)
    α_list = vcat(collect(1:0.2:2.8), collect(3:0.5:α_max))
    E_list = []
    for α in α_list
        println("---------------------------\nEcut_fine = $(α) * Ecut")
        Ep_fine, ψp_fine, ρp_fine = perturbation(basis, kgrid, scfres, α*Ecut)
        push!(E_list, sum(values(Ep_fine)))
    end
    (α_list, E_list)
end

(α_list, E_list) = test_perturbation(5)
figure(figsize=(20,10))
subplot(121)
title("Total energy for α = Ecut_fine/Ecut")
plot(α_list, E_list, "-+", label = "perturbation from Ecut = $(Ecut)")
plot(α_list, [Etot_ref for α in α_list], "-+", label = "Ecut_ref = $(Ecut_ref)")
legend()
subplot(122)
title("Relative energy error for α = Ecut_fine/Ecut")
error_list = abs.((E_list .- Etot_ref)/Etot_ref)
semilogy(α_list, error_list, "-+", label = "perturbation from Ecut = $(Ecut)")
semilogy(α_list, [error_list[1] for α in α_list], "-+", label = "Ecut = $(Ecut)")
legend()

savefig("first_order_perturbation_silicon_kpoints$(length(basis.kpoints))_Ecut_ref$(Ecut_ref)_Ecut$(Ecut).pdf")
