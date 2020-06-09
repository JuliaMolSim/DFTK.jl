#
# Perturbation routines
#

import Statistics: mean

"""
Orthonormalize the occupied eigenvectors with Lowdin
"""
DFTK.@timing function Lowdin_orthonormalization!(ψp_fine::AbstractArray, occ::AbstractArray)

    Nk = length(occ)

    for ik in 1:Nk
        occ_bands = [n for n in 1:length(occ[ik]) if occ[ik][n] != 0.0]
        # overlap matrix
        S = ψp_fine[ik][:, occ_bands]'ψp_fine[ik][:, occ_bands]
        E, V = eigen(Hermitian(S))
        Sdiag = Diagonal(sqrt.(1.0./E))
        S = V * Sdiag * V^(-1)
        ψp_fine[ik][:, occ_bands] = ψp_fine[ik][:, occ_bands] * S
        ### check orthonormalization
        S = ψp_fine[ik][:, occ_bands]'ψp_fine[ik][:, occ_bands]
        @assert(norm(S - I) < 1e-12)
    end
    ψp_fine
end

function LinearAlgebra.Diagonal(opnl::DFTK.NonlocalOperator)
    [dot(p, opnl.D * p) for p in eachrow(opnl.P)]
end

"""
Compute the average of the local potential and the nonlocal potential
"""
DFTK.@timing function compute_avg(basis_fine::PlaneWaveBasis, H_fine::Hamiltonian)

    # average of the local part of the potential of the Hamiltonian on the fine
    # grid
    avg_local_pot = mean(DFTK.total_local_potential(H_fine))

    # adding the average on the nonlocal part of the potential depending on the k point
    total_pot_avg = []
    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)
        non_local_op = [op for op in H_fine.blocks[ik].operators
                        if (op isa DFTK.NonlocalOperator)][1]
        avg_non_local_op = Diagonal(non_local_op)

        # compute potential average if used in the perturbation
        if avg
            total_pot_avgk = avg_local_pot .+ avg_non_local_op
        else
            total_pot_avgk = 0*avg_non_local_op
        end
        push!(total_pot_avg, total_pot_avgk)
    end
    total_pot_avg
end

"""
Compute first order perturbation of the eigenvectors
"""
@views DFTK.@timing function perturbed_eigenvectors(basis_fine::PlaneWaveBasis,
                                                    H_fine::Hamiltonian, ψ_fine::AbstractArray,
                                                    total_pot_avg, idcs_fine_cplmt,
                                                    egval::AbstractArray, occ::AbstractArray)

    ψ1_fine = empty(ψ_fine)
    Hψ_fine = mul!(deepcopy(ψ_fine), H_fine, ψ_fine)

    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)

        # kinetic components
        kin = [sum(abs2, basis_fine.model.recip_lattice * (G + kpt_fine.coordinate))
              for G in G_vectors(kpt_fine)] ./ 2

        # occupied bands
        egvalk = egval[ik]
        occ_bands = [n for n in 1:length(occ[ik]) if occ[ik][n] != 0.0]

        # residual on the fine grid
        λψ_fine = similar(ψ_fine[ik])
        λψ_fine .= 0
        for n in occ_bands
            λψ_fine[:, n] .= ψ_fine[ik][:, n] .* egvalk[n]
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
            .- 1 ./ (kin[idcs_fine_cplmt[ik]] .+ total_pot_avg[ik][idcs_fine_cplmt[ik]] .- egvalk[n]) .* r_fine[idcs_fine_cplmt[ik], n]
        end
        push!(ψ1_fine, ψ1k_fine)
    end
    ψ1_fine
end

"""
Compute second and third order perturbed eigenvalues
"""
DFTK.@timing function perturbed_eigenvalues(basis_fine::PlaneWaveBasis, H_fine::Hamiltonian,
                                           ψ1_fine::AbstractArray, ψ_fine::AbstractArray,
                                           total_pot_avg, egval::AbstractArray, occ::AbstractArray)

    egvalp2 = deepcopy(egval) # second order perturbation
    egvalp3 = deepcopy(egval) # third order perturbation

    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)
        occ_bands = [n for n in 1:length(egval[ik]) if occ[ik][n] != 0.0]

        # pre-allocated scratch arrays to compute the HamiltonianBlock
        T = eltype(basis_fine)
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
            egval2k = dot(ψ_fine[ik][:, n], potψ1k[:, n])
            egval3k = dot(ψ1_fine[ik][:, n], potψ1k[:, n]) - dot(ψ1_fine[ik][:, n],
                      Diagonal(total_pot_avg[ik]) * ψ1_fine[ik][:, n])
            egvalp2[ik][n] += real(egval2k)
            egvalp3[ik][n] += real(egval2k) + real(egval3k)
        end
    end
    egvalp2, egvalp3
end

"""
Compute perturbed eigenvalues using Rayliegh-Ritz method
"""
DFTK.@timing function Rayleigh_Ritz(basis_fine::PlaneWaveBasis,
                                    H_fine::Hamiltonian, ψp_fine::AbstractArray,
                                    egval::AbstractArray, occ::AbstractArray)

    egvalp = deepcopy(egval)

    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)
        nk = length(occ[ik])
        occ_bands = [n for n in 1:nk if occ[ik][n] != 0.0]
        Hψp_fine = mul!(similar(ψp_fine[ik]), H_fine.blocks[ik], ψp_fine[ik])
        ψpHψp_fine = ψp_fine[ik]'Hψp_fine
        egvalp[ik][occ_bands] .= real.(eigen(ψpHψp_fine).values[occ_bands])
    end
    egvalp
end

"""
Perturbation function to compute perturbed solutions on finer grids
"""
DFTK.@timing function perturbation(basis::PlaneWaveBasis,
                                   kcoords::AbstractVector, ksymops::AbstractVector,
                                   scfres, Ecut_fine, compute_forces=false)

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
    ψ_fine, idcs_fine, idcs_fine_cplmt = DFTK.interpolate_blochwave(ψ, basis, basis_fine)
    ρ_fine = compute_density(basis_fine, ψ_fine, occ)
    H_fine = Hamiltonian(basis_fine, ψ=ψ_fine, occ=occ; ρ=ρ_fine)

    # compute avg of both local and non-local part of the potential
    total_pot_avg = compute_avg(basis_fine, H_fine)

    # first order perturbation of the eigenvectors
    ψ1_fine = perturbed_eigenvectors(basis_fine, H_fine, ψ_fine, total_pot_avg,
                                     idcs_fine_cplmt, egval, occ)

    # apply the perturbation and orthonormalize the occupied eigenvectors
    ψp_fine = ψ_fine .+ ψ1_fine
    Lowdin_orthonormalization!(ψp_fine, occ)

    # compute perturbed density and Hamiltonian
    ρp_fine = compute_density(basis_fine, ψp_fine, occ)
    Hp_fine = Hamiltonian(basis_fine, ψ=ψp_fine, occ=occ; ρ=ρp_fine)

    # compute the eigenvalue perturbation λp = λ + λ2 + λ3
    # first order peturbation = 0
    egvalp2, egvalp3 = perturbed_eigenvalues(basis_fine, H_fine, ψ1_fine, ψ_fine,
                                             total_pot_avg, egval, occ)

    # Rayleigh - Ritz method to compute eigenvalues from the perturbed
    # eigenvectors
    egvalp_rr = Rayleigh_Ritz(basis_fine, H_fine, ψp_fine, egval, occ)

    # compute energies
    Ep_fine, Hp_fine = energy_hamiltonian(basis_fine, ψp_fine, occ; ρ=ρp_fine)

    # compute forces
    if compute_forces
        forcesp_fine = forces(basis_fine, ψp_fine, occ; ρ=ρp_fine)
    else
        forcesp_fine = 0
    end

    (Ep_fine, ψp_fine, ρp_fine, egvalp2, egvalp3, egvalp_rr, forcesp_fine)
end
