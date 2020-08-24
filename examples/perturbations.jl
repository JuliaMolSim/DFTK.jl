#
# Perturbation routines
#

import Statistics: mean

"""
Orthonormalize the occupied eigenvectors with Lowdin
"""
DFTK.@timing function Lowdin_orthonormalization!(ψ::AbstractArray, occ::AbstractArray)

    Nk = length(occ)

    for ik in 1:Nk
        occ_bands = [n for n in 1:length(occ[ik]) if occ[ik][n] != 0.0]
        # overlap matrix
        S = ψ[ik][:, occ_bands]'ψ[ik][:, occ_bands]
        E, V = eigen(Hermitian(S))
        Sdiag = Diagonal(sqrt.(1.0./E))
        S = V * Sdiag * V^(-1)
        ψ[ik][:, occ_bands] = ψ[ik][:, occ_bands] * S
        # check orthonormalization
        S = ψ[ik][:, occ_bands]'ψ[ik][:, occ_bands]
        @assert(norm(S - I) < 1e-12)
    end
    ψ
end

"""
Extension of Diagonal to nonlocal operators
"""
function LinearAlgebra.Diagonal(opnl::DFTK.NonlocalOperator)
    [dot(p, opnl.D * p) for p in eachrow(opnl.P)]
end

"""
Compute the average of the local potential and the nonlocal potential
"""
DFTK.@timing function compute_avg(basis::PlaneWaveBasis, H::Hamiltonian)

    # average of the local part of the potential of the Hamiltonian
    avg_local_pot = mean(DFTK.total_local_potential(H))

    # adding the average on the nonlocal part of the potential depending on the
    # k point
    total_pot_avg = []
    for (ik, kpt) in enumerate(basis.kpoints)
        non_local_op = [op for op in H.blocks[ik].operators
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
        ψ1k_fine .= 0
        for n in occ_bands
            ψ1k_fine[idcs_fine_cplmt[ik], n] .=
            .- 1.0 ./ (kin[idcs_fine_cplmt[ik]] .+ total_pot_avg[ik][idcs_fine_cplmt[ik]] .- egvalk[n]) .* r_fine[idcs_fine_cplmt[ik], n]
        end
        push!(ψ1_fine, ψ1k_fine)

    end

    ψ1_fine
end

@views DFTK.@timing function perturbed_eigenvectors_2order(basis::PlaneWaveBasis,
                                                           H::Hamiltonian,
                                                           ψ::AbstractArray,
                                                           basis_fine::PlaneWaveBasis,
                                                           H_fine::Hamiltonian,
                                                           ψ_fine::AbstractArray,
                                                           ψ1_fine::AbstractArray,
                                                           total_pot_avg, idcs_fine_cplmt,
                                                           egval::AbstractArray,
                                                           egval_p2::AbstractArray,
                                                           occ::AbstractArray)

    ψ2 = empty(ψ)
    potψ1_fine = deepcopy(ψ1_fine)

    # compute W * ψ1
    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)
        potψ1_fine[ik] .= 0
        # pre-allocated scratch arrays to compute the HamiltonianBlock
        T = eltype(basis_fine)
        scratch = (
            ψ_reals=[zeros(complex(T), basis_fine.fft_size...) for tid = 1:Threads.nthreads()],
            Hψ_reals=[zeros(complex(T), basis_fine.fft_size...) for tid = 1:Threads.nthreads()]
        )

        ops_no_kin = [op for op in H_fine.blocks[ik].operators
                      if !(op isa DFTK.FourierMultiplication)]
        H_fine_block_no_kin = HamiltonianBlock(basis_fine, kpt_fine, ops_no_kin, scratch)
        potψ1_fine[ik] = mul!(similar(ψ1_fine[ik]), H_fine_block_no_kin, ψ1_fine[ik])
    end

    potψ1, _ = DFTK.interpolate_blochwave(potψ1_fine, basis_fine, basis)

    # solve the linear system (H-λn)ψ2_coarse = -(potψ1-λ_n^(2)ψ2_coarse)
    for (ik, kpt) in enumerate(basis.kpoints)

        ψ2k = deepcopy(ψ[ik])
        ψ2k .= 0
        potψ1k = potψ1[ik]

        # occupied bands
        egvalk = egval[ik]
        occ_bands = [n for n in 1:length(occ[ik]) if occ[ik][n] != 0.0]

        for n in occ_bands
            A = Array(H.blocks[ik]) - egvalk[n]*I
            E, V = eigen(A)
            D = Diagonal(E)
            Dinv = deepcopy(D)
            # we keep only the orthogonal of the eigenvectors to inverse the
            # system
            for i in 1:size(D)[1]
                if abs(D[i,i]) > 1e-8
                    Dinv[i,i] = 1.0 / D[i,i]
                else
                    Dinv[i,i] = 0.0
                end
            end
            Ainv = V*Dinv*V^(-1)
            egval2k = egval_p2[ik][n] - egvalk[n]
            b = -(potψ1k[:, n] - egval2k*ψ[ik][:, n])
            ψ2k[:, n] .= Ainv*b - 1. / 2. * (norm(ψ1_fine)^2) * ψ[ik][:, n]
        end
        push!(ψ2, ψ2k)
    end

    ψ2_fine, _ = DFTK.interpolate_blochwave(ψ2, basis, basis_fine)

    # compute ψ2 components on the fine grid
    # ψ2_fine = -(-Δ|orth + <W> - λ)^{-1} * W * ψ1
    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)

        # kinetic components
        kin = [sum(abs2, basis_fine.model.recip_lattice * (G + kpt_fine.coordinate))
              for G in G_vectors(kpt_fine)] ./ 2

        # occupied bands
        egvalk = egval[ik]
        occ_bands = [n for n in 1:length(occ[ik]) if occ[ik][n] != 0.0]

        for n in occ_bands
            ψ2_fine[ik][idcs_fine_cplmt[ik], n] .=
            .- 1.0 ./ (kin[idcs_fine_cplmt[ik]] .+ total_pot_avg[ik][idcs_fine_cplmt[ik]] .- egvalk[n]) .* potψ1_fine[ik][idcs_fine_cplmt[ik], n]
        end
    end

    ψ2_fine
end

"""
Compute second and third order perturbed eigenvalues
"""
DFTK.@timing function perturbed_eigenvalues(basis_fine::PlaneWaveBasis, H_p::Hamiltonian,
                                            H_fine::Hamiltonian, H_ref::Hamiltonian,
                                            ψ1_fine::AbstractArray, ψ_fine::AbstractArray,
                                            total_pot_avg, egval::AbstractArray, occ::AbstractArray)

    egval_p2 = deepcopy(egval) # second order perturbation
    egval_p3 = deepcopy(egval) # third order perturbation

    for (ik, kpt_fine) in enumerate(basis_fine.kpoints)
        occ_bands = [n for n in 1:length(egval[ik]) if occ[ik][n] != 0.0]

        # pre-allocated scratch arrays to compute the HamiltonianBlock
        T = eltype(basis_fine)
        scratch = (
            ψ_reals=[zeros(complex(T), basis_fine.fft_size...) for tid = 1:Threads.nthreads()],
            Hψ_reals=[zeros(complex(T), basis_fine.fft_size...) for tid = 1:Threads.nthreads()]
        )

        # compute W * ψ1 where W is the total potential (ie H without kinetic)
        ops_no_kin = [op for op in H_p.blocks[ik].operators
                      if !(op isa DFTK.FourierMultiplication)]
        H_p_block_no_kin = HamiltonianBlock(basis_fine, kpt_fine, ops_no_kin, scratch)
        potψ1k = mul!(similar(ψ1_fine[ik]), H_p_block_no_kin, ψ1_fine[ik])

        # perturbation of the eigenvalue (first order is 0 in the linear case)
        #
        #         λp = λ + <ψ|Rn|ψ> + <ψ|W+Rn|ψ1> + <ψ1|W-<W>+Rn|ψ1>
        #
        # where
        # W = Vion + Vcoul(ρn) + Vxc(ρn)
        # Rn = Vcoul(ρ) + Vxc(ρ) - Vcoul(ρn) - Vxc(ρn)
        # note that Rn arises from the nonlinearity and is 0 in the linear case

        # coefficient to take into account (or not) Rn
        coeff_nl = 0

        # pre-allocated scratch arrays to compute the HamiltonianBlock
        scratch_ref = (
            ψ_reals=[zeros(complex(T), basis_fine.fft_size...) for tid = 1:Threads.nthreads()],
            Hψ_reals=[zeros(complex(T), basis_fine.fft_size...) for tid = 1:Threads.nthreads()]
        )
        # perturbed
        ops_hartree_xc = H_p.blocks[ik].operators[end-1:end]
        H_p_block_hxc = HamiltonianBlock(basis_fine, kpt_fine,
                                         ops_hartree_xc, scratch)
        # fine grid
        ops_hartree_xc = H_fine.blocks[ik].operators[end-1:end]
        H_fine_block_hxc = HamiltonianBlock(basis_fine, kpt_fine,
                                            ops_hartree_xc, scratch)

        potrefψk = mul!(similar(ψ[ik]), H_p_block_hxc, ψ_fine[ik])
        potrefψ1k = mul!(similar(ψ1[ik]), H_p_block_hxc, ψ1_fine[ik])

        potfineψk = mul!(similar(ψ_fine[ik]), H_fine_block_hxc, ψ_fine[ik])
        potfineψ1k = mul!(similar(ψ1_fine[ik]), H_fine_block_hxc, ψ1_fine[ik])

        for n in occ_bands
            egval1k = coeff_nl*(dot(ψ_fine[ik][:, n], potrefψk[:, n]) - dot(ψ_fine[ik][:, n], potfineψk[:, n]))
            egval2k = dot(ψ_fine[ik][:, n], potψ1k[:, n]) + coeff_nl*(dot(ψ_fine[ik][:, n],
                      potrefψ1k[:, n]) - dot(ψ_fine[ik][:, n], potfineψ1k[:, n]))
            egval3k = dot(ψ1_fine[ik][:, n], potψ1k[:, n]) - dot(ψ1_fine[ik][:, n],
                      Diagonal(total_pot_avg[ik]) * ψ1_fine[ik][:, n]) + coeff_nl*(dot(ψ1_fine[ik][:, n],
                      potrefψ1k[:, n]) - dot(ψ1_fine[ik][:, n], potfineψ1k[:, n]))
            egval_p2[ik][n] += real(egval1k + egval2k)
            egval_p3[ik][n] += real(egval1k + egval2k + egval3k)
        end
    end
    egval_p2, egval_p3
end

"""
Compute eigenvalues using Rayleigh-Ritz method
"""
DFTK.@timing function Rayleigh_Ritz(basis::PlaneWaveBasis,
                                    H::Hamiltonian, ψ::AbstractArray,
                                    egval::AbstractArray, occ::AbstractArray)

    egval_p = deepcopy(egval)

    for (ik, kpt) in enumerate(basis.kpoints)
        nk = length(occ[ik])
        occ_bands = [n for n in 1:nk if occ[ik][n] != 0.0]
        Hψ = mul!(similar(ψ[ik]), H.blocks[ik], ψ[ik])
        ψHψ = ψ[ik]'Hψ
        egval_p[ik][occ_bands] .= real.(eigen(ψHψ).values[occ_bands])
    end
    egval_p
end

"""
Perturbation function to compute perturbed solutions on finer grids
"""
DFTK.@timing function perturbation(basis::PlaneWaveBasis,
                                   kcoords::AbstractVector, ksymops::AbstractVector,
                                   scfres, Ecut_fine; compute_forces=false,
                                   compute_egval=true, schur=false, avg=true)

    Nk = length(basis.kpoints)

    # coarse grid
    occ = deepcopy(scfres.occupation)
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
    ψ_p = ψ_fine .+ ψ1_fine
    Lowdin_orthonormalization!(ψ_p, occ)

    # compute perturbed density and Hamiltonian
    ρ_p = compute_density(basis_fine, ψ_p, occ)
    # compute energies
    E_p, H_p = energy_hamiltonian(basis_fine, ψ_p, occ; ρ=ρ_p)

    if compute_egval
        # compute the eigenvalue perturbation λp = λ + λ2 + λ3
        # first order peturbation = 0
        H_ref = scfres_ref.ham
        egval_p2, egval_p3 = perturbed_eigenvalues(basis_fine, H_p,
                                                   H_fine, H_ref,
                                                   ψ1_fine, ψ_fine,
                                                   total_pot_avg, egval, occ)

        # Rayleigh - Ritz method to compute eigenvalues from the perturbed
        # eigenvectors
        egval_p_rr = Rayleigh_Ritz(basis_fine, H_p, ψ_p, egval, occ)
    else
        egval_p2 = 0
        egval_p3 = 0
        egval_p_rr = 0
    end

    ## apply schur <=> second order
    if compute_egval && schur
        ψ2_fine = perturbed_eigenvectors_2order(basis, H, ψ,
                                                basis_fine, H_p, ψ_fine, ψ1_fine,
                                                total_pot_avg, idcs_fine_cplmt,
                                                egval, egval_p2, occ)
        ψ_p2 = ψ_fine .+ ψ1_fine .+ ψ2_fine
        Lowdin_orthonormalization!(ψ_p2, occ)
        ρ_p2 = compute_density(basis_fine, ψ_p2, occ)
        E_p2, H_p2 = energy_hamiltonian(basis_fine, ψ_p2, occ; ρ=ρ_p2)
    else
        E_p2 = 0
    end

    # compute forces
    if compute_forces
        forces_p = forces(basis_fine, ψ_p, occ; ρ=ρ_p)
    else
        forces_p = 0
    end

    (E_p, ψ_p, ρ_p, egval_p2, egval_p3, egval_p_rr, forces_p, E_p2)
end
