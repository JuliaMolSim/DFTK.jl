## we compute an estimation of the forces F - F*
function compute_forces_estimate(basis, vec, φ, Pks, occ; term="local")

    T = eltype(basis)
    atoms = basis.model.atoms
    recip_lattice = basis.model.recip_lattice
    unit_cell_volume = basis.model.unit_cell_volume

    f_est = [zeros(Vec3{T}, length(positions)) for (el, positions) in atoms]
    cs = nothing
    for (iel, (el, positions)) in enumerate(atoms)
        form_factors = [Complex{T}(DFTK.local_potential_fourier(el, norm(recip_lattice * G)))
                        for G in G_vectors(basis)]

        for (ir, r) in enumerate(positions)
            f_est[iel][ir] = force_estimate(basis, φ, vec, Pks, occ, form_factors, r, ir, el, term)
        end
    end
    f_est
end

function force_estimate(basis, φ, vec, Pks, occ, form_factors, r, ir, el, term)

    T = eltype(basis)
    unit_cell_volume = basis.model.unit_cell_volume
    ## |F-F*| <= |M^-1/2 ∇V φ| * |M^-1/2vec|

    f_est = zeros(T, 3)

    if term == "local"
        ## |M^-1/2 G∇V φ| --> 3 components vector
        ∇VG = [zero(Vec3{Complex{T}}) for (iG, G) in enumerate(G_vectors(basis))]
        for (iG, G) in enumerate(G_vectors(basis))
            ∇VG[iG] = form_factors[iG] .* cis(-2T(π) * dot(G, r)) .* (-2T(π)) .* G .* im
        end
        (n1, n2, n3) = size(∇VG)
        for i in 1:3
            ∇VGi = [∇VG[j,k,l][i] for j in 1:n1, k in 1:n2, l in 1:n3]
            ∇V_real = G_to_r(basis, ∇VGi)
            ∇Vφ = similar(φ)
            for (ik, kpt) in enumerate(basis.kpoints)
                ∇Vφ[ik] = similar(φ[ik])
                for iband = 1:size(φ[ik], 2)
                    φki_real = G_to_r(basis, kpt, φ[ik][:,iband])
                    ∇Vφ[ik][:,iband] = r_to_G(basis, kpt, ∇V_real .* φki_real)
                end
            end
            M∇Vφ = apply_inv_sqrt_M(basis, φ, Pks, ∇Vφ)
            for (ik, kpt) in enumerate(basis.kpoints)
                for iband = 1:size(φ[ik], 2)
                    @views f_est[i] -= 4 .* (real(dot(M∇Vφ[ik][:,iband], vec[ik][:,iband]))
                                              / sqrt(basis.model.unit_cell_volume))
                end  #iband
            end #ik

        end #i

        return f_est
    elseif term == "nonlocal"
        C = DFTK.build_projection_coefficients_(el.psp)
        for i = 1:3
            tot_red_kpt_number = sum([length(symops) for symops in basis.ksymops])
            for (ik, kpt_irred) in enumerate(basis.kpoints)
                # Here we need to do an explicit loop over
                # symmetries, because the atom displacement might break them
                for isym in 1:length(basis.ksymops[ik])
                    (S, τ) = basis.ksymops[ik][isym]
                    Skpoint, vecSk = DFTK.apply_ksymop((S, τ), basis, kpt_irred, vec[ik])
                    Skpoint, φSk = DFTK.apply_ksymop((S, τ), basis, kpt_irred, φ[ik])
                    Skcoord = Skpoint.coordinate
                    # energy terms are of the form <psi, P C P' psi>,
                    # where P(G) = form_factor(G) * structure_factor(G)
                    qs = [basis.model.recip_lattice * (Skcoord + G)
                          for G in G_vectors(Skpoint)]
                    form_factors = DFTK.build_form_factors(el.psp, qs)
                    structure_factors = [cis(-2T(π) * dot(Skcoord + G, r))
                                         for G in G_vectors(Skpoint)]
                    P = structure_factors .* form_factors ./ sqrt(unit_cell_volume)
                    dPdR = [-2T(π)*im*(Skcoord + G)[i] for G in G_vectors(Skpoint)] .* P

                    dHφSk = P * (C * (dPdR' * φSk))
                    #  MdHφSk = apply_inv_sqrt_M(basis, φ, Pks, [P * (C * (dPdR' * φSk))])[1]
                    for iband = 1:size(φ[ik], 2)
                        @views f_est[i] -= (occ[ik][iband] / tot_red_kpt_number
                                            * basis.model.n_spin_components
                                            * 2real(  dot(vecSk[:, iband], dHφSk[:, iband])))
                    end  #iband
                end #isym
            end #ik
        end  #i
        return f_est
    else
        return 0
    end
end

