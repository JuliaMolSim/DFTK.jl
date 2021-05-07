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
            f_est[iel][ir], cs = force_estimate(basis, φ, vec, Pks, occ, form_factors, r, ir, el, term)
        end
    end
    f_est, cs
end

function force_estimate(basis, φ, vec, Pks, occ, form_factors, r, ir, el, term)

    T = eltype(basis)
    unit_cell_volume = basis.model.unit_cell_volume
    ## |F-F*| <= |M^-1/2 ∇V φ| * |M^-1/2vec|
    ## only first kpoint and one eigenvector supported at the moment
    kpt = basis.kpoints[1]
    φ_real = G_to_r(basis, kpt, φ[1][:,1])

    f_est = zeros(T, 3)

    if term == "local"
        ## |M^-1/2 G∇V φ| --> 3 components vector
        ∇VG = [zero(Vec3{Complex{T}}) for (iG, G) in enumerate(G_vectors(basis))]
        for (iG, G) in enumerate(G_vectors(basis))
            ∇VG[iG] = form_factors[iG] .* cis(-2T(π) * dot(G, r)) .* (-2T(π)) .* G .* im
        end
        (n1, n2, n3) = size(∇VG)
        global v, Mv
        v = nothing
        for i in 1:3
            ∇VGi     = [∇VG[j,k,l][i] for j in 1:n1, k in 1:n2, l in 1:n3]
            ∇V_real  = G_to_r(basis, ∇VGi)
            ∇Vφ      = r_to_G(basis, kpt, ∇V_real .* φ_real)
            M∇Vφ     = apply_inv_sqrt_M(basis, φ, Pks, [∇Vφ])
            if v ==  nothing
                v = ∇Vφ
                Mv = M∇Vφ[1]
            end
            f_est[i] = - 4 .* real(dot(M∇Vφ[1], vec[1])) / sqrt(basis.model.unit_cell_volume)
        end

        # plot for debug
        #  figure()
        G_energies = DFTK.G_vectors_cart(basis.kpoints[1])
        normG = norm.(G_energies)
        vec_sort = vec[1][sortperm(normG)]
        #  plot(abs.(vec_sort), label="Mvec")
        #  xlabel("index of G by increasing norm")
        #  legend()

        #  figure()
        #  plot(abs.(v[sortperm(normG)]), label="M∇Vφ")
        #  xlabel("index of G by increasing norm")
        #  legend()

        #  figure(ir)
        #  plot(real(conj.(v[sortperm(normG)]) .* vec_sort), label="real(conj(M∇Vφ) * Mvec) ")
        #  xlabel("index of G by increasing norm")
        #  legend()

        # cumsum
        cs = real.(conj.(Mv) .* vec[1] )
        cs = 4 .* cs ./ sqrt(basis.model.unit_cell_volume)
        cs = cs[sortperm(normG)]

        return f_est, cs
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
                    MdHφSk = apply_inv_sqrt_M(basis, φ, Pks, [P * (C * (dPdR' * φSk))])[1]
                    for iband = 1:size(φ[ik], 2)
                        @views f_est[i] -= (occ[ik][iband] / tot_red_kpt_number
                                            * basis.model.n_spin_components
                                            * 2real(  dot(vecSk[:, iband], MdHφSk[:, iband])))
                    end  #iband
                end #isym
            end #ik
        end  #i
        cs = 0
        return f_est, cs
    else
        return 0, 0
    end
end

