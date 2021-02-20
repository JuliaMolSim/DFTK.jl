## we compute an estimation of the forces F - F*
function compute_forces_estimate(basis, res, φ)


    T = eltype(basis)
    atoms = basis.model.atoms
    recip_lattice = basis.model.recip_lattice
    unit_cell_volume = basis.model.unit_cell_volume

    f_est = [zeros(Vec3{T}, length(positions)) for (el, positions) in atoms]
    for (iel, (el, positions)) in enumerate(atoms)
        form_factors = [Complex{T}(DFTK.local_potential_fourier(el, norm(recip_lattice * G)))
                        for G in G_vectors(basis)]

        for (ir, r) in enumerate(positions)
            f_est[iel][ir] = force_estimate(basis, φ, res, form_factors, r)
        end
    end
    f_est
end

function force_estimate(basis, φ, res, form_factors, r)

    ## |F-F*| <= |M^-1/2 ∇V φ| * |M^-1/2res|
    ## only first kpoint and one eigenvector supported at the moment
    kpt = basis.kpoints[1]
    φ_real = G_to_r(basis, kpt, φ[1][:,1])

    ## M^-1/2 res
    Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    for ik = 1:length(Pks)
        DFTK.precondprep!(Pks[ik], φ[ik])
    end
    Mres = apply_inv_sqrt_M(basis, φ, Pks, res)

    f_est = zeros(3)

    ## |M^-1/2 ∇V φ| --> 3 components vector
    T = eltype(basis)
    ∇VG = [zero(Vec3{Complex{T}}) for (iG, G) in enumerate(G_vectors(basis))]
    for (iG, G) in enumerate(G_vectors(basis))
        ∇VG[iG] = form_factors[iG] .* cis(-2T(π) * dot(G, r)) .* (-2T(π)) .* G .* im
    end
    (n1, n2, n3) = size(∇VG)
    for i in 1:3
        ∇VGi     = [∇VG[j,k,l][i] for j in 1:n1, k in 1:n2, l in 1:n3]
        ∇V_real  = G_to_r(basis, ∇VGi)
        ∇Vφ      = r_to_G(basis, kpt, ∇V_real .* φ_real)
        M∇Vφ     = apply_inv_sqrt_M(basis, φ, Pks, [∇Vφ])
        f_est[i] = 2. * real(dot(M∇Vφ[1], Mres[1]))
    end
    f_est
end

