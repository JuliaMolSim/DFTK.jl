"""
Computes minus the derivatives of the energy with respect to atomic positions.
"""
function forces(scfres)
    # By generalized Hellmann-Feynman, dE/dR = ∂E/∂R. The atomic
    # positions come up explicitly only in the external and nonlocal
    # part of the energy,

    # TODO this assumes that the build_external and build_nonlocal are consistent with model.atoms
    # Find a way to move this computation closer to each term

    # minus signs here because f = -∇E
    ham = scfres.ham
    basis = ham.basis
    model = basis.model
    Psi = scfres.Psi
    ρ = scfres.ρ
    T = real(eltype(ham))

    forces = []
    for (type, positions) in model.atoms
        @assert type.psp != nothing
        # force for this atom type
        f = zeros(Vec3{T}, length(positions))

        # Local part
        @assert model.build_external !== nothing
        # energy = sum of form_factor(G) * struct_factor(G) * rho(G)
        # where struct_factor(G) = cis(-2T(π) * dot(G, r))
        form_factors = [Complex{T}(eval_psp_local_fourier(type.psp, model.recip_lattice * G))
                        for G in G_vectors(basis)] ./ sqrt(model.unit_cell_volume)
        form_factors[1] = 0

        for (ir, r) in enumerate(positions)
            f[ir] -= real(sum(conj(ρ.fourier[iG]) .*
                              form_factors[iG] .*
                              cis(-2T(π) * dot(G, r)) .*
                              (-2T(π)) .* G .* im
                              for (iG, G) in enumerate(G_vectors(basis))))
        end

        # Nonlocal part
        @assert model.build_nonlocal !== nothing
        C = build_projection_coefficients_(type.psp)
        for (ir, r) in enumerate(positions)
            fr = zeros(T, 3)
            for idir = 1:3
                for (ik, kpt) in enumerate(basis.kpoints)
                    # energy terms are of the form <psi, P C P' psi>, where P(G) = form_factor(G) * structure_factor(G)
                    qs = [model.recip_lattice * (kpt.coordinate + G) for G in G_vectors(kpt)]
                    form_factors = build_form_factors(type.psp, qs)
                    structure_factors = [cis(-2T(π)*dot(kpt.coordinate + G, r)) for G in G_vectors(kpt)]
                    P = structure_factors .* form_factors ./ sqrt(model.unit_cell_volume)
                    dPdR = [-2T(π)*im*(kpt.coordinate + G)[idir] for G in G_vectors(kpt)] .* P

                    # TODO BLASify this further
                    for iband = 1:size(Psi[ik], 2)
                        psi = Psi[ik][:, iband]
                        fr[idir] -= basis.kweights[ik] *
                                    scfres.occupation[ik][iband] *
                                    real(dot(psi, P*C*dPdR'*psi) + dot(psi, dPdR*C*P'*psi))
                    end
                end
            end
            f[ir] += fr
        end

        push!(forces, f)
    end

    # Add Ewald forces
    forces_ewald = zeros(Vec3{T}, sum(length(positions) for (elem, positions) in model.atoms))
    energy_nuclear_ewald(model; forces=forces_ewald)

    count = 1
    for i = 1:length(model.atoms)
        for j = 1:length(model.atoms[i][2])
            forces[i][j] += forces_ewald[count]
            count += 1
        end
    end

    forces
end
