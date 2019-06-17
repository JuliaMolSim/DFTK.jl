#
# TODO This file is a hackish first try ... don't take it serious
#
include("SphericalHarmonics.jl")

struct PotNonLocal
    basis::PlaneWaveBasis

    # TODO I feel these could all be absorbed into one array only
    proj_coeffs       # 
    proj_vectors      # 
    structure_factor  # 
end


"""
positions is mapping from species to list of coordinates
psps is mapping from specios to list of Pseudopotentials
"""
function PotNonLocal(pw::PlaneWaveBasis, positions, psps)
    positions = Dict(positions)
    psps = Dict(psps)

    # For now only one species implemented
    @assert length(positions) == 1

    atoms = first(positions).second
    psp = psps[first(positions).first]
    n_atoms = length(atoms)
    n_k = length(pw.kpoints)

    proj_vectors_all = Vector{Vector{Array{Float64, 3}}}(undef, n_k)
    structure_factor_all = Vector{Matrix{ComplexF64}}(undef, n_k)
    for (ik, k) in enumerate(pw.kpoints)
        n_G = length(pw.basis_wf[ik])

        # Evaluate projection vectors
        proj_vectors = Vector{Array{Float64, 3}}(undef, psp.lmax + 1)
        for l in 0:psp.lmax
            n_proj = size(psp.h[l + 1], 1)
            proj_l = zeros(Float64, n_G, n_proj, 2l + 1)
            for m in -l:l
                for iproj in 1:n_proj
                    for (icont, G) in enumerate(pw.basis_wf[ik])
                        # Compute projector for q and add it to proj_l
                        # including structure factor
                        q = pw.recip_lattice * (G+k)
                        radial_il = eval_psp_projection_radial(psp, iproj, l, sum(abs2, q))
                        proj_l[icont, iproj, l + m + 1] = radial_il * ylm_real(l, m, q)
                        # im^l *
                    end # ig
                end  # iproj
            end  # m
            proj_vectors[l + 1] = proj_l
        end  # l
        proj_vectors_all[ik] = proj_vectors

        Gvectors = [pw.recip_lattice * G for G in pw.basis_wf[ik]]
        structure_factor = Matrix{ComplexF64}(undef, n_G, length(atoms))
        for (iatom, R) in enumerate(atoms)
            for (icont, G) in enumerate(Gvectors)
                structure_factor[icont, iatom] = cis(dot(R, G))
            end
        end
        structure_factor_all[ik] = structure_factor
    end

    PotNonLocal(pw, psp.h, proj_vectors_all, structure_factor_all)
end

function apply_fourier!(out_Xk, pot::PotNonLocal, ik::Int, in_Xk)
    pw = pot.basis
    structure_factor = pot.structure_factor[ik]
    proj_vectors = pot.proj_vectors[ik]
    proj_coeffs = pot.proj_coeffs

    n_G = length(pw.basis_wf[ik])
    n_vec = size(in_Xk, 2)
    n_atoms = size(structure_factor, 2)
    lmax = length(proj_vectors) - 1
    in_Xk = reshape(in_Xk, (n_G, n_vec))

    # TODO Maybe precompute this?
    # Amend projection vector by structure factor
    projsf = [
        broadcast(*, reshape(proj_vectors[l + 1], n_G, :, 2l+1, 1),
                  reshape(structure_factor, n_G, 1, 1, n_atoms))
        for l in 0:lmax
    ]

    # Compute product of transposed projection operator
    # times in for each angular momentum l
    projTin = Vector{Array{ComplexF64, 4}}(undef, lmax + 1)
    for l in 0:lmax
        n_proj = size(proj_vectors[l + 1], 2)
        projsf_l = projsf[l + 1]
        @assert size(projsf_l) == (n_G, n_proj, 2l + 1, n_atoms)

        # TODO use dot
        # Perform application of projector times in_Xk as matrix-matrix product
        projTin_l = adjoint(reshape(projsf_l, n_G, :)) *  in_Xk
        @assert size(projTin_l) ==  (n_proj * (2l + 1) * n_atoms, n_vec)

        projTin[l + 1] = reshape(projTin_l, n_proj, 2l + 1, n_atoms, n_vec)
    end

    # Compute contraction of above result with coefficients h
    # and another projector
    Ω = pw.unit_cell_volume
    out_Xk[:] = zero(out_Xk)
    out = reshape(out_Xk, n_G, n_vec)
    for l in 0:lmax, midx in 1:2l + 1, iatom in 1:n_atoms
        h_l = proj_coeffs[l + 1]
        projsf_l = projsf[l + 1]
        projTin_l = projTin[l + 1]
        out .+= projsf_l[:, :, midx, iatom] * (h_l * projTin_l[:, midx, iatom, :] / Ω)
    end

    out_Xk
end
