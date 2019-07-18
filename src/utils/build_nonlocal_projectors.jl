include("SphericalHarmonics.jl")


"""
    build_nonlocal_projectors(pw::PlaneWaveBasis, positions, psps)

Build a Kleinman-Bylander representation of the non-local potential term
for the given `basis`. `positions` is a mapping from an identifier to
a set of positions in fractional coordinates and `psps` is a mapping from
the identifier to the pseudopotential object associated to this idendifier.
"""
function build_nonlocal_projectors(basis::PlaneWaveBasis, positions, psps)
    positions = Dict(positions)
    psps = Dict(psps)
    T = eltype(basis.lattice)
    n_species = length(positions)
    n_k = length(basis.kpoints)
    Ω = basis.unit_cell_volume

    # Compute n_proj
    n_proj = 0
    for (ispec, species) in enumerate(keys(positions))
        if !haskey(psps, species)
            error("Could not find pseudopotential definition for species $species.")
        end
        psp = psps[species]
        n_proj_psp = sum(size(psp.h[l + 1], 1) * (2l + 1) for l in 0:psp.lmax)
        n_proj += length(positions[species]) * n_proj_psp
    end

    # Build proj_coeffs and proj_vectors
    # TODO In the current version the proj_coeffs still has a lot of zeros.
    #      One could improve this by storing the blocks as a list or in a
    #      BlockDiagonal data structure
    proj_coeffs = zeros(n_proj, n_proj)
    proj_vectors = [zeros(Complex{T}, length(basis_k), n_proj) for basis_k in basis.basis_wf]

    for (ik, k) in enumerate(basis.kpoints)
        qs = [basis.recip_lattice * (k + G) for G in basis.basis_wf[ik]]
        qsqs = [sum(abs2, q) for q in qs]

        count = 0
        for (species, atoms) in positions, r in atoms
            structure_factors = [cis(2π * dot(G, r)) for G in basis.basis_wf[ik]]

            psp = psps[species]
            radial_proj(iproj, l, qsq) = eval_psp_projection_radial(psp, iproj, l, qsq)

            for l in 0:psp.lmax, m in -l:l
                prefac_lm = im^l .* structure_factors .* ylm_real.(l, m, qs) ./ sqrt(Ω)
                n_proj_l = size(psp.h[l + 1], 1)
                range = count .+ (1:n_proj_l)
                proj_coeffs[range, range] = psp.h[l + 1]

                for iproj in 1:n_proj_l
                    radial_il = radial_proj.(iproj, l, qsqs)
                    proj_vectors[ik][:, count + iproj] = prefac_lm .* radial_il
                end # iproj

                count += n_proj_l
            end # l, m
        end # species, r
        @assert count == n_proj
    end # k

    PotNonLocal(basis, proj_vectors, proj_coeffs)
end
