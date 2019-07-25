include("SphericalHarmonics.jl")


"""
    build_nonlocal_projectors(pw::PlaneWaveBasis, psp_or_composition...)

Build a Kleinman-Bylander representation of the non-local potential term
for the given `basis`. `psp_or_composition` are pairs mapping from a Pseudopotential object
or a `Species` object to a list of positions in fractional coordinates.

## Examples
```julia-repl
julia> psp = load_psp("si-pade-q4.hgh")
       nlpot = build_nonlocal_projectors(basis, psp => [[0,0,0], [0,1/2,1/2]])
```
or similarly using a Species object
```julia-repl
julia> si = Species(14, psp=load_psp("si-pade-q4.hgh"))
       nlpot = build_nonlocal_projectors(basis, si => [[0,0,0], [0,1/2,1/2]])
```
Of course multiple psps or species are possible:
```julia-repl
julia> si = Species(14, psp=load_psp("si-pade-q4.hgh"))
       c = Species(6, psp=load_psp("c-pade-q4.hgh"))
       nlpot = build_nonlocal_projectors(basis, si => [[0,0,0]], c =>  [[0,1/2,1/2]])
```

Notice: If a species does not have an assoiciated pseudopotential it will be silently
ignored by this function.
"""
function build_nonlocal_projectors(basis::PlaneWaveBasis, psp_or_composition...)
    T = eltype(basis.lattice)
    n_species = length(psp_or_composition)
    n_k = length(basis.kpoints)
    Ω = basis.unit_cell_volume

    # Function to extract the psp object in case the passed items are "Species"
    extract_psp(elem::Species) = elem.psp
    extract_psp(elem) = elem
    potentials = [extract_psp(elem) => positions
                  for (elem, positions) in psp_or_composition
                  if extract_psp(elem) !== nothing]

    # Compute n_proj
    n_proj = 0
    for (psp, positions) in potentials
        n_proj_psp = sum(size(psp.h[l + 1], 1) * (2l + 1) for l in 0:psp.lmax)
        n_proj += length(positions) * n_proj_psp
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
        for (psp, positions) in potentials
            for r in positions
                structure_factors = [cis(2π * dot(G, r)) for G in basis.basis_wf[ik]]
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
            end # r
        end # psp, positions
        @assert count == n_proj
    end # k

    PotNonLocal(basis, proj_vectors, proj_coeffs)
end
