using Memoize

# Functionality for building the non-local potential term
# and constructing the builder itself.

"""
    term_nonlocal(psp_or_composition...)

Return a representation of the non-local potential term in Kleinman-Bylander form.
`psp_or_composition` are pairs mapping from a Pseudopotential object or a `Species` object
to a list of positions in fractional coordinates.

## Examples
```julia-repl
julia> psp = load_psp("si-pade-q4.hgh")
       nlpot = term_nonlocal(psp => [[0,0,0], [0,1/2,1/2]])
```
or similarly using a Species object
```julia-repl
julia> si = Species(14, psp=load_psp("si-pade-q4.hgh"))
       nlpot = term_nonlocal(si => [[0,0,0], [0,1/2,1/2]])
```
Of course multiple psps or species are possible:
```julia-repl
julia> si = Species(14, psp=load_psp("si-pade-q4.hgh"))
       c = Species(6, psp=load_psp("c-pade-q4.hgh"))
       nlpot = term_nonlocal(si => [[0,0,0]], c =>  [[0,1/2,1/2]])
```

Notice: If a species does not have an associated pseudopotential it will be silently
ignored by this function.
"""
function term_nonlocal(psps_or_composition...)
    n_species = length(psp_or_composition)

    # Function to extract the psp object in case the passed items are "Species"
    extract_psp(elem::Species) = elem.psp
    extract_psp(elem) = elem
    potentials = [extract_psp(elem) => positions
                  for (elem, positions) in psp_or_composition
                  if extract_psp(elem) !== nothing]

    # Compute n_proj
    n_proj = 0
    for (psp, positions) in potentials
        psp.lmax < 0 && continue  # No non-local projectors
        n_proj_psp = sum(size(psp.h[l + 1], 1) * (2l + 1) for l in 0:psp.lmax)
        n_proj += length(positions) * n_proj_psp
    end

    function build_proj_coeffs(basis)
        # TODO In the current version the proj_coeffs still has a lot of zeros.
        #      One could improve this by storing the blocks as a list or in a
        #      BlockDiagonal data structure
        proj_coeffs = zeros(n_proj, n_proj)

        count = 0
        for (psp, positions) in potentials, r in positions
            for l in 0:psp.lmax, m in -l:l
                n_proj_l = size(psp.h[l + 1], 1)
                range = count .+ (1:n_proj_l)
                proj_coeffs[range, range] = psp.h[l + 1]
                count += n_proj_l
            end # l, m
        end # psp, r
        @assert count == n_proj

        proj_coeffs
    end

    @memoize function build_projection_vectors(basis::PlaneWaveModel, kpt::Kpoint)
        model = basis.model
        T = eltype(basis.kpoints[1].coordinate)

        proj_vectors = zeros(Complex{T}, length(kpt.basis), n_proj)
        qs = [model.recip_lattice * (kpt.coordinate + G) for G in kpt.basis]
        qsqs = [sum(abs2, q) for q in qs]

        count = 0
        for (psp, positions) in potentials, r in positions
                structure_factors = [cis(2π * dot(G, r)) for G in kpt.basis]
                radial_proj(iproj, l, qsq) = eval_psp_projection_radial(psp, iproj, l, qsq)

                for l in 0:psp.lmax, m in -l:l
                    Ω = model.unit_cell_volume
                    prefac_lm = im^l .* structure_factors .* ylm_real.(l, m, qs) ./ sqrt(Ω)
                    n_proj_l = size(psp.h[l + 1], 1)

                    for iproj in 1:n_proj_l
                        radial_il = radial_proj.(iproj, l, qsqs)
                        proj_vectors[:, count + iproj] = prefac_lm .* radial_il
                    end # iproj

                    count += n_proj_l
                end # l, m
        end # psp, r
        @assert count == n_proj

        proj_vectors
    end

    function inner(basis::PlaneWaveModel, energy, potential; kwargs...)
        @assert energy === nothing "Energy computation not yet implemented"
        potential === nothing && return energy, nothing
        energy, PotNonLocal(basis, build_proj_coeffs(basis),
                            kpt -> build_projection_vectors(basis, kpt))
    end
    inner
end
