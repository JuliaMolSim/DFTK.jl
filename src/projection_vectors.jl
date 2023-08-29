using SparseArrays
using Base.Iterators: flatten


function build_projection_vectors(
    basis::PlaneWaveBasis{T},
    kpt::Kpoint,
    projectors,
    positions::AbstractVector{<:AbstractVector{Vec3{T}}}
) where {T}
    @assert length(projectors) == length(positions)
    l_max = maximum(length, projectors) - 1
    ## Indices:
    # - `a`: species
    # - `j`: position
    # - `l`: angular momentum
    # - `m`: magnetic quantum number
    # - `i`: projector
    # - `q`: Fourier-space point
    ## Inputs:
    # - `projectors[a][l][i]`
    # - `positions[a][j]`

    ## Compute structure factors (the position-dependent part of the projection vectors):
    ## sf[a][j][q]
    structure_factors = compute_structure_factors(basis, kpt, positions)

    ## Compute the form factors (the position-independent part of the projection vectors):
    # Angular part: ffa[l][m][q]
    form_factors_angular = compute_form_factors_angular(basis, kpt, 0:l_max)
    # Radial part: ffr[a][l][i][q]
    form_factors_radial = compute_form_factors_radial(basis, kpt, projectors)

    ## Combine the structure factors and form factors to build the projection vectors:
    ## P[a][j][l][m][i][q] -> P[q,ajlmi]
    volume_normalization = 1 / sqrt(basis.model.unit_cell_volume)
    # Atomic species
    map(zip(form_factors_radial, structure_factors)) do (ffr_a, sf_a)
        # Atom positions
        map(sf_a) do sf_aj
            # Angular momenta
            map(zip(ffr_a, form_factors_angular)) do (ffr_al, ffa_l)
                # Magnetic quantum numbers ∈ -l:+l
                map(ffa_l) do ffa_lm
                    # Projectors
                    map(ffr_al) do ffr_ali
                        sf_aj .* ffr_ali .* ffa_lm .* volume_normalization  # P_ajlmi[q]
                    end  # i
                end  # m
            end  # l
        end  # j
    end  # a
end
function projection_vectors_to_matrix(P)
    P |> flatten |> flatten |> flatten |> flatten |> collect |> Base.Fix1(reduce, hcat)
end

function build_projection_coupling(couplings, positions::Vector{Vector{Vec3{T}}}) where {T}
    @assert length(couplings) == length(positions)
    n_sites = length.(positions)  # Number of sites for each atomic species
    # Atomic species
    map(zip(couplings, n_sites)) do (couplings_a, n_sites_a)
        # Angular momentum
        Da = map(enumerate(couplings_a)) do (l_index, couplings_al)
            l = l_index - 1
            # Repeat the coupling matrix at each angular momentum (-l:+l) times
            fill(Symmetric(couplings_al), 2l + 1)  # m
        end  # l
        # Repeat the list of coupling matrices for each l number of sites times
        fill(Da, n_sites_a)  # j
    end  # a
end
function projection_coupling_to_matrix(D)
    # D[a][j][l][m][i,i'] -> D[ajlm][i,i'] -> D[ajlmi,a'j'l'm'i']
    D |> flatten |> flatten |> flatten .|> sparse |> splat(blockdiag)
end
