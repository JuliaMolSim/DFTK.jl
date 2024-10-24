# Compute densities of states

# IDOS (integrated density of states)
# N(ε) = sum_n f_n = sum_n f((εn-ε)/temperature)
# DOS (density of states)
# D(ε) = N'(ε)
#
# LDOS (local density of states)
# LD(ε) = sum_n f_n' |ψn|^2 = sum_n δ(ε - ε_n) |ψn|^2
#
# PD(ε) = sum_n f_n' |<ψn,ϕ>|^2
# ϕ = ∑_R ϕilm(r-pos-R) is the periodized atomic wavefunction, obtained from the pseudopotential
# This is computed for a given (i,l) (eg i=2,l=2 for the 3p) and summed over all m

"""
Total density of states at energy ε
"""
function compute_dos(ε, basis, eigenvalues; smearing=basis.model.smearing,
                     temperature=basis.model.temperature)
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_dos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    D = zeros(typeof(ε), basis.model.n_spin_components)
    for σ = 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        for (iband, εnk) in enumerate(eigenvalues[ik])
            enred = (εnk - ε) / temperature
            D[σ] -= (filled_occ * basis.kweights[ik] / temperature
                     * Smearing.occupation_derivative(smearing, enred))
        end
    end
    D = mpi_sum(D, basis.comm_kpts)
end
function compute_dos(scfres::NamedTuple; ε=scfres.εF, kwargs...)
    compute_dos(ε, scfres.basis, scfres.eigenvalues; kwargs...)
end

"""
Local density of states, in real space. `weight_threshold` is a threshold
to screen away small contributions to the LDOS.
"""
function compute_ldos(ε, basis::PlaneWaveBasis{T}, eigenvalues, ψ;
                      smearing=basis.model.smearing,
                      temperature=basis.model.temperature,
                      weight_threshold=eps(T)) where {T}
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_ldos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    weights = deepcopy(eigenvalues)
    for (ik, εk) in enumerate(eigenvalues)
        for (iband, εnk) in enumerate(εk)
            enred = (εnk - ε) / temperature
            weights[ik][iband] = (-filled_occ / temperature
                                  * Smearing.occupation_derivative(smearing, enred))
        end
    end

    # Use compute_density routine to compute LDOS, using just the modified
    # weights (as "occupations") at each k-point. Note, that this automatically puts in the
    # required symmetrization with respect to kpoints and BZ symmetry
    compute_density(basis, ψ, weights; occupation_threshold=weight_threshold)
end
function compute_ldos(scfres::NamedTuple; ε=scfres.εF, kwargs...)
    compute_ldos(ε, scfres.basis, scfres.eigenvalues, scfres.ψ; kwargs...)
end

"""
Projected density of states at energy ε for an atom with given i and l.
"""
function compute_pdos(ε, basis, eigenvalues, ψ, i, l, psp, position;
                      smearing=basis.model.smearing,
                      temperature=basis.model.temperature)
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_pdos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    projs = compute_pdos_projs(basis, eigenvalues, ψ, i, l, psp, position)

    D = zeros(typeof(ε[1]), length(ε), 2l+1)
    for (i, iε) in enumerate(ε)
        for (ik, projk) in enumerate(projs)
            @views for (iband, εnk) in enumerate(eigenvalues[ik])
                enred = (εnk - iε) / temperature
                D[i, :] .-= (filled_occ .* basis.kweights[ik] .* projk[iband, :]
                             ./ temperature
                             .* Smearing.occupation_derivative(smearing, enred))       
            end
        end
    end
    D = mpi_sum(D, basis.comm_kpts)
end

function compute_pdos(scfres::NamedTuple, iatom, i, l; ε=scfres.εF, kwargs...)
    psp = scfres.basis.model.atoms[iatom].psp
    position = scfres.basis.model.positions[iatom]
    # TODO do the symmetrization instead of unfolding    
    scfres = unfold_bz(scfres)
    compute_pdos(ε, scfres.basis, scfres.eigenvalues, scfres.ψ, i, l, psp, position; kwargs...)
end

# Build atomic orbitals projections projs[ik][iband,m] = |<ψnk, ϕ>|^2 for a single atom.
# TODO optimization ? accept a range of positions, in case we want to compute the PDOS
# for all atoms of the same type (and reuse the computation of the atomic orbitals)
function compute_pdos_projs(basis, eigenvalues, ψ, i, l, psp::NormConservingPsp, position)
    # Precompute the form factors on all kpoints at once so we can better use the cache (memory-compute tradeoff).
    # Revisit this (pass the cache around explicitly) if RAM becomes an issue.
    G_plus_k_all = [Gplusk_vectors(basis, basis.kpoints[ik])
                    for ik = 1:length(basis.kpoints)]
    G_plus_k_all_cart = [map(recip_vector_red_to_cart(basis.model), gpk) 
                         for gpk in G_plus_k_all]

    # Build form factors of pseudo-wavefunctions centered at 0.
    fun(p) = eval_psp_pswfc_fourier(psp, i, l, p)
    # form_factors_all[ik][p,m]
    form_factors_all = build_form_factors(fun, l, G_plus_k_all_cart)

    projs = Vector{Matrix}(undef, length(basis.kpoints))
    for (ik, ψk) in enumerate(ψ)
        structure_factor = [cis2pi(-dot(position, p)) for p in G_plus_k_all[ik]]
        # TODO orthogonalize pseudo-atomic wave functions?
        proj_vectors = structure_factor .* form_factors_all[ik] ./ sqrt(basis.model.unit_cell_volume)
        projs[ik] = abs2.(ψk' * proj_vectors) # contract on p -> projs[ik][iband,m]
    end

    projs
end

"""
Plot the density of states over a reasonable range. Requires to load `Plots.jl` beforehand.
"""
function plot_dos end

function plot_ldos end

function plot_pdos end
