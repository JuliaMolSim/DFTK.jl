# Compute densities of states

# IDOS (integrated density of states)
# N(ε) = sum_n f_n = sum_n f((εn-ε)/temperature)
# DOS (density of states)
# D(ε) = N'(ε)
#
# LDOS (local density of states)
# LD(ε) = sum_n f_n' |ψn|^2 = sum_n δ(ε - ε_n) |ψn|^2
#
# PDOS (projected density of states)
# PD(ε) = sum_n f_n' |<ψn,ϕlmi>|^2
# ϕlmi = ∑_R ϕlmi(r-pos-R) is a periodic atomic wavefunction centered at pos

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
Projected density of states at energy ε for the first atom in the atom group.
"""
function compute_pdos(ε, basis, eigenvalues, ψ, psp_group; 
                      smearing=basis.model.smearing,
                      temperature=basis.model.temperature)
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_pdos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    # Calculate the projections of the first atom in the atom group
    psp = basis.model.atoms[first(psp_group)].psp
    position = basis.model.positions[first(psp_group)]
    projs = compute_pdos_projs(basis, eigenvalues, ψ, psp, position)

    D = zeros(typeof(ε[1]), length(ε), size(projs[1], 2))
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

function compute_pdos(scfres::NamedTuple; ε=scfres.εF, kwargs...)
    # TODO Require symmetrization with respect to kpoints and BZ symmetry 
    #      (now achieved by unfolding all the quantities).
    scfres = unfold_bz(scfres)
    psp_groups = [group for group in scfres.basis.model.atom_groups
                  if scfres.basis.model.atoms[first(group)] isa ElementPsp]
    [compute_pdos(ε, scfres.basis, scfres.eigenvalues, scfres.ψ, group; kwargs...)
     for group in psp_groups]
end

# Build projection matrix for a single atom
# Stored as projs[K][nk,lmi] = |<ψnk, f_lmi>|^2,
# where K is running over all k-points, l, m are AM quantum numbers, 
# and i is running over all pseudo-atomic wavefunctions for a given l.
function compute_pdos_projs(basis, eigenvalues, ψ, psp::PspUpf, position)
    position = vector_red_to_cart(basis.model, position)
    G_plus_k_all = [to_cpu(Gplusk_vectors_cart(basis, basis.kpoints[ik])) 
                    for ik = 1:length(basis.kpoints)]

    # Build Fourier transform factors of pseudo-wavefunctions centered at 0.
    fourier_form = atomic_centered_function_form_factors(psp, eval_psp_pswfc_fourier,
                                    count_n_pswfc_radial, count_n_pswfc, G_plus_k_all)

    projs = Vector{Matrix}(undef, length(basis.kpoints))
    for (ik, ψk) in enumerate(ψ)
        fourier_form_ik = fourier_form[ik]
        structure_factor_ik = exp.(-im .* [dot(position, Gik) for Gik in G_plus_k_all[ik]])
        @views for iproj = 1:size(fourier_form_ik, 2)
            # TODO Pseudo-atomic wave functions need to be orthogonalized.
            fourier_form_ik[:, iproj] .= (structure_factor_ik .* fourier_form_ik[:, iproj]
                                          ./ sqrt(basis.model.unit_cell_volume))
        end
        projs[ik] = abs2.(ψk' * fourier_form_ik)
    end
    projs
end

"""
Plot the density of states over a reasonable range. Requires to load `Plots.jl` beforehand.
"""
function plot_dos end

function plot_ldos end

function plot_pdos end
