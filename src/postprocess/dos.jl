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
Compute the projected density of states (PDOS) for all atoms and orbitals.
 Input: 
 - εs: vector of energies at which to compute the PDOS
 - bands: Bands object containing the eigenvalues, wavefunction, basis and positions
 Output:
 - pdos: 3D array of PDOS, pdos[iε_idx, iproj, σ] = PDOS at energy εs[iε_idx] for projector iproj and spin σ
 - projector_labels: vector of tuples (iatom, n, l, m) for each projector, that maps the iproj index to the 
                    corresponding atomic orbital (atom index, principal quantum number, angular momentum, magnetic quantum number)
Note: 
 - The pdos matrix has different projectors for each atom, even if they are of the same atom type. 
   As such, the sum of all iproj columns for each σ yields the total DOS at each energy εs[iε_idx].
   This is different from Quantum ESPRESSO, where the pdos for atoms of the same type are summed together 
     even though they are printed separately (i.e. summing over all QE pdos from all output files does not yield the DOS).
"""

function compute_pdos(εs, basis::PlaneWaveBasis{T}, ψ, eigenvalues,
                      psps::AbstractVector{<: NormConservingPsp}, # Is it needed? It's already in basis.model.atoms
                      positions;                                  # Is it needed? It's already in basis.model.positions
                      smearing=basis.model.smearing, 
                      temperature=basis.model.temperature) where {T}
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_pdos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    projections, labels = build_projections(basis, ψ)
    nprojs = length(labels)

    D = zeros(typeof(εs[1]), length(εs), nprojs, basis.model.n_spin_components)  
    for (iε, ε) in enumerate(εs)
        for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
            projsk = projections[ik]  
            @views for (iband, εnk) in enumerate(eigenvalues[ik])
                enred = (εnk - ε) / temperature
                # Loop over all projections
                for iproj in 1:size(projsk, 2)
                    projk = projsk[:, iproj]
                    D[iε, iproj,σ] -= (filled_occ * basis.kweights[ik] * projk[iband]
                                             ./ temperature
                                             .* Smearing.occupation_derivative(smearing, enred))
                end
            end
        end
    end
    pdos = mpi_sum(D, basis.comm_kpts)  # Sum over all k-points

    return (; pdos, projector_labels=labels)
end

function compute_pdos(εs, bands; kwargs...)
    psps = Vector{NormConservingPsp}([bands.basis.model.atoms[i].psp for i in 1:length(bands.basis.model.atoms)])  # PSP for all atoms
    
    compute_pdos(εs, bands.basis, bands.ψ, bands.eigenvalues, psps, bands.basis.model.positions; kwargs...)
end

# TODO: function compute_single_pdos, or add optional arguments to the previous function to make it print only one pdos

"""
Build the projection matrices projsk for all k-points at the same time.
   projs[ik][iband, iproj] = projsk[iband, iproj] = |<ψnk, ϕilm>|^2
 where ψnk is the wavefunction for band iband at k-point kpt,
 and ϕilm is the atomic orbital for atom i, quantum numbers (n,l,m)

 Input:
 - basis: PlaneWaveBasis
 - ψ: wavefunctions of the basis 
 - psps: vector of pseudopotentials for each atom
 - positions: positions of the atoms in the unit cell
 - labels: vector of tuples (iatom, n, l, m) for each projector
 Output:
 - projs: vector of matrices of projections, where 
           projs[ik][iband, iproj] = |<ψnk, ϕilm>|^2 for each kpoint kpt
"""

function build_projections(basis::PlaneWaveBasis{T}, ψ;
                            positions = basis.model.positions           
                          ) where {T}

    G_plus_k_all = [Gplusk_vectors(basis, basis.kpoints[ik])
                    for ik = 1:length(basis.kpoints)]
    G_plus_k_all_cart = [map(recip_vector_red_to_cart(basis.model), gpk) 
                         for gpk in G_plus_k_all]

    psps = Vector{NormConservingPsp}(undef, length(basis.model.atoms))
    labels = []
    #form_factors = [Matrix{Complex{T}}(undef, length(G_plus_k), nprojs)  for G_plus_k in G_plus_k_all_cart]  # Initialize form factors for all k-points
    form_factors = [Matrix{Complex{T}}(undef, length(G_plus_k), 0)  for G_plus_k in G_plus_k_all_cart]
    for (iatom, atom) in enumerate(basis.model.atoms)
        psps[iatom] = atom.psp
        for l in 0:psps[iatom].lmax
            for n in 1:DFTK.count_n_pswfc_radial(psps[iatom], l)
                fun(p) = eval_psp_pswfc_fourier(psps[iatom], n, l, p)
                form_factors_l = build_form_factors(fun, l, G_plus_k_all_cart)
                for ik in 1:length(G_plus_k_all_cart)
                   #form_factors[ik][:,offset:offset+2*l] = form_factors_l[ik] #Can't work here because I don't know nproj a priori
                   form_factors[ik] = hcat(form_factors[ik], form_factors_l[ik])  # Concatenate the form factors for this l
                end
                for m in -l:l
                    label = psps[iatom].pswfc_labels[n+l]
                    push!(labels, (; iatom, atom.species, n, l, m, label))
                end
            end
        end
    end
    nprojs = length(labels)

    projs = Vector{Matrix}(undef, length(basis.kpoints))
    for (ik, ψk) in enumerate(ψ) # The loop now iterates over k-points regardless of the spin component
        proj_vectors = Matrix{Complex{T}}(undef, length(G_plus_k_all[ik]), nprojs)  # Collect all projection vectors for this k-point
        for (iproj, proj) in enumerate(labels)
            structure_factor = [cis2pi(-dot(positions[proj.iatom], p)) for p in G_plus_k_all[ik]]
            @assert length(structure_factor) == length(G_plus_k_all[ik]) "Structure factor length mismatch: $(length(structure_factor)) != $(length(G_plus_k))"
            proj_vectors[:,iproj] = structure_factor .* form_factors[ik][:, iproj] ./ sqrt(basis.model.unit_cell_volume)    
        end

        @assert size(proj_vectors, 2) == nprojs "Projection matrix size mismatch: $(size(proj_vectors)) != $nprojs"
        # At this point proj_vectors is a matrix containing all orbital projectors from all atoms. 
        #   What we want is to have them all orthogonal, to avoid double counting in the Hubbard U term contribution.
        #   We use Lowdin orthogonalization to minimize the "identity loss" of individual orbital projectors after the orthogonalization
        proj_vectors = ortho_lowdin(proj_vectors)  # Lowdin-orthogonal
        
        projs[ik] = abs2.(ψk' * proj_vectors)   # Contract on ψk to get the projections
        @assert size(projs[ik]) == (size(ψk,2), nprojs) "Projection matrix size mismatch: $(size(projsk)) != $(length(ψk)), $nprojs"
    end

    (;projs, labels)
end

function get_pdos(res, atom::Integer, n, l, spin)
    idx = findall(orb -> (orb.iatom==atom && orb.n==n && orb.l==l), res.projector_labels)
    pdos_values = zeros(Float64, length(εs))
    for i in idx
        pdos_values += res.pdos[:, i, spin]
    end
    return [((ε .- eshift) .* to_unit, p) for (ε, p) in zip(εs, pdos_values)]
end

function get_pdos(res, atom::Integer, l, spin)
    idx = findall(orb -> (orb.iatom==atom && orb.l==l), res.projector_labels)
    pdos_values = zeros(Float64, length(εs))
    for i in idx
        pdos_values += res.pdos[:, i, spin]
    end
    return [((ε .- eshift) .* to_unit, p) for (ε, p) in zip(εs, pdos_values)]
end

function get_pdos_spinless(res, atom::Integer, l)
    idx = findall(orb -> (orb.iatom==atom && orb.l==l), res.projector_labels)
    pdos_values = zeros(Float64, length(εs))
    for i in idx
        pdos_values += res.pdos[:, i, 1]
    end
    return [((ε .- eshift) .* to_unit, p) for (ε, p) in zip(εs, pdos_values)]
end

function get_pdos(res, atom::Symbol, n, l, spin)
    idx = findall(orb -> (orb.species==atom && orb.n==n && orb.l==l), res.projector_labels)
    pdos_values = zeros(Float64, length(εs))
    for i in idx
        pdos_values += res.pdos[:, i, spin]
    end
    return [((ε .- eshift) .* to_unit, p) for (ε, p) in zip(εs, pdos_values)]
end

function get_pdos_spinless(res, atom::Symbol, n, l)
    idx = findall(orb -> (orb.species==atom && orb.n==n && orb.l==l), res.projector_labels)
    pdos_values = zeros(Float64, length(εs))
    for i in idx
        pdos_values += res.pdos[:, i, 1]
    end
    return [((ε .- eshift) .* to_unit, p) for (ε, p) in zip(εs, pdos_values)]
end

function get_pdos_spinless(res, atom::Symbol, l)
    idx = findall(orb -> (orb.species==atom && orb.l==l), res.projector_labels)
    pdos_values = zeros(Float64, length(εs))
    for i in idx
        pdos_values += res.pdos[:, i, 1]
    end
    return [((ε .- eshift) .* to_unit, p) for (ε, p) in zip(εs, pdos_values)]
end

"""
Plot the density of states over a reasonable range. Requires to load `Plots.jl` beforehand.
"""
function plot_dos end

function plot_ldos end

function plot_pdos end
