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
Compute the projected density of states (PDOS) for all atoms and orbitals
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
                      psps::AbstractVector{<: NormConservingPsp}, 
                      positions;
                      smearing=basis.model.smearing, 
                      temperature=basis.model.temperature) where {T}
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_pdos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)
          
    natoms = length(positions)
    projector_labels = Vector{NTuple{4, Int64}}(undef, 0)
    for iatom in 1:natoms
        for l in 0:psps[iatom].lmax
            for n in 1:DFTK.count_n_pswfc_radial(psps[iatom], l)
                for m in -l:l
                    push!(projector_labels, (iatom, n, l, m))
                end 
            end
        end
    end
    nprojs = length(projector_labels) 

    projectors = build_projections(basis, ψ)

    D = zeros(typeof(εs[1]), length(εs), nprojs, basis.model.n_spin_components)  
    for (iε, ε) in enumerate(εs)
        for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
            projsk = projectors[ik]  
            @views for (iband, εnk) in enumerate(eigenvalues[ik])
                enred = (εnk - ε) / temperature
                # Loop over all projectors
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

    return (; pdos, projector_labels)
end

function compute_pdos(εs, bands; kwargs...)
    psps = Vector{NormConservingPsp}([bands.basis.model.atoms[i].psp for i in 1:length(bands.basis.model.atoms)])  # PSP for all atoms
    compute_pdos(εs, bands.basis, bands.ψ, bands.eigenvalues, psps, bands.basis.model.positions; kwargs...)
end

# TODO: function compute_single_pdos, or add optional arguments to the previous function to make it print only one pdos

"""
Build the projection matrix projsk for a given k-point kpt.
    projsk[iband, iproj] = |<ψnk, ϕilm>|^2
 where ψnk is the wavefunction for band iband at k-point kpt,
 and ϕilm is the atomic orbital for atom i, quantum numbers (n,l,m)

 Input:
 - basis: PlaneWaveBasis
 - kpt: Kpoint for which to compute the projections
 - ψ: wavefunctions of the basis 
 - psps: vector of pseudopotentials for each atom
 - positions: positions of the atoms in the unit cell
 - labels: vector of tuples (iatom, n, l, m) for each projector
 Output:
 - projsk: matrix of projections, where projsk[iband, iproj]
           = |<ψnk, ϕilm>|^2 for the given kpoint kpt
"""

function build_projsk(basis::PlaneWaveBasis{T}, kpt::Kpoint, ψk,
                    psps::AbstractVector{<: NormConservingPsp}, 
                    positions, labels::Vector{Tuple{Int,Int,Int,Int}}) where {T}

    nprojs = length(labels)
    projsk = zeros(T, length(ψk), nprojs)  # Initialize the projection matrix

    G_plus_k = Gplusk_vectors(basis, kpt)
    G_plus_k_cart = map(recip_vector_red_to_cart(basis.model), G_plus_k)

    # Build form factors of pseudo-wavefunctions centered at 0.
    form_factors = Matrix{Complex{T}}(undef, length(G_plus_k_cart), nprojs)
    for (iproj, (iatom, n, l, m)) in enumerate(labels)
        psp = psps[iatom]
        fun(p) = eval_psp_pswfc_fourier(psp, n, l, p)
        # Build form factors for the current projector
        radials = IdDict{T,T}()  # IdDict for Dual compatibility
    
        for p in G_plus_k_cart
            p_norm = norm(p)
            if !haskey(radials, p_norm)
                radials_p = fun(p_norm)
                radials[p_norm] = radials_p
            end
        end

        for (ip, p) in enumerate(G_plus_k_cart)
            radials_p = radials[norm(p)]
            # see "Fourier transforms of centered functions" in the docs for the formula
            angular = (-im)^l * ylm_real(l, m, p)
            form_factors[ip, iproj] = radials_p * angular
        end
    end

    proj_vectors = Vector{Vector{Any}}(undef, 0)  # Collect all projection vectors for this k-point
    for (iproj, (iatom, n, l, m)) in enumerate(labels)
        structure_factor = [cis2pi(-dot(positions[iatom], p)) for p in G_plus_k_cart]
        proj_vector = structure_factor .* form_factors[:, iproj] ./ sqrt(basis.model.unit_cell_volume)
        push!(proj_vectors, proj_vector)  # Collect all projection vectors for this k-point    
    end

    proj_vectors = hcat(proj_vectors...)  # Create a matrix of projections for this k-point
    @assert size(proj_vectors, 2) == nprojs "Projection matrix size mismatch: $(size(proj_vectors)) != $nprojs"
    proj_vectors = ortho_lowdin(proj_vectors)  # Lowdin-orthogonalization
    
    projsk = abs2.(ψk' * proj_vectors)  # Contract on ψk to get the projections
    @assert size(projsk) == (size(ψk,2), nprojs) "Projection matrix size mismatch: $(size(projsk)) != $(length(ψk)), $nprojs"

    return projsk
end

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
 - labels: structure containing iatom, species, n, l, m and orbital name for each projector
 Note: 'n' is not exactly the principal quantum number, but rather the index of the radial function in the pseudopotential.
       As an example, if the pseudopotential contains the 3S and 4S orbitals, then those are indexed as n=1, l=0 and n=2, l=0 respectively.
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
    form_factors = [Matrix{Complex{T}}(undef, length(G_plus_k), 0)  for G_plus_k in G_plus_k_all_cart]
    for (iatom, atom) in enumerate(basis.model.atoms)
        psps[iatom] = atom.psp
        for l in 0:psps[iatom].lmax
            for n in 1:DFTK.count_n_pswfc_radial(psps[iatom], l)
                fun(p) = eval_psp_pswfc_fourier(psps[iatom], n, l, p)
                form_factors_l = build_form_factors(fun, l, G_plus_k_all_cart)
                for ik in 1:length(G_plus_k_all_cart)
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


"""
Plot the density of states over a reasonable range. Requires to load `Plots.jl` beforehand.
"""
function plot_dos end

function plot_ldos end

function plot_pdos end
