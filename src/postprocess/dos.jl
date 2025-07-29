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

function compute_all_pdos(εs, bands;
                      smearing=bands.basis.model.smearing, 
                      temperature=bands.basis.model.temperature)
# Compute the projected density of states (PDOS) for all atoms and orbitals
# Input: 
# - εs: vector of energies at which to compute the PDOS
# - bands: Bands object containing the eigenvalues, wavefunction, basis and positions
# Output:
# - pdos: 3D array of PDOS, pdos[iε_idx, iproj, σ] = PDOS at energy εs[iε_idx] for projector iproj and spin σ
# - projector_labels: vector of tuples (iatom, n, l, m) for each projector, that maps the iproj index to the 
#                    corresponding atomic orbital (atom index, principal quantum number, angular momentum, magnetic quantum number)

    if (temperature == 0) || smearing isa Smearing.None
        error("compute_pdos only supports finite temperature")
    end
          
    ψ = bands.ψ
    eigenvalues = bands.eigenvalues
    basis = bands.basis
    model = basis.model
    positions = model.positions
    natoms = length(positions)
    psps = Vector{NormConservingPsp}([model.atoms[i].psp for i in 1:natoms])  # PSP for all atoms
    
    max_l = Vector{Any}(undef, natoms)                
    max_l = [psps[iatom].lmax for iatom in 1:natoms]  # lmax for all atoms
    max_n = Vector{Vector{Int}}(undef, natoms)        # Principal quantum number for all atoms
    for iatom in 1:natoms
        max_n[iatom] = [DFTK.count_n_pswfc_radial(psps[iatom], l) for l in 0:max_l[iatom]]
    end
    
    projector_labels = Vector{NTuple{4, Int64}}(undef, 0)
    for iatom in 1:natoms
        for l in 0:max_l[iatom]
            for n in 1:max_n[iatom][l+1]
                for m in -l:l
                    push!(projector_labels, (iatom, n, l, m))
                end 
            end
        end
    end
    nprojs = length(projector_labels) 

    println("---Building projectors for $(nprojs) projectors...")
    projectors = build_projections(basis, ψ, psps, positions, projector_labels)

    println("---Computing PDOS for $(length(εs)) energies and $(nprojs) projectors...")
    D = zeros(typeof(εs[1]), length(εs), nprojs, model.n_spin_components)  
    for (iε_idx, iε) in enumerate(εs)
        for σ in 1:model.n_spin_components, ik = krange_spin(basis, σ)
            projsk = projectors[ik]  
            @views for (iband, εnk) in enumerate(eigenvalues[ik])
                enred = (εnk - iε) / temperature
                # Loop over all projectors
                for iproj in 1:size(projsk, 2)
                    projk = projsk[:, iproj]
                    D[iε_idx, iproj,σ] -= (basis.kweights[ik] * projk[iband]
                                             ./ temperature
                                             .* Smearing.occupation_derivative(smearing, enred))
                end
            end
        end
    end
    pdos = mpi_sum(D, basis.comm_kpts)  # Sum over all k-points

    return (; pdos, projector_labels)
end

# TODO: function compute_single_pdos, or add optional arguments to the previous function to make it print only one pdos

function build_projsk(basis::PlaneWaveBasis{T}, kpt::Kpoint, ψk,
                    psps::AbstractVector{<: NormConservingPsp}, positions, labels::Vector{Tuple{Int,Int,Int,Int}}) where {T}
# Build the projection matrix projsk for a given k-point kpt.
# projsk[iband, iproj] = |<ψnk, ϕilm>|^2
# where ψnk is the wavefunction for band iband at k-point kpt,
# and ϕilm is the atomic orbital for atom i, quantum numbers (n,l,m)
#
# Input:
# - basis: PlaneWaveBasis
# - kpt: Kpoint for which to compute the projections
# - ψ: wavefunctions of the basis 
# - psps: vector of pseudopotentials for each atom
# - positions: positions of the atoms in the unit cell
# - labels: vector of tuples (iatom, n, l, m) for each projector
# Output:
# - projsk: matrix of projections, where projsk[iband, iproj]
#           = |<ψnk, ϕilm>|^2 for the given kpoint kpt

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
    proj_vectors = ortho_lwd(proj_vectors)  # Lowdin-orthogonalization
    
    projsk = abs2.(ψk' * proj_vectors)  # Contract on ψk to get the projections
    @assert size(projsk) == (size(ψk,2), nprojs) "Projection matrix size mismatch: $(size(projsk)) != $(length(ψk)), $nprojs"

    return projsk
end

function build_projections(basis::PlaneWaveBasis{T}, ψ,
                    psps::AbstractVector{<: NormConservingPsp}, 
                    positions, 
                    labels::Vector{Tuple{Int,Int,Int,Int}}) where {T}
# Build the projection matrices projsk for all k-points at the same time.
# projs[ik][iband, iproj] = projsk[iband, iproj] = |<ψnk, ϕilm>|^2
# where ψnk is the wavefunction for band iband at k-point kpt,
# and ϕilm is the atomic orbital for atom i, quantum numbers (n,l,m)
#
# Input:
# - basis: PlaneWaveBasis
# - ψ: wavefunctions of the basis 
# - psps: vector of pseudopotentials for each atom
# - positions: positions of the atoms in the unit cell
# - labels: vector of tuples (iatom, n, l, m) for each projector
# Output:
# - projs: vector of matrices of projections, where 
#           projs[ik][iband, iproj] = |<ψnk, ϕilm>|^2 for each kpoint kpt

    nprojs = length(labels)
    projs = Vector{Matrix}(undef, length(basis.kpoints))  # Initialize the projection matrix

    G_plus_k_all = [Gplusk_vectors(basis, basis.kpoints[ik])
                    for ik = 1:length(basis.kpoints)]
    G_plus_k_all_cart = [map(recip_vector_red_to_cart(basis.model), gpk) 
                         for gpk in G_plus_k_all]
    
    # Build form factors of pseudo-wavefunctions centered at 0.
     
    form_factors = [Matrix{Complex{T}}(undef, length(G_plus_k), nprojs)  for G_plus_k in G_plus_k_all_cart]  # Initialize form factors for all k-points
    for (iproj, (iatom, n, l, m)) in enumerate(labels)
        psp = psps[iatom]
        fun(p) = eval_psp_pswfc_fourier(psp, n, l, p)
        # Build form factors for the current projector
        
        radials = IdDict{T,T}()  # IdDict for Dual compatibility
        for G_plus_k in G_plus_k_all_cart
            for p in G_plus_k
                p_norm = norm(p)
                if !haskey(radials, p_norm)
                    radials_p = fun(p_norm)
                    radials[p_norm] = radials_p
                end
            end
        end

        for (ik, G_plus_k) in enumerate(G_plus_k_all_cart)
            for (ip, p) in enumerate(G_plus_k)
                radials_p = radials[norm(p)]
                # see "Fourier transforms of centered functions" in the docs for the formula
                angular = (-im)^l * ylm_real(l, m, p)
                form_factors[ik][ip, iproj] = radials_p * angular
            end
        end
    end

    projs = Vector{Matrix}(undef, length(basis.kpoints))
    for (ik, ψk) in enumerate(ψ) # The loop now iterates over k-points regardless of the spin component
        proj_vectors = Vector{Vector{Any}}(undef, 0)
        for (iproj, (iatom, n, l, m)) in enumerate(labels)
            structure_factor = [cis2pi(-dot(positions[iatom], p)) for p in G_plus_k_all[ik]]
            @assert length(structure_factor) == length(G_plus_k_all[ik]) "Structure factor length mismatch: $(length(structure_factor)) != $(length(G_plus_k))"
            proj_vector = structure_factor .* form_factors[ik][:, iproj] ./ sqrt(basis.model.unit_cell_volume)
            push!(proj_vectors, proj_vector)    # Collect all projection vectors for this k-point    
        end

        proj_vectors = hcat(proj_vectors...)    # Create a matrix of projections for this k-point
        @assert size(proj_vectors, 2) == nprojs "Projection matrix size mismatch: $(size(proj_vectors)) != $nprojs"
        proj_vectors = ortho_lwd(proj_vectors)  # Lowdin-orthogonal
        
        projs[ik] = abs2.(ψk' * proj_vectors)   # Contract on ψk to get the projections
        @assert size(projs[ik]) == (size(ψk,2), nprojs) "Projection matrix size mismatch: $(size(projsk)) != $(length(ψk)), $nprojs"
    end

    return projs
end

function build_projectors(basis::PlaneWaveBasis{T},
                    psps::AbstractVector{<: NormConservingPsp}, 
                    positions, 
                    labels::Vector{Tuple{Int,Int,Int,Int}}) where {T}
# Build the projection matrices projsk for all k-points at the same time.
# projectors[ik][iband, iproj] = projsk[iband, iproj] = |<ψnk, ϕinlm>|^2
# where ψnk is the wavefunction for band iband at k-point kpt,
# and ϕinlm is the atomic orbital for atom i, quantum numbers (n,l,m)
#
# Input:
# - basis: PlaneWaveBasis
# - ψ: wavefunctions of the basis 
# - psps: vector of pseudopotentials for each atom
# - positions: positions of the atoms in the unit cell
# - labels: vector of tuples (iatom, n, l, m) for each projector
# Output:
# - projectors: vector of matrices of projectors, where 
#           projectors[ik][iband, iproj] = |ϕinlm>(iband,kpt) for each kpoint kpt

    nprojs = length(labels)
    projectors = Vector{Matrix}(undef, length(basis.kpoints))  # Initialize the projection matrix

    G_plus_k_all = [Gplusk_vectors(basis, basis.kpoints[ik])
                    for ik = 1:length(basis.kpoints)]
    G_plus_k_all_cart = [map(recip_vector_red_to_cart(basis.model), gpk) 
                         for gpk in G_plus_k_all]
    
    # Build form factors of pseudo-wavefunctions centered at 0.
     
    form_factors = [Matrix{Complex{T}}(undef, length(G_plus_k), nprojs)  for G_plus_k in G_plus_k_all_cart]  # Initialize form factors for all k-points
    for (iproj, (iatom, n, l, m)) in enumerate(labels)
        psp = psps[iatom]
        fun(p) = eval_psp_pswfc_fourier(psp, n, l, p)
        # Build form factors for the current projector
        
        radials = IdDict{T,T}()  # IdDict for Dual compatibility
        for G_plus_k in G_plus_k_all_cart
            for p in G_plus_k
                p_norm = norm(p)
                if !haskey(radials, p_norm)
                    radials_p = fun(p_norm)
                    radials[p_norm] = radials_p
                end
            end
        end

        for (ik, G_plus_k) in enumerate(G_plus_k_all_cart)
            for (ip, p) in enumerate(G_plus_k)
                radials_p = radials[norm(p)]
                # see "Fourier transforms of centered functions" in the docs for the formula
                angular = (-im)^l * ylm_real(l, m, p)
                form_factors[ik][ip, iproj] = radials_p * angular
            end
        end
    end

    projectors = Vector{Matrix}(undef, length(basis.kpoints))
    for (ik, G_plus_k) in enumerate(G_plus_k_all)  # The loop now iterates over k-points regardless of the spin component
        proj_vectors = Vector{Vector{Any}}(undef, 0)
        for (iproj, (iatom, n, l, m)) in enumerate(labels)
            structure_factor = [cis2pi(-dot(positions[iatom], p)) for p in G_plus_k]
            @assert length(structure_factor) == length(G_plus_k) "Structure factor length mismatch: $(length(structure_factor)) != $(length(G_plus_k))"
            proj_vector = structure_factor .* form_factors[ik][:, iproj] ./ sqrt(basis.model.unit_cell_volume)
            push!(proj_vectors, proj_vector)    # Collect all projection vectors for this k-point    
        end

        proj_vectors = hcat(proj_vectors...)    # Create a matrix of projections for this k-point
        @assert size(proj_vectors, 2) == nprojs "Projection matrix size mismatch: $(size(proj_vectors)) != $nprojs"
        proj_vectors = ortho_lwd(proj_vectors)  # Lowdin-orthogonal
        
        projectors[ik] = proj_vectors           # Contract on ψk to get the projections
    end

    return projectors
end

function compute_overlap_matrix(basis::PlaneWaveBasis{T},
                    psps::AbstractVector{<: NormConservingPsp}, 
                    positions, 
                    labels::Vector{Tuple{Int,Int,Int,Int}}) where {T}
    
    projectors = build_projectors(basis, psps, positions, labels)
    overlap_matrix = Vector{Matrix{T}}(undef, length(basis.kpoints))  

    for (ik, projk) in enumerate(projectors)
        overlap_matrix[ik] = abs2.(projk' * projk)  
    end

    return overlap_matrix
end

function compute_overlap_matrix(bands)

    basis = bands.basis
    model = basis.model
    positions = model.positions
    natoms = length(positions)
    psps = Vector{NormConservingPsp}([model.atoms[i].psp for i in 1:natoms])  # PSP for all atoms
    
    max_l = Vector{Any}(undef, natoms)  
    max_l = [psps[iatom].lmax for iatom in 1:natoms]  # lmax for all atoms
    max_n = Vector{Vector{Int}}(undef, natoms)        # Principal quantum number for all atoms
    for iatom in 1:natoms
        max_n[iatom] = [DFTK.count_n_pswfc_radial(psps[iatom], l) for l in 0:max_l[iatom]]
    end
    
    projector_labels = Vector{NTuple{4, Int64}}(undef, 0)
    for iatom in 1:natoms
        for l in 0:max_l[iatom]
            for n in 1:max_n[iatom][l+1]
                for m in -l:l
                    push!(projector_labels, (iatom, n, l, m))
                end 
            end
        end
    end
    nprojs = length(projector_labels) 

    S_mat = compute_overlap_matrix(basis, psps, positions, projector_labels)

    return (; S_mat, projector_labels)
end

function compute_density_matrix(basis::PlaneWaveBasis{T}, ψ,
                    psps::AbstractVector{<: NormConservingPsp}, 
                    positions, 
                    labels::Vector{Tuple{Int,Int,Int,Int}}) where {T}
    
    projectors = build_projectors(basis, psps, positions, labels)
    density_matrix = Vector{Matrix{T}}(undef, length(basis.kpoints))  

    for (ik, projk) in enumerate(projectors)
        ψk = ψ[ik]  
        density_matrix[ik] = (ψk' * projk) * (projk' * ψk)  
    end

    return density_matrix
end

function compute_density_matrix(bands)

    ψ = bands.ψ
    basis = bands.basis
    model = basis.model
    positions = model.positions
    natoms = length(positions)
    psps = Vector{NormConservingPsp}([model.atoms[i].psp for i in 1:natoms])  # PSP for all atoms
    
    max_l = Vector{Any}(undef, natoms)  
    max_l = [psps[iatom].lmax for iatom in 1:natoms]  # lmax for all atoms
    max_n = Vector{Vector{Int}}(undef, natoms)        # Principal quantum number for all atoms
    for iatom in 1:natoms
        max_n[iatom] = [DFTK.count_n_pswfc_radial(psps[iatom], l) for l in 0:max_l[iatom]]
    end
    
    projector_labels = Vector{NTuple{4, Int64}}(undef, 0)
    for iatom in 1:natoms
        for l in 0:max_l[iatom]
            for n in 1:max_n[iatom][l+1]
                for m in -l:l
                    push!(projector_labels, (iatom, n, l, m))
                end 
            end
        end
    end
    nprojs = length(projector_labels) 

    n_mat = compute_density_matrix(basis, ψ, psps, positions, projector_labels)

    return (; n_mat, projector_labels)
end


"""
Plot the density of states over a reasonable range. Requires to load `Plots.jl` beforehand.
"""
function plot_dos end

function plot_ldos end

function plot_pdos end
