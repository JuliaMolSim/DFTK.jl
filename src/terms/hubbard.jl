function build_projectors(basis::PlaneWaveBasis{T},
                    psps::AbstractVector{<: NormConservingPsp}, positions, labels::Vector{Tuple{Int,Int,Int,Int}}) where {T}
    """
        Build the projection matrices projsk for all k-points at the same time.
         projectors[ik][iband, iproj] = projsk[iband, iproj] = |<ψnk, ϕinlm>|^2
         where ψnk is the wavefunction for band iband at k-point kpt,
         and ϕinlm is the atomic orbital for atom i, quantum numbers (n,l,m)
        
         Input:
         - basis: PlaneWaveBasis
         - ψ: wavefunctions of the basis 
         - psps: vector of pseudopotentials for each atom
         - positions: positions of the atoms in the unit cell
         - labels: vector of tuples (iatom, n, l, m) for each projector
         Output:
         - projectors: vector of matrices of projectors, where 
                   projectors[ik][iband, iproj] = |ϕinlm>(iband,kpt) for each kpoint kpt
    """
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
            push!(proj_vectors, proj_vector)  # Collect all projection vectors for this k-point    
        end

        proj_vectors = hcat(proj_vectors...)  # Create a matrix of projections for this k-point
        @assert size(proj_vectors, 2) == nprojs "Projection matrix size mismatch: $(size(proj_vectors)) != $nprojs"
        proj_vectors = ortho_lwd(proj_vectors)  # Lowdin-orthogonal
        
        projectors[ik] = proj_vectors  # Contract on ψk to get the projections
    end

    return projectors
end

function build_projectors(bands)

    basis = bands.basis
    model = basis.model
    positions = model.positions
    natoms = length(positions)
    psps = Vector{NormConservingPsp}([model.atoms[i].psp for i in 1:natoms])  # PSP for all atoms
    
    max_l = Vector{Any}(undef, natoms)  # lmax for all atoms
    max_l = [psps[iatom].lmax for iatom in 1:natoms]  # lmax for all atoms
    max_n = Vector{Vector{Int}}(undef, natoms)  # Principal quantum number for all atoms
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

    projectors = build_projectors(basis, psps, positions, projector_labels)

    return (; projectors, projector_labels)
end

function compute_overlap_matrix(basis::PlaneWaveBasis{T},
                    psps::AbstractVector{<: NormConservingPsp}, positions, labels::Vector{Tuple{Int,Int,Int,Int}}) where {T}
    
    projectors = build_projectors(basis, psps, positions, labels)
    overlap_matrix = Vector{Matrix{T}}(undef, length(basis.kpoints))  # Initialize the density matrix

    for (ik, projk) in enumerate(projectors)
        #@show "Processing k-point $(ik) with projk size $(size(projk))"
        #density_matrix[ik] = projk' * projk  # Compute the density matrix for this k-point
        overlap_matrix[ik] = abs2.(projk' * projk)  # Compute the density matrix for this k-point
    end

    return overlap_matrix
end

function compute_overlap_matrix(bands)

    basis = bands.basis
    model = basis.model
    positions = model.positions
    natoms = length(positions)
    psps = Vector{NormConservingPsp}([model.atoms[i].psp for i in 1:natoms])  # PSP for all atoms
    
    max_l = Vector{Any}(undef, natoms)  # lmax for all atoms
    max_l = [psps[iatom].lmax for iatom in 1:natoms]  # lmax for all atoms
    max_n = Vector{Vector{Int}}(undef, natoms)  # Principal quantum number for all atoms
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

function compute_hubbard_matrix(manifold::NTuple{3,Int64}, basis::PlaneWaveBasis{T}, ψ,
                    psps::AbstractVector{<: NormConservingPsp}, positions, labels::Vector{Tuple{Int,Int,Int,Int}}) where {T}
    
    projectors = build_projectors(basis, psps, positions, labels)
    nprojs = length(labels)
    density_matrix = zeros(Complex{T}, nprojs, nprojs)  # Initialize the density matrix

    for (ik, projk) in enumerate(projectors)
        ψk = ψ[ik]
        n_mat_k = zeros(Complex{T}, size(projk, 2), size(projk, 2))  # Initialize the density matrix for this k-point
        for iband in 1:size(ψk, 2)
            ψnk = ψk[:, iband]  # Get the wavefunction for this band
            n_mat_k += projk' * ψnk * ψnk' * projk  # Compute the density matrix for this band
        end
        density_matrix += n_mat_k * basis.kweights[ik] # Store the summed density matrix for this k-point
    end

    dim_manifold = 0
    manifold_labels = Vector{NTuple{4, Int64}}(undef, 0)
    for orbital in labels
        if manifold[1] == orbital[1] && manifold[2] == orbital[2] && manifold[3] == orbital[3]
            dim_manifold += 1
            push!(manifold_labels, orbital)  # Collect labels for the manifold
        end
    end
    hubbard_matrix = zeros(Complex{T}, dim_manifold, dim_manifold)  # Initialize the Hubbard matrix
    ih = 1
    for (i, orbital_i) in enumerate(labels)
        jh = 1
        if manifold[1] == orbital_i[1] && manifold[2] == orbital_i[2] && manifold[3] == orbital_i[3]
            for (j, orbital_j) in enumerate(labels)
                if manifold[1] == orbital_j[1] && manifold[2] == orbital_j[2] && manifold[3] == orbital_j[3]
                    hubbard_matrix[ih, jh] = density_matrix[i, j]
                    jh += 1  # Increment the index for the manifold
                end
            end
            ih += 1  # Increment the index for the manifold
        end
    end

    return hubbard_matrix, manifold_labels
end

function compute_hubbard_matrix(manifold, bands)

    ψ = bands.ψ
    basis = bands.basis
    model = basis.model
    positions = model.positions
    natoms = length(positions)
    psps = Vector{NormConservingPsp}([model.atoms[i].psp for i in 1:natoms])  # PSP for all atoms
    
    max_l = Vector{Any}(undef, natoms)  # lmax for all atoms
    max_l = [psps[iatom].lmax for iatom in 1:natoms]  # lmax for all atoms
    max_n = Vector{Vector{Int}}(undef, natoms)  # Principal quantum number for all atoms
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

    hubbard_matrix, manifold_labels = compute_hubbard_matrix(manifold, basis, ψ, psps, positions, projector_labels)

    return (; hubbard_matrix, manifold_labels)
end


# Questo operatore non serve ad una mazza, almeno per ora. 
# Anzi scritto così è pericoloso perché probabilmen riempie la RAM

# !!!!!!NON USARE!!!!!!!!!!!

#function projection_operator(basis::PlaneWaveBasis{T},
#                    psps::AbstractVector{<: NormConservingPsp}, 
#                    positions, 
#                    labels::Vector{Tuple{Int,Int,Int,Int}}) where {T}
#    
#    projectors = build_projectors(basis, psps, positions, labels)
#    nproj = length(labels)
#    P_mat = Matrix{Vector{Matrix{Complex{T}}}}(undef, nproj, nproj)  # Initialize the projection matrix
#
#    for i = 1:nproj
#        for j in 1:nproj
#            P_ij = Vector{Matrix{Complex{T}}}(undef, length(basis.kpoints))  # Initialize the outer product matrix for this pair of projectors
#            for ik in 1:length(basis.kpoints)
#                projk = projectors[ik]
#                P_ij[ik] = projk * projk'   # Compute the outer product
#            end
#            P_mat[i, j] = P_ij  # Store the outer product in the matrix
#        end
#    end
#    
#    return P_mat
#end
#
#function projection_operator(bands)
#
#    basis = bands.basis
#    model = basis.model
#    positions = model.positions
#    natoms = length(positions)
#    psps = Vector{NormConservingPsp}([model.atoms[i].psp for i in 1:natoms])  # PSP for all atoms
#    
#    max_l = Vector{Any}(undef, natoms)  # lmax for all atoms
#    max_l = [psps[iatom].lmax for iatom in 1:natoms]  # lmax for all atoms
#    max_n = Vector{Vector{Int}}(undef, natoms)  # Principal quantum number for all atoms
#    for iatom in 1:natoms
#        max_n[iatom] = [DFTK.count_n_pswfc_radial(psps[iatom], l) for l in 0:max_l[iatom]]
#    end
#    
#    projector_labels = Vector{NTuple{4, Int64}}(undef, 0)
#    for iatom in 1:natoms
#        for l in 0:max_l[iatom]
#            for n in 1:max_n[iatom][l+1]
#                for m in -l:l
#                    push!(projector_labels, (iatom, n, l, m))
#                end 
#            end
#        end
#    end
#    nprojs = length(projector_labels) 
#
#    proj_matrix = projection_operator(basis, psps, positions, projector_labels)
#
#    return (; proj_matrix, projector_labels)
#end
