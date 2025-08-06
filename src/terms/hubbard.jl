using LinearAlgebra

"""
    Build the projection matrices projsk for all k-points at the same time.
     projection[ik][iband, iproj] = projsk[iband, iproj] = |<ψnk, ϕinlm>|^2
     where ψnk is the wavefunction for band iband at k-point kpt,
     and ϕinlm is the atomic orbital for atom i, quantum numbers (n,l,m)
    
     Input:
     - basis: PlaneWaveBasis
     - ψ: wavefunctions of the basis 
     - psps: vector of pseudopotentials for each atom
     - positions: positions of the atoms in the unit cell
     - labels: structure containing iatom, species, n, l, m and orbital name for each projector
     Note: 'n' is not exactly the principal quantum number, but rather the index of the radial function in the pseudopotential.
           As an example, if the pseudopotential contains the 3S and 4S orbitals, then those are indexed as n=1, l=0 and n=2, l=0 respectively.
     Output:
     - projectors: vector of matrices of projectors, where 
               projectors[ik][iband, iproj] = |ϕinlm>(iband,kpt) for each kpoint kpt
"""
function build_projectors(basis::PlaneWaveBasis{T};
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
                    label = psps[iatom].pswfc_labels[n+l][1]
                    push!(labels, (; iatom, atom.species, n, l, m, label))
                end
            end
        end
    end
    nprojs = length(labels)

    projectors = Vector{Matrix}(undef, length(basis.kpoints))
    for ik in 1:length(basis.kpoints) # The loop now iterates over k-points regardless of the spin component
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
        
        projectors[ik] = proj_vectors  # Contract on ψk to get the projections
    end

    return (;projectors, labels)
end

function compute_overlap_matrix(basis::PlaneWaveBasis{T};
                                positions = basis.model.positions
                                ) where {T}
    
    proj = build_projectors(basis; positions) # Get the projectors for all k-points
    projectors = proj.projectors
    labels = proj.labels
    overlap_matrix = Vector{Matrix{T}}(undef, length(basis.kpoints))  # Initialize the density matrix

    for (ik, projk) in enumerate(projectors)
        #@show "Processing k-point $(ik) with projk size $(size(projk))"
        #density_matrix[ik] = projk' * projk  # Compute the density matrix for this k-point
        overlap_matrix[ik] = abs2.(projk' * projk)  # Compute the density matrix for this k-point
    end

    return (;overlap_matrix, projectors, labels)
end

"""
   This function computes the Hubbard matrix for a given manifold and basis.

   Input:
    - manifold: a tuple (species, orbital) defining the manifold. At the moment it only support one species and one orbital.
    - basis: PlaneWaveBasis containing the wavefunctions and k-points
    - ψ: wavefunctions from the scf calculation
    - (optional) positions: positions of the atoms in the unit cell
   Output:
    - hubbard_matrix: a matrix containing the Hubbard interaction terms for the specified manifold
    - manifold_labels: labels for the projectors in the Hubbard matrix, which is made like the labels in `build_projectors`

"""
function compute_hubbard_matrix(manifold::Tuple{Symbol, String}, basis::PlaneWaveBasis{T}, ψ;
                                positions = basis.model.positions
                                ) where {T}
    
    projectors, labels = build_projectors(basis; positions).projectors, build_projectors(basis; positions).labels
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
    manifold_labels = Vector{NamedTuple}(undef, 0)
    for orbital in labels
        if manifold[1] == Symbol(orbital.species) && lowercase(manifold[2]) == lowercase(orbital.label)
            dim_manifold += 1
            push!(manifold_labels, orbital)  # Collect labels for the manifold
        end
    end
    hubbard_matrix = zeros(Complex{T}, dim_manifold, dim_manifold)  # Initialize the Hubbard matrix
    ih = 1
    for (i, orbital_i) in enumerate(labels)
        jh = 1
        if manifold[1] == Symbol(orbital_i.species) && lowercase(manifold[2]) == lowercase(orbital_i.label) 
            for (j, orbital_j) in enumerate(labels)
                if manifold[1] == Symbol(orbital_j.species) && lowercase(manifold[2]) == lowercase(orbital_j.label) 
                    hubbard_matrix[ih, jh] = density_matrix[i, j]
                    jh += 1  # Increment the index for the manifold
                end
            end
            ih += 1  # Increment the index for the manifold
        end
    end

    return (;hubbard_matrix, manifold_labels, projectors, labels, density_matrix)
end

function compute_hubbard_matrix(manifold, bands; positions = bands.basis.model.positions)

    ψ = bands.ψ
    basis = bands.basis
    
    hubbard_matrix = compute_hubbard_matrix(manifold, basis, ψ; positions).hubbard_matrix
    manifold_labels = compute_hubbard_matrix(manifold, basis, ψ; positions).manifold_labels

    return (; hubbard_matrix, manifold_labels)
end


# This part of the code is currently useless and even dangerour since it fills the RAM.

# !!!!!!DO NOT USE!!!!!!!!!!!

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
struct Hubbard
    manifold::Tuple{Symbol, String}
    U::Float64
end

struct TermHubbard <: Term
    manifold::Tuple{Symbol, String}
    U::Float64
end

(hubbard::Hubbard)(::AbstractBasis) = TermHubbard(hubbard.manifold, hubbard.U)

function ene_ops(term::TermHubbard, basis::PlaneWaveBasis{T}, ψ, occupation; kwargs...) where {T}
    if isnothing(ψ)
        return (; E=zero(T), ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
    end
    # Compute Hubbard matrix 
    hubbard_matrix = compute_hubbard_matrix(term.manifold, basis, ψ).hubbard_matrix
    E = term.U * real(tr(hubbard_matrix * (I - hubbard_matrix)))
    
    # Return proper operators
    ops = [NoopOperator(basis, kpt) for kpt in basis.kpoints]  # or proper Hubbard operators

    (; E, ops)
end