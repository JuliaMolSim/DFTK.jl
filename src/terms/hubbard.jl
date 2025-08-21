using LinearAlgebra
#using WignerD

"""
    Build the projectors matrices projsk for all k-points at the same time.

             projector[ik][:, iproj] = |ϕinlm>(kpt)

     where ϕinlm is the atomic orbital for atom i, quantum numbers (n,l,m)
       and iproj is the corresponding column index. The mapping is recorded in 'labels'.
     Note: 'n' is not exactly the principal quantum number, but rather the index of the radial function in the pseudopotential.
           As an example, if the pseudopotential contains the 3S and 4S orbitals, then those are indexed as n=1, l=0 and n=2, l=0 respectively.
    
     Input: 
     - basis           : PlaneWaveBasis
     - manifold  (opt) : tuple of (Atom, Orbital) to select only a subset of orbitals for the computation. 'Atom' must be either a Symbol or an Int64, 'Orbital' must be a String with the orbital name in uppercase.
     - positions (opt) : positions of the atoms in the unit cell
     Output:
     - projectors      : vector of matrices of projectors
     - labels          : structure containing iatom, species, n, l, m and orbital name for each projector
"""
function build_projectors(basis::PlaneWaveBasis{T};
                          manifold = nothing,
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
        if !isnothing(manifold)
            if manifold[1] != Symbol(atom.species) && manifold[1] != iatom
               continue # Skip atoms that do not match the manifold species, if any is provided
            end
        end
        psps[iatom] = atom.psp
        for l in 0:psps[iatom].lmax
            for n in 1:DFTK.count_n_pswfc_radial(psps[iatom], l)
                label = psps[iatom].pswfc_labels[l+1][n]
                if !isnothing(manifold) && lowercase(manifold[2]) != lowercase(label)
                    continue # Skip atoms that do not match the manifold species, if any is provided
                end
                fun(p) = eval_psp_pswfc_fourier(psps[iatom], n, l, p)
                form_factors_l = build_form_factors(fun, l, G_plus_k_all_cart)
                for ik in 1:length(G_plus_k_all_cart)
                   form_factors[ik] = hcat(form_factors[ik], form_factors_l[ik])  # Concatenate the form factors for this l
                end
                for m in -l:l
                    push!(labels, (; iatom, atom.species, n, l, m, label))
                end
            end
        end
    end
    nprojs = length(labels)

    projectors = Vector{Matrix}(undef, length(basis.kpoints))
    for ik in 1:length(basis.kpoints) # The projectors don't depend on the spin
        proj_vectors = zeros(Complex{T}, length(G_plus_k_all[ik]), nprojs)  # Collect all projection vectors for this k-point
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

function build_manifold(basis::PlaneWaveBasis{T}, projectors, labels, manifold::Tuple{Symbol, String}) where {T}
    manifold_labels = []
    manifold_projectors = Vector{Matrix{Complex{T}}}(undef, length(basis.kpoints))
    #for (iatom, atom) in enumerate(basis.model.atoms)
    #    if manifold[1] == Symbol(atom.species)
    #        psps[iatom] = atom.psp
    #        for l in 0:psps[iatom].lmax
    #            for n in 1:DFTK.count_n_pswfc_radial(psps[iatom], l)
    #                label = psps[iatom].pswfc_labels[n+l][1]
    #                if lowercase(manifold[2]) == lowercase(label)
    #                    # If the label matches the manifold, we add it to the labels
    #                    # This is useful for extracting specific orbitals from the basis
    #                    # e.g., (:Si, "3S") will match all 3S orbitals of Si atoms
    #                    for m in -l:l
    #                        push!(manifold_labels, (; iatom, atom.species, n, l, m, label))
    #                    end
    #                end
    #            end
    #        end
    #    end
    #end
    for (iproj, orb) in enumerate(labels)
        if manifold[1] == Symbol(orb.species) && lowercase(manifold[2]) == lowercase(orb.label)
            # If the label matches the manifold, we add it to the labels
            # This is useful for extracting specific orbitals from the basis
            # e.g., (:Si, "3S") will match all 3S orbitals of Si atoms
            for m in -orb.l:orb.l
                push!(manifold_labels, (; orb.iatom, orb.species, orb.n, orb.l, m, orb.label))
            end
        end
    end
    for (ik, projk) in enumerate(projectors)
        manifold_projectors[ik] = zeros(Complex{T}, size(projectors[ik], 1), length(manifold_labels))
        for (iproj, orb) in enumerate(manifold_labels)
            # Find the index of the projector that matches the manifold label
            proj_index = findfirst(p -> p.iatom == orb.iatom && p.species == orb.species &&
                                       p.n == orb.n && p.l == orb.l && p.m == orb.m, labels)
            if proj_index !== nothing
                manifold_projectors[ik][:, iproj] = projk[:, proj_index]
            else
                @warn "Projector for $(orb.species) with n=$(orb.n), l=$(orb.l), m=$(orb.m) not found in projectors."
            end
        end
    end
    return (; manifold_labels, manifold_projectors)
end

function compute_overlap_matrix(basis::PlaneWaveBasis{T};
                                manifold::Tuple{Symbol, String} = nothing,
                                positions = basis.model.positions
                                ) where {T}
    
    proj = build_projectors(basis; manifold, positions) # Get the projectors for all k-points
    projectors = proj.projectors
    labels = proj.labels
    overlap_matrix = Vector{Matrix{T}}(undef, length(basis.kpoints))  # Initialize the density matrix

    for (ik, projk) in enumerate(projectors)
        overlap_matrix[ik] = abs2.(projk' * projk)  # Compute the density matrix for this k-point
    end

    return (;overlap_matrix, projectors, labels)
end

"""
Symmetrize the Hubbard occupation matrix according to the l quantum number of the manifold.
"""
function symmetrize(n_IJ::Array{Matrix{Complex{T}}}, lattice, symmetry, l, positions) where {T}
    # For now we apply symmetries only on nII terms, not on cross-atom terms (nIJ)
    # WARNING: To implement +V this will need to be changed!

    nspins = size(n_IJ, 1)
    natoms = size(n_IJ, 2)
    nsym = length(symmetry)
    d1, d2, d3 = Wigner_sym(1, lattice, symmetry), Wigner_sym(2, lattice, symmetry), Wigner_sym(3, lattice, symmetry)

     # Initialize the n_IJ matrix
    ns = Array{Matrix{Complex{T}}}(undef, nspins, natoms, natoms) 
    for σ in 1:nspins, iatom in 1:natoms, jatom in 1:natoms
        ns[σ, iatom, jatom] = zeros(Complex{T}, size(n_IJ[σ, iatom, jatom],1), size(n_IJ[σ, iatom, jatom],2))
    end

    # TODO: Write better the symmetrization loop
    for σ in 1:nspins, iatom in 1:natoms
        for m1 in 1:size(ns[σ, iatom, iatom], 1), m2 in 1:size(ns[σ, iatom, iatom], 2)  # Iterate over the rows of the n_IJ matrix
            for isym in 1:nsym
                sym_atom = find_symmetric(iatom, symmetry, isym, positions)
                # TODO: Here QE flips spin for time-reversal in collinear systems, should we?
                for m0 in 1:size(n_IJ[σ, iatom, iatom], 1), m00 in 1:size(n_IJ[σ, iatom, iatom], 2)
                    if l == 0
                        # For s-orbitals we only average over symmetric atoms 
                        ns[σ, iatom, iatom][m1, m2] += n_IJ[σ, sym_atom, sym_atom][m0, m00] / nsym
                    elseif l == 1
                        # apply d1 Wigner matrix
                        ns[σ, iatom, iatom][m1, m2] += d1[m0, m1, isym] * 
                                                       n_IJ[σ, sym_atom, sym_atom][m0, m00] * 
                                                       d1[m00, m2, isym] / nsym
                    elseif l == 2
                        # apply d2 Wigner matrix
                        ns[σ, iatom, iatom][m1, m2] += d2[m0, m1, isym] * 
                                                       n_IJ[σ, sym_atom, sym_atom][m0, m00] * 
                                                       d2[m00, m2, isym] / nsym
                    elseif l == 3
                        # apply d3 Wigner matrix
                        ns[σ, iatom, iatom][m1, m2] += d3[m0, m1, isym] * 
                                                       n_IJ[σ, sym_atom, sym_atom][m0, m00] * 
                                                       d3[m00, m2, isym] / nsym
                    else
                        @warn "Symmetrization for l > 3 not implemented yet, skipping symmetrization for l=$(l)"
                        # For now we skip symmetrization for l > 3
                    end
                end
            end
        end
    end

    return ns
end
"""
Find the symmetric atom index for a given atom and symmetry operation
"""
function find_symmetric(iatom::Int64, symmetry::Vector{SymOp{T}}, 
                        isym::Int64, positions
                        ) where {T}
    sym_atom = iatom
    W, w = symmetry[isym].W, symmetry[isym].w
    p = positions[iatom]
    p2 = W * p + w  
    for (jatom, pos) in enumerate(positions)
        if isapprox(pos, p2, atol=1e-8)  
            sym_atom = jatom
            break
        end
    end
    return sym_atom
end

"""
 This function returns the Wigner matrix for a given l and symmetry operation solving a randomized linear system.
    The lattice L is needed to convert reduced symmetries to Cartesian space.
"""
function Wigner_sym(l::Int64, L, symmetries::Vector{SymOp{T}}) where {T}
    
    nsym = length(symmetries)
    D = Array{Float64}(undef, 2*l+1, 2*l+1, nsym)
    for (isym,symmetry) in enumerate(symmetries)
        W = symmetry.W
        for m1 in -l:l
            b = Vector{Float64}(undef, 2*l+1)
            A = Matrix{Float64}(undef, 2*l+1, 2*l+1)
            for n in 1:2*l+1
                r = rand(Float64, 3)
                r0 = L * W * inv(L) * r
                b[n] = DFTK.ylm_real(l, m1, r0)
                for m2 in -l:l
                    A[n,m2+l+1] = DFTK.ylm_real(l, m2, r)
                end
            end
            D[m1+l+1,:,isym] = A\b
        end
    end

    return D
end

"""
Computes a matrix n_IJ of size (nspins, natoms, natoms), where each entry n_IJ[iatom, jatom] contains the submatrix of the occupation matrix
    corresponding to the projectors of atom iatom and atom jatom, with dimensions determined by the number of projectors for each atom.
    The atoms and orbitals are defined by the manifold tuple.

    Input:
      -> manifold         : Tuple{Symbol, String} with the atomic orbital type to define the Hubbard manifold
      -> basis            : PlaneWaveBasis containing the wavefunctions and k-points
      -> ψ                : wavefunctions from the scf calculation
      -> occupation       : Occupation matrix for the bands
      -> positions (opt)  : positions of the atoms in the unit cell
    Output:
      ->  n_IJ            : 3-tensor of matrices. 
                            Outer indices select spin, iatom and jatom, inner indices select m1 and m2 in the manifold.
      ->  manifold_labels : Labels for all manifold orbitals, corresponding to different columns of p_I.
      ->  p_I             : Projectors for the manifold. 
                            Those are orthonormalized against all orbitals, also outside of the manifold.
"""
function compute_hubbard_nIJ(manifold::Tuple{Symbol, String},
                                basis::PlaneWaveBasis{T},
                                ψ, occupation;
                                positions = basis.model.positions) where {T}
    filled_occ = filled_occupation(basis.model)
    proj = build_projectors(basis; positions)
    projs = proj.projectors
    labs = proj.labels
    labels, projectors = build_manifold(basis, projs, labs, manifold)
    nprojs = length(labels)
    nspins = basis.model.n_spin_components
    n_matrix = zeros(Complex{T}, nspins, nprojs, nprojs) 

    for σ in 1:nspins, ik = krange_spin(basis, σ)  
        # We divide by filled_occ to deal with the physical two spin channels separately.
        ψk, projk, nk = @views ψ[ik], projectors[ik], occupation[ik]/filled_occ  
        c = projk' * ψk      # <ϕ|ψ>
        # The matrix product is done over the bands. In QE, basis.kweights[ik]*nk[ibnd] would be wg(ik,ibnd)
        # TODO: Do I have to worry about the computational efficiency? In Fortran I would split this product
        n_matrix[σ, :, :] .+= basis.kweights[ik] * c * diagm(nk) * c' 
    end
    n_matrix = mpi_sum(n_matrix, basis.comm_kpts)

    # Now I want to reshape it to match the notation used in the papers.
    types = findall(at -> at.species == Symbol(manifold[1]), basis.model.atoms)
    natoms = length(types)  # Number of atoms of the species in the manifold
    n_IJ = Array{Matrix{Complex{T}}}(undef, nspins, natoms, natoms)
    p_I = [Vector{Matrix{Complex{T}}}(undef, natoms) for i in 1:length(basis.kpoints)]
    # Very low-level, but works
    for σ in 1:nspins
        i = 1
        while i <= nprojs
            il = labels[i].l
            iatom = labels[i].iatom
            j = 1
            while j <= nprojs
                jl = labels[j].l
                jatom = labels[j].iatom
                n_IJ[σ, iatom, jatom] = copy(n_matrix[σ, i:i+2*il, j:j+2*jl])
                j += 2*jl + 1
            end
            for (ik, projk) in enumerate(projectors)
                p_I[ik][iatom] = Matrix{Complex{T}}(undef, size(projk,1), 2*il + 1)  
                p_I[ik][iatom] = copy(projk[:, i:i+2*il])  # Store the projector for this atom
            end
            i += 2*il + 1
        end
    end

    # Still to be fully implemented. 
    # TODO: should we use basis.symmetries or basis.model.symmetries?
    l = labels[findfirst(orb -> orb.label == manifold[2], labels)].l  
    n_IJ = symmetrize(n_IJ, basis.model.lattice, basis.symmetries, l, basis.model.positions)

    return (; n_IJ=n_IJ, manifold_labels=labels, p_I=p_I)
end

# TODO: U should become a vector, with one value for each atom.
struct Hubbard
    manifold::Tuple{Any, String}
    U::Float64
end
(hubbard::Hubbard)(::AbstractBasis) = TermHubbard(hubbard.manifold, hubbard.U)

struct TermHubbard <: Term
    manifold::Tuple{Any, String}
    U::Float64
end

@timing "ene_ops: hubbard" function ene_ops(term::TermHubbard, 
                                            basis::PlaneWaveBasis{T}, 
                                            ψ, occupation; 
                                            n=nothing, qe=false, kwargs...) where {T}
    if ψ === nothing
        return (; E=zero(T), ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
    end

    filled_occ = filled_occupation(basis.model)
    totatoms = length(basis.model.atoms)
    types = findall(at -> at.species == Symbol(term.manifold[1]), basis.model.atoms)
    natoms = length(types)  # Number of atoms of the species in the manifold
    nspins = basis.model.n_spin_components
    if isnothing(n)
        Hubbard = compute_hubbard_nIJ(term.manifold, basis, ψ, occupation)
        n = Hubbard.n_IJ
        proj = Hubbard.p_I
    else
        if qe   # This part is for debugging using the qe matrix as input, converting it to DFTK format
            n_IJ = Array{Matrix{Complex{T}}}(undef, nspins, natoms, natoms)  
            for σ in 1:nspins, iatom in 1:natoms, jatom in 1:natoms
                if jatom == iatom
                    n_IJ[σ, iatom, jatom] = n  
                else
                    n_IJ[σ, iatom, jatom] = zeros(typeof(n[1,1][1,1]), size(n,1), size(n,2))  # Off-diagonal terms are irrelevant
                end
            end
            n = n_IJ 
        end
        Hubbard = compute_hubbard_nIJ(term.manifold, basis, ψ, occupation)
        proj = Hubbard.p_I
    end

    # To compare the results with Quantum ESPRESSO, we need to convert the U value from eV.
    #   In QE the U value is given in eV in the input but DFTK works in Hartrees.
    to_unit = ustrip(auconvert(u"eV", 1.0))  # Ha to eV conversion factor
    U = term.U / to_unit  # Convert U to Ha
    E = zero(T)
    for σ in 1:nspins, iatom in 1:natoms
        E += filled_occ * 0.5 * U * real(tr(n[σ, iatom,iatom] * (I - n[σ, iatom,iatom])))
    end

    ops = [HubbardUOperator(basis, kpt, U, n, proj[ik]) for (ik,kpt) in enumerate(basis.kpoints)]
    (; E, ops, n)
end

# TODO: Once this is done, adding the V term as well should be trivial.
#       But remember to change the symmetrization as well!