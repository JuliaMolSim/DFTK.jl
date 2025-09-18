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
     -> εs               : Vector of energies at which the PDOS will be computed
     -> basis            : PlaneWaveBasis object from bands computation
     -> ψ                : Wavefunction from the bands
     -> eigenvalues      : Eigenvalues from the bands
    Output:
     -> pdos             : 3D array of PDOS, pdos[iε_idx, iproj, σ] = PDOS at energy εs[iε_idx] for projector iproj and spin σ
     -> projector_labels : Vector of tuples (iatom, n, l, m) for each projector, that maps the iproj index to the corresponding atomic orbital (atom index, principal quantum number, angular momentum, magnetic quantum number)
Notes: 
 - All the information about projectors is taken from the PseudoPotential files used to build the atoms. 
    There may be cases where the relevant data are missing for the desired projectors, despite being apparently declared.
    As an example, it may happen that the PseudoPotential file does not have all atomic projectors up to the l_max declared in the psp, 
    since such l_max refers instead to the beta projectors.
"""
function compute_pdos(εs, basis::PlaneWaveBasis{T}, ψ, eigenvalues; 
                      positions=basis.model.positions,
                      smearing=basis.model.smearing, 
                      temperature=basis.model.temperature) where {T}
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_pdos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)
    
    projections, projector_labels = atomic_orbital_projections(basis, ψ; positions)

    nprojs = length(projector_labels) 

    D = zeros(typeof(εs[1]), length(εs), nprojs, basis.model.n_spin_components)  
    for (iε, ε) in enumerate(εs)
        for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
            projsk = projections[ik]  
            @views for (iband, εnk) in enumerate(eigenvalues[ik])
                enred = (εnk - ε) / temperature
                for iproj in 1:size(projsk, 2)
                    D[iε, iproj, σ] -= (filled_occ * basis.kweights[ik] * projsk[iband, iproj]
                                        ./ temperature
                                        .* Smearing.occupation_derivative(smearing, enred))
                end
            end
        end
    end
    pdos = mpi_sum(D, basis.comm_kpts)  

    return (; pdos, projector_labels, εs)
end

function compute_pdos(εs, bands; kwargs...)
    compute_pdos(εs, bands.basis, bands.ψ, bands.eigenvalues; kwargs...)
end

"""
Structure for manifold choice and projectors extraction. Fields:
    -> iatom   : Atom position in the atoms array.
    -> species : Chemical Element as in ElementPsp.
    -> label   : Orbital name, e.g.: "3S".
All fields are optional, only the given ones will be used for selection.
Implemented function for OrbitalManifold can be applied to an orbital NamedTuple and returns a boolean
    stating whether the orbital belongs to the manifold.
"""
@kwdef struct OrbitalManifold
    iatom   ::Union{Int64,  Nothing} = nothing
    species ::Union{Symbol, Nothing} = nothing
    label   ::Union{String, Nothing} = nothing
end
function (s::OrbitalManifold)(orb)
    iatom_match    = isnothing(s.iatom)   || (s.iatom == orb.iatom)
    species_match  = isnothing(s.species) || (s.species == orb.species)
    label_match    = isnothing(s.label)   || (s.label == orb.label)

    iatom_match && species_match && label_match
end
"""
Build the projectors matrices projsk for all k-points at the same time.
         
    projector[ik][:, iproj] = |ϕinlm>(kpt)

 where ϕinlm is the atomic orbital for atom i, quantum numbers (n,l,m)
   and iproj is the corresponding column index. The mapping is recorded in 'labels'.
  
    Input: 
     - basis            : PlaneWaveBasis
     - ismanifold (opt) : (see notes below) OrbitalManifold struct to select only a subset of orbitals for the computation.
     - positions        : Positions of the atoms in the unit cell. Default is model.positions.
    Output:
     - projectors       : Vector of matrices of projectors
     - labels           : Vector of NamedTuples containing iatom, species, n, l, m and orbital name for each projector

Notes: 

- 'n' in labels is not exactly the principal quantum number, but rather the index of the radial function in the pseudopotential. As an example, if the only S orbitals in the pseudopotential are 3S and 4S, then those are indexed as n=1, l=0 and n=2, l=0 respectively.
- Use 'ismanifold' kwarg with caution, since the resulting projectors would be orthonormalized only against the manifold basis. Most applications require the whole projectors basis to be orthonormal instead.
"""
function atomic_orbital_projectors(basis::PlaneWaveBasis{T};
                                   ismanifold = l -> true,  #Should we allow to take and orthogonalize only the manifold?
                                   positions = basis.model.positions) where {T}
    
    G_plus_k_all = [Gplusk_vectors(basis, basis.kpoints[ik])
                    for ik = 1:length(basis.kpoints)]
    G_plus_k_all_cart = [map(recip_vector_red_to_cart(basis.model), gpk) 
                         for gpk in G_plus_k_all]

    projectors = [Matrix{Complex{T}}(undef, length(G_plus_k), 0) for G_plus_k in G_plus_k_all]
    labels = []
    for (iatom, atom) in enumerate(basis.model.atoms)
        psp = atom.psp
        for l in 0:psp.lmax
            for n in 1:DFTK.count_n_pswfc_radial(psp, l)
                label = DFTK.get_pswfc_label(psp, n, l)
                if !ismanifold((;iatom, species=Symbol(atom.species), label))
                    continue
                end
                fun(p) = eval_psp_pswfc_fourier(psp, n, l, p)
                form_factors_l = build_form_factors(fun, l, G_plus_k_all_cart)
                for ik in 1:length(G_plus_k_all_cart)
                   structure_factor = [cis2pi(-dot(positions[iatom], p)) for p in G_plus_k_all[ik]]
                   projectors[ik] = hcat(projectors[ik], form_factors_l[ik] .* structure_factor ./ sqrt(basis.model.unit_cell_volume))
                end
                for m in -l:l
                    push!(labels, (; iatom, atom.species, n, l, m, label))
                end
            end
        end
    end

    projectors = ortho_lowdin.(projectors)

    return (; projectors, labels)
end

"""
Build the projection matrices projsk for all k-points at the same time.

    projection[ik][iband, iproj] = <ψnk|*projector[ik][iband, iproj] =  <ψnk|ϕinlm>(kpt)

 where ψnk is the atomic wavefunction component for band iband and kpoint kpt.

For more details, see documentation for 'atomic_orbital_projectors'.
"""
function atomic_orbital_projections(basis::PlaneWaveBasis{T}, ψ;
                                    ismanifold = l -> true,
                                    positions = basis.model.positions) where {T}
    projectors, labels = atomic_orbital_projectors(basis; ismanifold, positions)
    projections = map(zip(ψ, projectors)) do (ψk, projectorsk)
        abs2.(ψk' * projectorsk)
    end

    return (; projections, labels)
end

"""
This function extracts the required pdos from the output of the `compute_pdos` function. 

    Input:
     -> pdos_res  : Whole output from compute_pdos.
     -> manifolds : Vector of OrbitalManifolds to select the desired projectors pdos.
    Output:
     -> pdos      : Vector containing the pdos(ε).
"""
function sum_pdos(pdos_res, manifolds::AbstractVector)
    pdos = []
    for σ in 1:size(pdos_res.pdos, 3)
        pdos_values = zeros(Float64, length(pdos_res.εs))
        for ismanifold in manifolds
            for (j, orb) in enumerate(pdos_res.projector_labels)
                if ismanifold(orb)
                    pdos_values += pdos_res.pdos[:, j, σ]
                end
            end
        end
        push!(pdos, pdos_values)
    end
    return pdos
end

"""
Plot the density of states over a reasonable range. Requires to load `Plots.jl` beforehand.
"""
function plot_dos end

function plot_ldos end

function plot_pdos end