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
    compute_pdos(εs, basis, ψ, eigenvalues; [positions, smearing, temperature])

Compute the projected density of states (PDOS) for all atoms and orbitals.

Overview of inputs: 
- `εs`: Vector of energies at which the PDOS will be computed

Overview of outputs:
- `pdos`: 3D array of PDOS, pdos[iε_idx, iproj, σ] = PDOS at energy εs[iε_idx] for projector iproj and spin σ
- `projector_labels` : Vector of tuples (iatom, species, label, n, l, m) for each projector, 
     that maps the iproj index to the corresponding atomic orbital. 
     For details see the documentation for `atomic_orbital_projectors`.

Notes: 
- The atomic orbital projectors are taken from the pseudopotential files used to build the atoms. 
    In doubt, consult the pseudopotential file for the list of available atomic orbitals.
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
                                        / temperature
                                        * Smearing.occupation_derivative(smearing, enred))
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
    atomic_orbital_projectors(basis; [isonmanifold, positions])   

Build the matrices of projectors onto the pseudoatomic orbitals.
         
    projector[ik][iG, iproj] = 1/√Ω FT{ ϕperₙₗₘ(. - Rᵢ) }(k+G) + orthogonalization

 where Ω is the unit cell volume, ϕperₙₗₘ(. - Rᵢ) is the periodized pseudoatomic orbital (n, l, m) centered at Rᵢ
  and iproj is the index corresponding to atom i and the quantum numbers (n, l, m). 
  This correspondance is recorded in `labels`.

The projectors are computed by decomposition into a form factor multiplied by a structure factor:
  FT{ ϕperₙₗₘ(. - Rᵢ) }(k+G) = Fourier transform of periodized atomic orbital ϕₙₗₘ (form factor)
                           * structure factor for atom center exp(-i<Rᵢ,k+G>)
  
Overview of inputs: 
- `positions` : Positions of the atoms in the unit cell. Default is model.positions.
- `isonmanifold` (opt) : (see notes below) OrbitalManifold struct to select only a subset of orbitals 
     for the computation.

Overview of outputs:
- `projectors`: Vector of matrices of projectors
- `labels`: Vector of NamedTuples containing iatom, species, n, l, m and orbital name for each projector

Notes: 
- The orbitals used for the projectors are all orthogonalized against each other. 
    This corresponds to ortho-atomic projectors in Quantum Espresso.
- 'n' in labels is not exactly the principal quantum number, but rather the index of the radial function 
    in the pseudopotential. As an example, if the only S orbitals in the pseudopotential are 3S and 4S, 
    then those are indexed as n=1, l=0 and n=2, l=0 respectively.
- Use 'isonmanifold' kwarg with caution, since the resulting projectors would be 
    orthonormalized only against the manifold basis. 
    Most applications require the whole projectors basis to be orthonormal instead.
"""
function atomic_orbital_projectors(basis::PlaneWaveBasis{T};
                                   isonmanifold = l -> true,
                                   positions = basis.model.positions) where {T}
    
    G_plus_k_all = [Gplusk_vectors(basis, basis.kpoints[ik])
                    for ik = 1:length(basis.kpoints)]
    G_plus_k_all_cart = [map(recip_vector_red_to_cart(basis.model), gpk) 
                         for gpk in G_plus_k_all]

    projectors = [Matrix{Complex{T}}(undef, length(G_plus_k), 0) for G_plus_k in G_plus_k_all]
    labels = []
    for (iatom, atom) in enumerate(basis.model.atoms)
        psp = atom.psp
        @assert count_n_pswfc(psp) > 0 # We need this to check if we have any atomic orbital projector
        for l in 0:psp.lmax, n in 1:DFTK.count_n_pswfc_radial(psp, l)
            label = DFTK.pswfc_label(psp, n, l)
            if !isonmanifold((; iatom, atom.species, label))
                continue
            end
            fun(p) = eval_psp_pswfc_fourier(psp, n, l, p)
            form_factors_l = build_form_factors(fun, l, G_plus_k_all_cart)
            for ik in 1:length(G_plus_k_all_cart)
               structure_factor = [cis2pi(-dot(positions[iatom], p)) for p in G_plus_k_all[ik]]
               projectors[ik] = hcat(projectors[ik], 
                                     form_factors_l[ik] .* structure_factor 
                                      ./ sqrt(basis.model.unit_cell_volume))
            end
            for m in -l:l
                push!(labels, (; iatom, atom.species, n, l, m, label))
            end
        end
    end

    projectors = ortho_lowdin.(projectors)

    return (; projectors, labels)
end

"""
    atomic_orbital_projections(basis, ψ; [isonmanifold, positions])

Build the projection matrices of ψ onto each pseudo-atomic orbital.

    projection[ik][iband, iproj] = ‖ ψ'[ik][:, iband] * projector[ik][:, iproj] ‖²

For more details, see documentation for [`atomic_orbital_projectors`](@ref).
"""
function atomic_orbital_projections(basis::PlaneWaveBasis{T}, ψ;
                                    isonmanifold = l -> true,
                                    positions = basis.model.positions) where {T}
    projectors, labels = atomic_orbital_projectors(basis; isonmanifold, positions)
    projections = map(ψ, projectors) do ψk, projectorsk
        abs2.(ψk' * projectorsk)
    end

    return (; projections, labels)
end

"""
    sum_pdos(pdos_res, manifolds)

This function extracts and sums up all the PDOSes, directly from the output of the `compute_pdos` function, 
  that match any of the manifolds.

Overview of inputs:
- `pdos_res`: Whole output from compute_pdos.
- `manifolds`: Vector of OrbitalManifolds to select the desired projectors pdos.

Overview of outputs:
- `pdos`: Vector containing the pdos(ε).
"""
function sum_pdos(pdos_res, manifolds::AbstractVector)
    pdos = zeros(Float64, length(pdos_res.εs), size(pdos_res.pdos, 3))
    for σ in 1:size(pdos_res.pdos, 3)
       for (j, orb) in enumerate(pdos_res.projector_labels)
            if any(is_on_manifold(orb, manifold) for manifold in manifolds)
                pdos[:, σ] += pdos_res.pdos[:, j, σ]
            end
        end
    end
    return pdos
end

"""
Plot the density of states over a reasonable range. Requires to load `Plots.jl` beforehand.
"""
function plot_dos end

function plot_ldos end

function plot_pdos end
