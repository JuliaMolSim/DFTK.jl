import Brillouin
import Brillouin.KPaths: Bravais

@doc raw"""
Extract the high-symmetry ``k``-point path corresponding to the passed model
using `Brillouin.jl`. Uses the conventions described in the reference work by
Cracknell, Davies, Miller, and Love (CDML). Of note, this has minor differences to
the ``k``-path reference
([Y. Himuma et. al. Comput. Mater. Sci. **128**, 140 (2017)](https://doi.org/10.1016/j.commatsci.2016.10.015))
underlying the path-choices of `Brillouin.jl`, specifically for oA and mC Bravais types.
The `kline_density` is given in number of ``k``-points per inverse bohrs (i.e.
overall in units of length).

Issues a warning in case the passed lattice does not match the expected primitive.
"""
function high_symmetry_kpath(model; kline_density=40)
    kline_density = austrip(kline_density)

    if model.n_dim == 1  # Return fast for 1D model
        # TODO Is this special-casing of 1D is not needed for Brillouin.jl any more
        #
        # Just use irrfbz_path(1, DirectBasis{1}([1.0]))
        # (see https://github.com/JuliaMolSim/DFTK.jl/pull/496/files#r725205860)
        #
        # Length of the kpath is recip_lattice[1, 1] in 1D
        n_points = max(2, 1 + ceil(Int, kline_density * model.recip_lattice[1, 1]))
        kcoords  = [@SVector[coord, 0, 0] for coord in range(-1//2, 1//2, length=n_points)]
        klabels  = Dict("Γ" => zeros(3), "-½" => [-0.5, 0.0, 0.0], "½" => [0.5, 0, 0])
        return (kcoords=kcoords, klabels=klabels,
                kpath=[["-½", "½"]], kbranches=[1:length(kcoords)])
    end

    # - Brillouin.jl expects the input direct lattice to be in the conventional lattice
    #   in the convention of the International Table of Crystallography Vol A (ITA).
    # - spglib uses this convention for the returned conventional lattice,
    #   so it can be directly used as input to Brillouin.jl
    # - The output k-Points and reciprocal lattices will be in the CDML convention.
    conv_latt = spglib_standardize_cell(model; primitive=false,correct_symmetry=false).lattice
    sgnum     = spglib_spacegroup_number(model)  # Get ITA space-group number
    direct_basis   = Bravais.DirectBasis(collect(eachcol(conv_latt)))
    primitive_latt = Bravais.primitivize(direct_basis, Bravais.centering(sgnum, 3))

    primitive_latt ≈ collect(eachcol(model.lattice)) || @warn(
        "DFTK's model.lattice and Brillouin's primitive lattice do not agree. " *
        "The kpath selected to plot the band structure might not be most appropriate."
    )

    kp     = Brillouin.irrfbz_path(sgnum, direct_basis)
    kinter = Brillouin.interpolate(kp, density=kline_density)

    # TODO Need to take care of time-reversal symmetry here!
    #      See https://github.com/JuliaMolSim/DFTK.jl/pull/496/files#r725203554

    # Need to double the points whenever a new path starts
    # (for temporary compatibility with pymatgen)
    # TODO Remove this later
    kcoords = empty(first(kinter.kpaths))
    for kbranch in kinter.kpaths
        idcs = findall(k -> any(sum(abs2, k - kcomp) < 1e-5
                                for kcomp in values(kp.points)), kbranch)
        @assert length(idcs) ≥ 2
        idcs = idcs[2:end-1]  # Don't duplicate first and last
        idcs = sort(append!(idcs, 1:length(kbranch)))
        append!(kcoords, kbranch[idcs])
    end

    T = eltype(kcoords[1])
    klabels = Dict{String,Vector{T}}(string(key) => val for (key, val) in kp.points)
    kpath   = [[string(el) for el in path] for path in kp.paths]
    (; kcoords, klabels, kpath)
end


@timing function compute_bands(basis, kcoords;
                               n_bands=default_n_bands_bandstructure(basis.model),
                               ρ=nothing, eigensolver=lobpcg_hyper,
                               tol=1e-3, show_progress=true, kwargs...)
    # Create basis with new kpoints, where we cheat by not using any symmetry operations.
    ksymops  = [[identity_symop()] for _ in 1:length(kcoords)]
    bs_basis = PlaneWaveBasis(basis, kcoords, ksymops)

    if isnothing(ρ)
        if any(t isa TermNonlinear for t in basis.terms)
            error("If a non-linear term is present in the model the converged density is required " *
                  "to compute bands. Either pass the self-consistent density as the ρ keyword " *
                  "argument or use the plot_bandstructure(scfres) function.")
        end
        ρ = guess_density(basis)
    end

    ham = Hamiltonian(bs_basis; ρ)
    band_data = diagonalize_all_kblocks(eigensolver, ham, n_bands + 3;
                                        n_conv_check=n_bands,
                                        tol=tol, show_progress=show_progress, kwargs...)
    if !band_data.converged
        @warn "Eigensolver not converged" iterations=band_data.iterations
    end
    merge((basis=bs_basis, ), select_eigenpairs_all_kblocks(band_data, 1:n_bands))
end


function split_into_branches(kcoords, data::Dict, klabels::Dict)
    # kcoords in cartesian coordinates, klabels uses cartesian coordinates
    function getlabel(kcoord; tol=1e-4)
        findfirst(c -> norm(c - kcoord) < tol, klabels)
    end

    branches = Any[(kindices = [0], kdistances=[0.0], ), ]
    for (ik, kcoord) in enumerate(kcoords)
        previous_kcoord = ik == 1 ? kcoords[1] : kcoords[ik - 1]
        if !isnothing(getlabel(kcoord)) && !isnothing(getlabel(previous_kcoord))
            # New branch encountered
            previous_distance = branches[end].kdistances[end]
            push!(branches, (kindices=[ik], kdistances=[previous_distance]))
        else
            # Keep adding to current branch
            distance = branches[end].kdistances[end] + norm(kcoord - kcoords[ik - 1])
            push!(branches[end].kdistances, distance)
            push!(branches[end].kindices, ik)
        end
    end

    map(branches[2:end]) do branch
        branch_data = Dict(key => data[key][branch.kindices, :, :] for key in keys(data))
        ret = (klabels=(getlabel(kcoords[branch.kindices[1]]),
                        getlabel(kcoords[branch.kindices[end]])),
               kdistances=branch.kdistances,
               kindices=branch.kindices)
        merge(ret, (; branch_data...))
    end
end


function prepare_band_data(band_data; datakeys=[:λ, :λerror],
                           klabels=Dict{String, Vector{Float64}}())
    basis = band_data.basis
    n_spin   = basis.model.n_spin_components
    n_kcoord = length(basis.kpoints) ÷ n_spin
    n_bands  = nothing

    # Convert coordinates to Cartesian
    kcoords_cart = [basis.kpoints[ik].coordinate_cart for ik in krange_spin(basis, 1)]
    klabels_cart = Dict(lal => basis.model.recip_lattice * vec for (lal, vec) in klabels)

    # Split data into branches
    data = Dict{Symbol, Any}()
    for key in datakeys
        hasproperty(band_data, key) || continue
        n_bands = length(band_data[key][1])
        data_per_kσ = similar(band_data[key][1], n_kcoord, n_bands, n_spin)
        for σ in 1:n_spin, (ito, ik) in enumerate(krange_spin(basis, σ))
            data_per_kσ[ito, :, σ] = band_data[key][ik]
        end
        data[key] = data_per_kσ
    end
    @assert !isnothing(n_bands)
    branches = split_into_branches(kcoords_cart, data, klabels_cart)

    tick_labels    = String[branches[1].klabels[1]]
    tick_distances = Float64[branches[1].kdistances[1]]
    for (i, br) in enumerate(branches)
        # Ignore branches with a single k-point
        branches[i].klabels[1] == branches[i].klabels[2] && continue

        label    = branches[i].klabels[2]
        distance = branches[i].kdistances[end]
        if i != length(branches) && branches[i+1].klabels[1] != label
            # Next branch is not continuous from the current
            label = label * " | " * branches[i+1].klabels[1]
        end
        push!(tick_labels, label)
        push!(tick_distances, distance)
    end

    (branches=branches, ticks=(distances=tick_distances, labels=tick_labels),
     n_bands=n_bands, n_kcoord=n_kcoord, n_spin=n_spin, basis=basis)
end


"""
    is_metal(band_data, εF, tol)

Determine whether the provided bands indicate the material is a metal,
i.e. where bands are cut by the Fermi level.
"""
function is_metal(band_data, εF, tol=1e-4)
    n_bands = length(band_data.λ[1])
    n_kpoints = length(band_data.λ)
    for ib in 1:n_bands
        some_larger = any(band_data.λ[ik][ib] - εF < -tol for ik in 1:n_kpoints)
        some_smaller = any(εF - band_data.λ[ik][ib] < -tol for ik in 1:n_kpoints)
        some_larger && some_smaller && return true
    end
    false
end

# Number of bands to compute when plotting the bandstructure
default_n_bands_bandstructure(n_bands_scf::Int) = ceil(Int, n_bands_scf + 5sqrt(n_bands_scf))
function default_n_bands_bandstructure(model::Model)
    default_n_bands_bandstructure(default_n_bands(model))
end
function default_n_bands_bandstructure(scfres::NamedTuple)
    n_bands_scf = length(scfres.occupation[1])
    default_n_bands_bandstructure(n_bands_scf)
end


"""
Compute and plot the band structure. `n_bands` selects the number of bands to compute.
If this value is absent and an `scfres` is used to start the calculation a default of
`n_bands_scf + 5sqrt(n_bands_scf)` is used. The unit used to plot the bands can
be selected using the `unit` parameter. Like in the rest of DFTK Hartree is used
by default. Another standard choices is `unit=u"eV"` (electron volts).
The `kline_density` is given in number of ``k``-points per inverse bohrs (i.e.
overall in units of length).
"""
function plot_bandstructure(basis::PlaneWaveBasis;
                            εF=nothing, kline_density=40u"bohr",
                            unit=u"hartree", kwargs_plot=(), kwargs...)
    mpi_nprocs() > 1 && error("Band structures with MPI not supported yet")
    if !isdefined(DFTK, :PLOTS_LOADED)
        error("Plots not loaded. Run 'using Plots' before calling plot_bandstructure.")
    end

    # Band structure calculation along high-symmetry path
    kcoords, klabels, kpath = high_symmetry_kpath(basis.model; kline_density)
    println("Computing bands along kpath:")
    println("       ", join(join.(kpath, " -> "), "  and  "))
    band_data = compute_bands(basis, kcoords; kwargs...)
    plot_band_data(band_data; εF, klabels, unit, kwargs_plot...)
end
function plot_bandstructure(scfres::NamedTuple;
                            n_bands=default_n_bands_bandstructure(scfres), kwargs...)
    plot_bandstructure(scfres.basis; n_bands, ρ=scfres.ρ, εF=scfres.εF, kwargs...)
end
