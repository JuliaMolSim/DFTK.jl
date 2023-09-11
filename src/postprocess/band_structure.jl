import Brillouin
import Brillouin.KPaths: KPath, KPathInterpolant, irrfbz_path

@doc raw"""
Extract the high-symmetry ``k``-point path corresponding to the passed `model`
using `Brillouin`. Uses the conventions described in the reference work by
Cracknell, Davies, Miller, and Love (CDML). Of note, this has minor differences to
the ``k``-path reference
([Y. Himuma et. al. Comput. Mater. Sci. **128**, 140 (2017)](https://doi.org/10.1016/j.commatsci.2016.10.015))
underlying the path-choices of `Brillouin.jl`, specifically for oA and mC Bravais types.

If the cell is a supercell of a smaller primitive cell, the standard ``k``-path of the
associated primitive cell is returned. So, the high-symmetry ``k`` points are those of the
primitive cell Brillouin zone, not those of the supercell Brillouin zone.

The `dim` argument allows to artificially truncate the dimension of the employed model,
e.g. allowing to plot a 2D bandstructure of a 3D model (useful for example for plotting
band structures of sheets with `dim=2`).

Due to lacking support in `Spglib.jl` for two-dimensional lattices it is (a) assumed that
`model.lattice` is a *conventional* lattice and (b) required to pass the space group
number using the `sgnum` keyword argument.
"""
function irrfbz_path(model; dim::Integer=model.n_dim, sgnum=nothing, magnetic_moments=[])
    @assert dim ≤ model.n_dim
    for i in dim:3, j in dim:3
        if i != j && !iszero(model.lattice[i, j])
            error("Reducing the dimension for band structure plotting only allowed " *
                  "if the dropped dimensions are orthogonal to the remaining ones.")
        end
    end
    if !isnothing(sgnum) && dim ∈ (1, 3)
        @warn("sgnum keyword argument unused in `irrfbz_path` unused " *
              "unless a 2-dimensional lattice is encountered.")
    end

    # Brillouin.jl expects the input direct lattice to be in the conventional lattice
    # in the convention of the International Table of Crystallography Vol A (ITA).
    #
    # The output of Brillouin.jl are k-Points and reciprocal lattice vectors
    # in the CDML convention.
    if dim == 1
        # Only one space group; avoid spglib here
        kpath = Brillouin.irrfbz_path(1, [[model.lattice[1, 1]]], Val(1))
    elseif dim == 2
        if isnothing(sgnum)
            error("sgnum keyword argument (specifying the ITA space group number) " *
                  "is required for band structure plots in 2D lattices.")
        end
        # TODO We assume to have the conventional lattice here.
        lattice_2d = [model.lattice[1:2, 1], model.lattice[1:2, 2]]
        kpath = Brillouin.irrfbz_path(sgnum, lattice_2d, Val(2))
    elseif dim == 3
        # Brillouin.jl has an interface to Spglib.jl to directly reduce the passed
        # lattice to the ITA conventional lattice and so the Spglib cell can be
        # directly used as an input.
        cell, _ = spglib_cell(model, magnetic_moments)
        kpath = Brillouin.irrfbz_path(cell)
    end

    # TODO In case of absence of time-reversal symmetry we need to explicitly
    #      add the inverted kpath here!
    #      See https://github.com/JuliaMolSim/DFTK.jl/pull/496/files#r725203554

    kpath
end

"""Return kpoint coordinates in reduced coordinates"""
function kpath_get_kcoords(kpath::KPathInterpolant{D}) where {D}
    map(k -> vcat(k, zeros_like(k, 3 - D)), kpath)
end
function kpath_get_branch(kpath::KPathInterpolant{D}, ibranch::Integer) where {D}
    map(k -> vcat(k, zeros_like(k, 3 - D)), kpath.kpaths[ibranch])
end

@timing function compute_bands(basis::PlaneWaveBasis, kcoords::AbstractVector;
                               n_bands=default_n_bands_bandstructure(basis.model),
                               ρ=nothing, eigensolver=lobpcg_hyper,
                               tol=1e-3, show_progress=true, kwargs...)
    # kcoords are the kpoint coordinates in fractional coordinates
    if isnothing(ρ)
        if any(t isa TermNonlinear for t in basis.terms)
            error("If a non-linear term is present in the model the converged density is required " *
                  "to compute bands. Either pass the self-consistent density as the ρ keyword " *
                  "argument or use the plot_bandstructure(scfres) function.")
        end
        ρ = guess_density(basis)
    end

    # Create basis with new kpoints, without any symmetry operations.
    kweights = ones(length(kcoords)) ./ length(kcoords)
    bs_basis = PlaneWaveBasis(basis, kcoords, kweights)

    ham = Hamiltonian(bs_basis; ρ)
    eigres = diagonalize_all_kblocks(eigensolver, ham, n_bands + 3;
                                     n_conv_check=n_bands, tol, show_progress, kwargs...)
    if !eigres.converged
        @warn "Eigensolver not converged" n_iter=eigres.n_iter
    end
    merge((; basis=bs_basis), select_eigenpairs_all_kblocks(eigres, 1:n_bands))
end
function compute_bands(basis::PlaneWaveBasis, kpath::KPathInterpolant; kwargs...)
    compute_bands(basis, kpath_get_kcoords(kpath); kwargs...)
end


function kdistances_and_ticks(kcoords, klabels::Dict, kbranches)
    # kcoords in cartesian coordinates, klabels uses cartesian coordinates
    function getlabel(kcoord; tol=1e-4)
        findfirst(c -> norm(c - kcoord) < tol, klabels)
    end

    kdistances = eltype(kcoords[1])[]
    tick_distances = eltype(kcoords[1])[]
    tick_labels = String[]
    for (ibranch, kbranch) in enumerate(kbranches)
        kdistances_branch = cumsum(append!([0.], [norm(kcoords[ik - 1] - kcoords[ik])
                                                  for ik in kbranch[2:end]]))
        if ibranch == 1
            append!(kdistances, kdistances_branch)
        else
            append!(kdistances, kdistances_branch .+ kdistances[end][end])
        end
        for ik in kbranch
            kcoord = kcoords[ik]
            if getlabel(kcoord) !== nothing
                if ibranch != 1 && ik == kbranch[1]
                    # New branch encountered. Do not add a new tick point but update label.
                    tick_labels[end] *= " | " * String(getlabel(kcoord))
                else
                    push!(tick_labels, String(getlabel(kcoord)))
                    push!(tick_distances, kdistances[ik])
                end
            end
        end
    end
    ticks = (distances=tick_distances, labels=tick_labels)
    (; kdistances, ticks)
end


function data_for_plotting(kpath::KPathInterpolant, band_data; datakeys=[:λ, :λerror])
    basis    = band_data.basis
    n_spin   = basis.model.n_spin_components
    n_kcoord = length(basis.kpoints) ÷ n_spin
    n_bands  = nothing

    # XXX Convert KPathInterpolant => kbranches, klabels
    kbranches = [1:length(kpath.kpaths[1])]
    for n in length.(kpath.kpaths)[2:end]
        push!(kbranches, kbranches[end].stop+1:kbranches[end].stop+n)
    end
    klabels = Dict{Symbol, Vec3{eltype(kpath[1])}}()
    for (ibranch, labels) in enumerate(kpath.labels)
        for (k, v) in pairs(labels)
            # Convert to Cartesian and add to labels
            klabels[v] = basis.model.recip_lattice * kpath_get_branch(kpath, ibranch)[k]
        end
    end

    # Convert coordinates to Cartesian
    kcoords_cart = [basis.model.recip_lattice * basis.kpoints[ik].coordinate
                    for ik in krange_spin(basis, 1)]

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

    kdistances, ticks = kdistances_and_ticks(kcoords_cart, klabels, kbranches)
    (; ticks, kdistances, kbranches, n_bands, n_kcoord, n_spin, data...)
end


"""
    is_metal(eigenvalues, εF; tol)

Determine whether the provided bands indicate the material is a metal,
i.e. where bands are cut by the Fermi level.
"""
function is_metal(eigenvalues, εF; tol=1e-4)
    n_bands = length(eigenvalues[1])
    for ib in 1:n_bands
        some_larger  = any(εk[ib] > εF + tol for εk in eigenvalues)
        some_smaller = any(εk[ib] < εF - tol for εk in eigenvalues)
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
function plot_bandstructure(basis::PlaneWaveBasis, kpath::KPath=irrfbz_path(basis.model);
                            εF=nothing, kline_density=40u"bohr",
                            unit=u"hartree", kwargs_plot=(; ),
                            kwargs...)
    mpi_nprocs() > 1 && error("Band structures with MPI not supported yet")
    if !isdefined(DFTK, :PLOTS_LOADED)
        error("Plots not loaded. Run 'using Plots' before calling plot_bandstructure.")
    end

    # Band structure calculation along high-symmetry path
    kinter = Brillouin.interpolate(kpath, density=austrip(kline_density))
    println("Computing bands along kpath:")
    sortlabels = map(bl -> last.(sort(collect(pairs(bl)))), kinter.labels)
    println("       ", join(join.(sortlabels, " -> "), "  and  "))
    band_data = compute_bands(basis, kinter; kwargs...)
    plot_band_data(kinter, band_data; εF, unit, kwargs_plot...)
end
function plot_bandstructure(scfres::NamedTuple, kpath::KPath=irrfbz_path(scfres.basis.model);
                            n_bands=default_n_bands_bandstructure(scfres), kwargs...)
    plot_bandstructure(scfres.basis, kpath; n_bands, scfres.ρ, scfres.εF, kwargs...)
end
