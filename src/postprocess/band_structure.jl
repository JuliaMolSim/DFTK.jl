import Brillouin
import Brillouin.KPaths: KPath, KPathInterpolant, irrfbz_path

"""
Compute `n_bands` eigenvalues and Bloch waves at the k-Points specified by the `kgrid`.
All kwargs not specified below are passed to [`diagonalize_all_kblocks`](@ref):

- `kgrid`: A custom kgrid to perform the band computation, e.g. a new
  [`MonkhorstPack`](@ref) grid.
- `tol` The default tolerance for the eigensolver is substantially lower than
  for SCF computations. Increase if higher accuracy desired.
- `eigensolver`: The diagonalisation method to be employed.
"""
@timing function compute_bands(basis::PlaneWaveBasis, kgrid::AbstractKgrid;
                               n_bands=default_n_bands_bandstructure(basis.model),
                               n_extra=3, ρ=nothing, εF=nothing, eigensolver=lobpcg_hyper,
                               tol=1e-3, kwargs...)
    # kcoords are the kpoint coordinates in fractional coordinates
    if isnothing(ρ)
        if any(t isa TermNonlinear for t in basis.terms)
            error("If a non-linear term is present in the model the converged density is required " *
                  "to compute bands. Either pass the self-consistent density as the ρ keyword " *
                  "argument or use the compute_bands(scfres) function.")
        end
        ρ = guess_density(basis)
    end

    # Create new basis with new kpoints
    bs_basis = PlaneWaveBasis(basis, kgrid)

    ham = Hamiltonian(bs_basis; ρ)
    eigres = diagonalize_all_kblocks(eigensolver, ham, n_bands + n_extra;
                                     n_conv_check=n_bands, tol, kwargs...)
    if !eigres.converged
        @warn "Eigensolver not converged" n_iter=eigres.n_iter
    end
    eigres = select_eigenpairs_all_kblocks(eigres, 1:n_bands)

    occupation=nothing
    if !isnothing(εF)
        (; occupation) = compute_occupation(bs_basis, eigres.λ, εF)
    end

    # TODO This return value should be as compatible as possible with the
    #      scfres named tuple to ensure the same functions can be used
    #      for saving the results. Maybe it even makes sense to have an
    #      AbstractBandData type from which the BandData and ScfResults
    #      types subtype. In a first version the ScfResult could just contain
    #      the currently used named tuple and forward all operations to it.
    (; basis=bs_basis, ψ=eigres.X, eigenvalues=eigres.λ, ρ, εF, occupation,
     diagonalization=[eigres])
end

"""
Compute band data starting from SCF results. `εF` and `ρ` from the `scfres` are forwarded
to the band computation and `n_bands` is by default selected
as `n_bands_scf + 5sqrt(n_bands_scf)`.
"""
function compute_bands(scfres::NamedTuple, kgrid::AbstractKgrid;
                       n_bands=default_n_bands_bandstructure(scfres), kwargs...)
    compute_bands(scfres.basis, kgrid; scfres.ρ, scfres.εF, n_bands, kwargs...)
end

"""
Compute band data along a specific `Brillouin.KPath` using a `kline_density`,
the number of ``k``-points per inverse bohrs (i.e. overall in units of length).

If not given, the path is determined automatically by inspecting the `Model`.
If you are using spin, you should pass the `magnetic_moments` as a kwarg to
ensure these are taken into account when determining the path.
"""
function compute_bands(basis_or_scfres, kpath::KPath; kline_density=40u"bohr", kwargs...)
    kinter = Brillouin.interpolate(kpath, density=austrip(kline_density))
    res = compute_bands(basis_or_scfres, ExplicitKpoints(kpath_get_kcoords(kinter));
                        kwargs...)
    merge(res, (; kinter))
end
function compute_bands(scfres::NamedTuple; magnetic_moments=[], kwargs...)
    compute_bands(scfres, irrfbz_path(scfres.basis.model, magnetic_moments); kwargs...)
end
function compute_bands(basis::AbstractBasis; magnetic_moments=[], kwargs...)
    compute_bands(basis, irrfbz_path(basis.model, magnetic_moments); kwargs...)
end


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
number using the `space_group_number` keyword argument.
"""
function irrfbz_path(model::Model, magnetic_moments=[]; dim::Integer=model.n_dim,
                     space_group_number::Int=0)
    @assert dim ≤ model.n_dim
    for i = dim:3, j = dim:3
        if i != j && !iszero(model.lattice[i, j])
            error("Reducing the dimension for band structure plotting only allowed " *
                  "if the dropped dimensions are orthogonal to the remaining ones.")
        end
    end
    if space_group_number > 0 && dim ∈ (1, 3)
        @warn("space_group_number keyword argument unused in `irrfbz_path` unused " *
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
        if space_group_number == 0
            error("space_group_number keyword argument (specifying the ITA space group number) " *
                  "is required for band structure plots in 2D lattices.")
        end
        # TODO We assume to have the conventional lattice here.
        lattice_2d = [model.lattice[1:2, 1], model.lattice[1:2, 2]]
        kpath = Brillouin.irrfbz_path(space_group_number, lattice_2d, Val(2))
    elseif dim == 3
        # Brillouin.jl has an interface to Spglib.jl to directly reduce the passed
        # lattice to the ITA conventional lattice and so the Spglib cell can be
        # directly used as an input.
        kpath = Brillouin.irrfbz_path(spglib_cell(model, magnetic_moments))
    end

    # TODO In case of absence of time-reversal symmetry we need to explicitly
    #      add the inverted kpath here!
    #      See https://github.com/JuliaMolSim/DFTK.jl/pull/496/files#r725203554

    kpath
end
# TODO We should generalise irrfbz_path to AbstractSystem and move that to Brillouin


"""Return kpoint coordinates in reduced coordinates"""
function kpath_get_kcoords(kinter::KPathInterpolant{D}) where {D}
    map(k -> vcat(k, zeros_like(k, 3 - D)), kinter)
end
function kpath_get_branch(kinter::KPathInterpolant{D}, ibranch::Integer) where {D}
    map(k -> vcat(k, zeros_like(k, 3 - D)), kinter.kpaths[ibranch])
end

function kdistances_and_ticks(kcoords, klabels::Dict, kbranches)
    # kcoords in Cartesian coordinates, klabels uses Cartesian coordinates
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
    ticks = (; distances=tick_distances, labels=tick_labels)
    (; kdistances, ticks)
end


function data_for_plotting(band_data; datakeys=[:eigenvalues, :eigenvalues_error])
    kinter   = band_data.kinter
    basis    = band_data.basis
    n_spin   = basis.model.n_spin_components
    n_kcoord = length(basis.kpoints) ÷ n_spin
    n_bands  = nothing

    # XXX Convert KPathInterpolant => kbranches, klabels
    kbranches = [1:length(kinter.kpaths[1])]
    for n in length.(kinter.kpaths)[2:end]
        push!(kbranches, kbranches[end].stop+1:kbranches[end].stop+n)
    end
    klabels = Dict{Symbol, Vec3{eltype(kinter[1])}}()
    for (ibranch, labels) in enumerate(kinter.labels)
        for (k, v) in pairs(labels)
            # Convert to Cartesian and add to labels
            klabels[v] = basis.model.recip_lattice * kpath_get_branch(kinter, ibranch)[k]
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
        for σ = 1:n_spin, (ito, ik) in enumerate(krange_spin(basis, σ))
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
    for ib = 1:n_bands
        some_larger  = any(εk[ib] > εF + tol for εk in eigenvalues)
        some_smaller = any(εk[ib] < εF - tol for εk in eigenvalues)
        some_larger && some_smaller && return true
    end
    false
end

function default_band_εrange(eigenvalues; εF=nothing)
    if isnothing(εF)
        # Can't decide where the interesting region is. Just plot everything
        (minimum(minimum, eigenvalues), maximum(maximum, eigenvalues))
    else
        # Stolen from Pymatgen
        width = is_metal(eigenvalues, εF) ? 10u"eV" : 4u"eV"
        (εF - austrip(width), εF + austrip(width))
    end
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
Compute and plot the band structure. Kwargs are like in [`compute_bands`](@ref).
Requires Plots.jl to be loaded to be defined and working properly.
The unit used to plot the bands can be selected using the `unit` parameter.
Like in the rest of DFTK Hartree is used by default. Another standard choices is
`unit=u"eV"` (electron volts).
"""
function plot_bandstructure end


"""
Write the computed bands to a file. On all processes, but the master one the
`filename` is ignored. `save_ψ` determines whether the wavefunction
is also saved or not. Note that this function can be both used on the results
of [`compute_bands`](@ref) and [`self_consistent_field`](@ref).

!!! warning "Changes to data format reserved"
    No guarantees are made with respect to the format of the keys at this point.
    We may change this incompatibly between DFTK versions (including patch versions).
    In particular changes with respect to the ψ structure are planned.
"""
function save_bands(filename::AbstractString, band_data::NamedTuple; save_ψ=false)
    filename = MPI.bcast(filename, 0, MPI.COMM_WORLD)
    _, ext = splitext(filename)
    ext = Symbol(ext[2:end])

    # Whitelist valid extensions
    !(ext in (:json, :jld2)) && error("Extension '$ext' not supported by DFTK.")

    # When writing these functions keep in mind that k-Point data is
    # distributed across MPI processors and thus this function is called
    # on *all* MPI processors.
    save_bands(Val(ext), filename, band_data; save_ψ)
end

function save_bands(::Any, filename::AbstractString, ::NamedTuple; kwargs...)
    error("The extension $(last(splitext(filename))) is currently not available. " *
          "A required package (e.g. JSON3 or JLD2) is not yet loaded.")
end
