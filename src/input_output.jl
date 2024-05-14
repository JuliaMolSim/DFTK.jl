# Implementation of functions for displaying and serialising key DFTK
# struct. This includes the show and todict functions for Model, Kpoints,
# PlaneWaveBasis to avoid clutter in their respective files. For "shorter" files
# (e.g. elements.jl, Energies.jl) the preference is still to keep
# show in the same file to avoid forgetting to add a print statement
# for an added field.

function Base.show(io::IO, model::Model)
    nD = model.n_dim == 3 ? "" : "$(model.n_dim)D, "
    print(io, "Model(", model.model_name, ", ", nD,
          "spin_polarization = :", model.spin_polarization, ")")
end

function Base.show(io::IO, ::MIME"text/plain", model::Model)
    println(io, "Model(", model.model_name, ", $(model.n_dim)D):")
    for i = 1:3
        header = i==1 ? "lattice (in Bohr)" : ""
        showfieldln(io, header, (@sprintf "[%-10.6g, %-10.6g, %-10.6g]" model.lattice[i, :]...))
    end
    dimexp = get(Dict(1 => "", 2 => "²", 3 => "³"), model.n_dim, "")
    showfieldln(io, "unit cell volume", @sprintf "%.5g Bohr%s" model.unit_cell_volume dimexp)

    if !isempty(model.atoms)
        println(io)
        showfieldln(io, "atoms", chemical_formula(model))
        for (i, el) in enumerate(model.atoms)
            header = i==1 ? "atom potentials" : ""
            showfieldln(io, header, el)
        end
    end

    println(io)
    if !isnothing(model.n_electrons)
        showfieldln(io, "num. electrons", model.n_electrons)
    end
    showfieldln(io, "spin polarization",  model.spin_polarization)
    showfieldln(io, "temperature",        @sprintf "%.5g Ha" model.temperature)
    if model.temperature > 0
        showfieldln(io, "smearing",       model.smearing)
    end
    if !isnothing(model.εF)
        showfieldln(io, "fixed Fermi level", model.εF)
    end

    println(io)
    for (i, term) in enumerate(model.term_types)
        header = i==1 ? "terms" : ""
        showfield(io, header, sprint(show, "text/plain", term))
        i < length(model.term_types) && println(io)
    end
end

# Make a new sublevel, which works slightly differently across data formats
make_subdict!(dict::Dict, name::AbstractString) = get!(dict, name, Dict{String,Any}())

"""
Convert a `Model` struct to a dictionary representation.
Intended to give a condensed set of useful metadata to post-processing scripts or
for storing computational results (e.g. bands, bloch waves etc.).

Some details on the conventions for the returned data:
- `lattice`, `recip_lattice`: Always a zero-padded 3x3 matrix, independent on the actual dimension
- `atomic_positions`, `atomic_positions_cart`:
  Atom positions in fractional or Cartesian coordinates, respectively.
- `atomic_symbols`: Atomic symbols if known.
- `terms`: Some rough information on the terms used for the computation.
- `n_electrons`: Number of electrons, may be missing if `εF` is fixed instead
- `εF`: Fixed Fermi level to use, may be missing if `n_electrons` is is specified instead.
"""
todict(model::Model) = todict!(Dict{String,Any}(), model)
function todict!(dict, model::Model)
    dict["model_name"]        = model.model_name
    dict["lattice"]           = model.lattice
    dict["recip_lattice"]     = model.recip_lattice
    dict["n_dim"]             = model.n_dim
    dict["spin_polarization"] = model.spin_polarization
    dict["n_spin_components"] = model.n_spin_components
    dict["temperature"]       = model.temperature
    dict["smearing"]          = string(model.smearing)
    dict["n_atoms"]           = length(model.atoms)
    dict["atomic_symbols"]    = map(e -> string(atomic_symbol(e)), model.atoms)
    dict["atomic_positions"]  = model.positions
    dict["atomic_positions_cart"] = vector_red_to_cart.(model, model.positions)
    !isnothing(model.εF)          && (dict["εF"]          = model.εF)
    !isnothing(model.n_electrons) && (dict["n_electrons"] = model.n_electrons)

    dict["symmetries_rotations"]    = [symop.W for symop in model.symmetries]
    dict["symmetries_translations"] = [symop.w for symop in model.symmetries]
    dict["terms"] = map(model.term_types) do term
        sprint(show, "text/plain", term)
    end
    dict
end

function Base.show(io::IO, kpoint::Kpoint)
    print(io, "KPoint(", (@sprintf "[%6.3g, %6.3g, %6.3g]" kpoint.coordinate...),
          ", spin = $(kpoint.spin), num. G vectors = ",
          (@sprintf "%5d" length(kpoint.G_vectors)), ")")
end

function Base.show(io::IO, basis::PlaneWaveBasis)
    print(io, "PlaneWaveBasis(model = ", basis.model, ", Ecut = ", basis.Ecut, " Ha")
    print(io, ", kgrid = ", basis.kgrid, ")")
end

function Base.show(io::IO, ::MIME"text/plain", basis::PlaneWaveBasis)
    println(io, "PlaneWaveBasis discretization:")

    showfieldln(io, "architecture", basis.architecture)
    showfieldln(io, "num. mpi processes", mpi_nprocs(basis.comm_kpts))
    showfieldln(io, "num. julia threads", Threads.nthreads())
    showfieldln(io, "num. blas  threads", BLAS.get_num_threads())
    showfieldln(io, "num. fft   threads", FFTW.get_num_threads())
    println(io)

    showfieldln(io, "Ecut",     basis.Ecut, " Ha")
    showfieldln(io, "fft_size", basis.fft_size, ", ", prod(basis.fft_size), " total points")
    if !basis.variational
        showfieldln(io, "variational", "false")
    end
    showfieldln(io, "kgrid",    basis.kgrid)
    showfieldln(io, "num.   red. kpoints", length(basis.kgrid))
    showfieldln(io, "num. irred. kpoints", length(basis.kcoords_global))

    println(io)
    modelstr = sprint(show, "text/plain", basis.model)
    indent = " " ^ SHOWINDENTION
    print(io, indent, "Discretized " * replace(modelstr, "\n" => "\n" * indent))
end


"""
Convert a `PlaneWaveBasis` struct to a dictionary representation.
Intended to give a condensed set of useful metadata to post-processing scripts or
for storing computational results (e.g. bands, bloch waves etc.). As such
the function is lossy and might not keep all data consistently. Returns
the same result on all MPI processors. See also the [`todict`](@ref) function
for the `Model`, which is called from this one to merge the data of both outputs.

Some details on the conventions for the returned data:
- `dvol`: Volume element for real-space integration
- `variational`: Is the k-point specific basis (for ψ) variationally consistent
  with the basis for ρ.
- `kweights`: Weights for the k-points, summing to 1.0
"""
todict(basis::PlaneWaveBasis) = todict!(Dict{String,Any}(), basis)
function todict!(dict, basis::PlaneWaveBasis)
    todict!(dict, basis.model)

    dict["kgrid"]        = sprint(show, "text/plain", basis.kgrid)
    dict["kcoords"]      = basis.kcoords_global
    dict["kcoords_cart"] = vector_red_to_cart.(basis.model, basis.kcoords_global)
    dict["kweights"]     = basis.kweights_global
    dict["n_kpoints"]    = length(basis.kcoords_global)
    dict["fft_size"]     = basis.fft_size
    dict["dvol"]         = basis.dvol
    dict["Ecut"]         = basis.Ecut
    dict["variational"]  = basis.variational
    dict["symmetries_respect_rgrid"] = basis.symmetries_respect_rgrid
    dict["use_symmetries_for_kpoint_reduction"] = basis.use_symmetries_for_kpoint_reduction

    # Update the symmetry as discretisation might have broken some symmetries
    delete!(dict, "symmetries_rotations")
    delete!(dict, "symmetries_translations")
    dict["symmetries_rotations"]    = [symop.W for symop in basis.symmetries]
    dict["symmetries_translations"] = [symop.w for symop in basis.symmetries]

    dict
end


"""
Convert a band computational result to a dictionary representation.
Intended to give a condensed set of results and useful metadata
for post processing. See also the [`todict`](@ref) function
for the [`Model`](@ref) and the [`PlaneWaveBasis`](@ref), which are
called from this function and the outputs merged. Note, that only
the master process returns meaningful data. All other processors
still return a dictionary (to simplify code in calling locations),
but the data may be dummy.

Some details on the conventions for the returned data:
- `εF`: Computed Fermi level (if present in band_data)
- `labels`: A mapping of high-symmetry k-Point labels to the index in
  the `kcoords` vector of the corresponding k-point.
- `eigenvalues`, `eigenvalues_error`, `occupation`, `residual_norms`:
  `(n_bands, n_kpoints, n_spin)` arrays of the respective data.
- `n_iter`: `(n_kpoints, n_spin)` array of the number of iterations the
  diagonalization routine required.
- `kpt_max_n_G`: Maximal number of G-vectors used for any k-point.
- `kpt_n_G_vectors`: `(n_kpoints, n_spin)` array, the number of valid G-vectors
  for each k-point, i.e. the extend along the first axis of `ψ` where data
  is valid.
- `kpt_G_vectors`: `(3, max_n_G, n_kpoints, n_spin)` array of the integer
  (reduced) coordinates of the G-points used for each k-point.
- `ψ`: `(max_n_G, n_bands, n_kpoints, n_spin)` arrays where `max_n_G` is the maximal
  number of G-vectors used for any k-point. The data is zero-padded, i.e.
  for k-points which have less G-vectors than max_n_G, then there are
  tailing zeros.
"""
function band_data_to_dict(band_data::NamedTuple; kwargs...)
    band_data_to_dict!(Dict{String,Any}(), band_data; kwargs...)
end
function band_data_to_dict!(dict, band_data::NamedTuple; save_ψ=false, save_ρ=nothing)
    # TODO Quick and dirty solution for now.
    #      The better would be to have a BandData struct and use
    #      a `todict` function for it, which does essentially this.
    #      See also the todo in compute_bands above.
    basis = band_data.basis
    todict!(dict, basis)

    n_bands = length(band_data.eigenvalues[1])
    dict["n_bands"] = n_bands  # n_spin_components and n_kpoints already stored

    if !isnothing(band_data.εF)
        haskey(dict, "εF") && delete!(dict, "εF")
        dict["εF"] = band_data.εF
    end
    if haskey(band_data, :kinter)
        dict["labels"] = map(band_data.kinter.labels) do labeldict
            Dict(k => string(v) for (k, v) in pairs(labeldict))
        end
    end

    function gather_and_store!(dict, key, basis, data)
        gathered = gather_kpts_block(basis, data)
        if !isnothing(gathered)
            n_kpoints = length(basis.kcoords_global)
            n_spin    = basis.model.n_spin_components
            dict[key] = reshape(gathered, (size(data[1])..., n_kpoints, n_spin))
        end
    end

    for key in (:eigenvalues, :eigenvalues_error, :occupation)
        if hasproperty(band_data, key) && !isnothing(getproperty(band_data, key))
            gather_and_store!(dict, string(key), basis, getproperty(band_data, key))
        end
    end

    if haskey(band_data, :diagonalization)
        diagonalization = make_subdict!(dict, "diagonalization")
        diag_resid  = last(band_data.diagonalization).residual_norms
        diag_n_iter = sum(diag -> diag.n_iter, band_data.diagonalization)
        diagonalization["n_matvec"] = mpi_sum(sum(diag -> diag.n_matvec,
                                                  band_data.diagonalization),
                                              band_data.basis.comm_kpts)
        diagonalization["converged"] = mpi_min(last(band_data.diagonalization).converged,
                                               band_data.basis.comm_kpts)
        gather_and_store!(diagonalization, "residual_norms", basis, diag_resid)
        gather_and_store!(diagonalization, "n_iter",         basis, diag_n_iter)
    end

    if save_ψ
        # Store the employed G vectors using the largest rectangular grid
        # on which all bands can live
        n_G_vectors = [length(kpt.mapping) for kpt in basis.kpoints]
        max_n_G = mpi_max(maximum(n_G_vectors), basis.comm_kpts)
        kpt_G_vectors = map(basis.kpoints) do kpt
            Gs_full = zeros(Int, 3, max_n_G)
            for (iG, G) in enumerate(G_vectors(basis, kpt))
                Gs_full[:, iG] = G
            end
            Gs_full
        end
        dict["kpt_max_n_G"]     = max_n_G
        gather_and_store!(dict, "kpt_n_G_vectors", basis, n_G_vectors)
        gather_and_store!(dict, "kpt_G_vectors",   basis, kpt_G_vectors)

        # TODO This gather_and_store! actually allocates a full array
        #      of size (max_n_G, n_bands, n_kpoints), which can lead to
        #      the master process running out of memory.
        #
        #      One way to avoid this full array allocation in the future by saving the data
        #      of each MPI rank in a separate key into the dict (one after the other).
        ψblock = blockify_ψ(basis, band_data.ψ).ψ
        gather_and_store!(dict, "ψ", basis, ψblock)
    end
    dict
end

"""
Convert an `scfres` to a dictionary representation.
Intended to give a condensed set of results and useful metadata
for post processing. See also the [`todict`](@ref) function
for the [`Model`](@ref) and the [`PlaneWaveBasis`](@ref) as well as
the [`band_data_to_dict`](@ref) functions, which are called by this
function and their outputs merged. Only the master process
returns meaningful data.

Some details on the conventions for the returned data:
- `ρ`: (fft_size[1], fft_size[2], fft_size[3], n_spin) array of density on real-space grid.
- `energies`: Dictionary / subdirectory containing the energy terms
- `converged`: Has the SCF reached convergence
- `norm_Δρ`: Most recent change in ρ during an SCF step
- `occupation_threshold`: Threshold below which orbitals are considered unoccupied
- `n_bands_converge`: Number of bands that have been fully converged numerically.
- `n_iter`: Number of iterations.
"""
function scfres_to_dict(scfres::NamedTuple; kwargs...)
    scfres_to_dict!(Dict{String,Any}(), scfres; kwargs...)
end
function scfres_to_dict!(dict, scfres::NamedTuple; save_ψ=true, save_ρ=true)
    # TODO Rename to todict(scfres) once scfres gets its proper type

    band_data_to_dict!(dict, scfres; save_ψ)

    # These are either already done above or will be ignored or dealt with below.
    special = (:ham, :basis, :energies, :stage,
               :ρ, :ψ, :eigenvalues, :occupation, :εF, :diagonalization)
    propmap = Dict(:α => :damping_value, )  # compatibility mapping
    if mpi_master()
        if save_ρ
            dict["ρ"] = scfres.ρ
        end
        energies = make_subdict!(dict, "energies")
        for (key, value) in todict(scfres.energies)
            energies[key] = value
        end

        scfres_extra_keys = String[]
        for symbol in propertynames(scfres)
            symbol in special && continue
            key = string(get(propmap, symbol, symbol))
            dict[key] = getproperty(scfres, symbol)
            push!(scfres_extra_keys, key)
        end
        dict["scfres_extra_keys"] = scfres_extra_keys
    end

    dict
end
