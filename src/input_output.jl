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
- lattice, recip_lattice: Always a zero-padded 3x3 matrix, independent on the actual dimension
- atomic_positions, atomic_positions_cart:
  Atom positions in fractional or cartesian coordinates, respectively.
- atomic_symbols: Atomic symbols if known.
- terms: Some rough information on the terms used for the computation.
- n_electrons: Number of electrons, may be missing if εF is fixed instead
- εF: Fixed Fermi level to use, may be missing if n_electronis is specified instead.
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
- dvol: Volume element for real-space integration
- variational: Is the k-point specific basis (for ψ) variationally consistent
  with the basis for ρ.
- kweights: Weights for the k-points, summing to 1.0
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
