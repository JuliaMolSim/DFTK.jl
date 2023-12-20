function gather_kpts_scfres(scfres::NamedTuple)
    # TODO Rename this to gather_kpts once scfres gets its proper type

    # Need gathering over k-points:
    kpt_properties = (:ψ, :occupation, :eigenvalues)
    scfdict = Dict{Symbol, Any}()
    for symbol in kpt_properties
        scfdict[symbol] = gather_kpts(getproperty(scfres, symbol), scfres.basis)
    end
    scfdict[:basis] = gather_kpts(scfres.basis)

    if mpi_master()
        (; (symbol => get(scfdict, symbol, getproperty(scfres, symbol))
            for symbol in propertynames(scfres))...)
    else
        nothing
    end
end

"""
Convert a band computational result to a dictionary representation.
Intended to give a condensed set of results and useful metadata
for post processing. See also the [`todict`](@ref) function
for the [`Model`](@ref) and the [`PlaneWaveBasis`](@ref) as well as
the [`band_data_to_dict`](@ref) functions, which are called by this
function and their outputs merged. Only the master process
returns meaningful data.

Some details on the conventions for the returned data:
- ρ: (fft_size[1], fft_size[2], fft_size[3], n_spin) array of density
  on real-space grid.

TODO more docs

"""
function scfres_to_dict(scfres::NamedTuple; save_ψ=false)
    scfres_to_dict!(Dict{String,Any}(), scfres; save_ψ)
end
function scfres_to_dict!(dict, scfres::NamedTuple; save_ψ=true)
    # TODO Rename to todict(scfres) once scfres gets its proper type

    band_data_to_dict!(dict, scfres; save_ψ)

    # These are either already done above or will be ignored or dealt with below.
    special = (:ham, :basis, :energies,
               :ρ, :ψ, :eigenvalues, :occupation, :εF, :diagonalization)
    propmap = Dict(:α => :damping_value, )  # compatibility mapping
    if mpi_master()
        dict["ρ"] = scfres.ρ
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


"""
Load back an `scfres`, which has previously been stored with [`save_scfres`](@ref).
Note the warning in [`save_scfres`](@ref).

If `basis` is `nothing`, the basis is also loaded and reconstructed from the file,
in which case `architecture=CPU()`. If a `basis` is passed, this one is used, which
can be used to continue computation on a slightly different model or to avoid the
cost of rebuilding the basis. If the stored basis and the passed basis are inconsistent
(e.g. different FFT size, Ecut, k-points etc.) the `load_scfres` will error out.

By default the `energies` and `ham` (`Hamiltonian` object) are recomputed. To avoid
this, set `skip_hamiltonian=true`.

!!! warning "No compatibility guarantees"
    No guarantees are made with respect to this function at this point.
    It may change incompatibly between DFTK versions (including patch versions)
    or stop working / be removed in the future.
"""
@timing function load_scfres(filename::AbstractString, basis=nothing; skip_hamiltonian=false)
    _, ext = splitext(filename)
    ext = Symbol(ext[2:end])
    if !(ext in (:jld2, :hdf5))
        error("Extension '$ext' not supported by DFTK.")
    end
    load_scfres(Val(ext), filename, basis; skip_hamiltonian)
end
function load_scfres(::Any, filename::AbstractString; kwargs...)
    error("The extension $(last(splitext(filename))) is currently not available. " *
          "A required package (e.g. JLD2 or HDF5) is not yet loaded.")
end


"""
Save an `scfres` obtained from `self_consistent_field` to a file. On all processes
but the master one the `filename` is ignored. The format is determined from the file extension.
Currently the following file extensions are recognized and supported:

- **jld2**: A JLD2 file. Stores the complete state and can be used
  (with [`load_scfres`](@ref)) to restart an SCF from a checkpoint or
  post-process an SCF solution. Note that this file is also a valid
  HDF5 file, which can thus similarly be read by external non-Julia libraries such
  as h5py or similar.
  See [Saving SCF results on disk and SCF checkpoints](@ref) for details.
- **vts**: A VTK file for visualisation e.g. in [paraview](https://www.paraview.org/).
  Stores the density, spin density and some metadata (energy, Fermi level,
  occupation etc.).
- **json**: A JSON file with basic information about the SCF run. Stores for example
   the number of iterations, occupations, some information about the basis,
   eigenvalues, Fermi level etc.

Keyword arguments:
- `save_ψ`: Save the orbitals as well (may lead to larger files). This is the default
  for `jld2`, but `false` for all other formats, where this is considerably more
  expensive.
- `extra_data`: Additional data to place into the file. The data is just copied
  like `fp["key"] = value`, where `fp` is a `JLD2.JLDFile`, `WriteVTK.vtk_grid`
  and so on.

!!! warning "No compatibility guarantees"
    No guarantees are made with respect to this function at this point.
    It may change incompatibly between DFTK versions (including patch versions)
    or stop working / be removed in the future.
"""
@timing function save_scfres(filename::AbstractString, scfres::NamedTuple;
                             save_ψ=nothing, extra_data=Dict{String,Any}())
    filename = MPI.bcast(filename, 0, MPI.COMM_WORLD)

    _, ext = splitext(filename)
    ext = Symbol(ext[2:end])
    if !(ext in (:jld2, :vts, :json))
        error("Extension '$ext' not supported by DFTK.")
    end
    if isnothing(save_ψ)
        save_ψ = (ext == :jld2)
    end
    save_scfres(Val(ext), filename, scfres; save_ψ, extra_data)
end
function save_scfres(::Any, filename::AbstractString, ::NamedTuple; kwargs...)
    error("The extension $(last(splitext(filename))) is currently not available. " *
          "A required package (e.g. JLD2 or JSON3) is not yet loaded.")
end
