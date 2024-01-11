"""
Load back an `scfres`, which has previously been stored with [`save_scfres`](@ref).
Note the warning in [`save_scfres`](@ref).

If `basis` is `nothing`, the basis is also loaded and reconstructed from the file,
in which case `architecture=CPU()`. If a `basis` is passed, this one is used, which
can be used to continue computation on a slightly different model or to avoid the
cost of rebuilding the basis. If the stored basis and the passed basis are inconsistent
(e.g. different FFT size, Ecut, k-points etc.) the `load_scfres` will error out.

By default the `energies` and `ham` (`Hamiltonian` object) are recomputed. To avoid
this, set `skip_hamiltonian=true`. On errors the routine exits unless `strict=false`
in which case it tries to recover from the file as much data as it can, but then the
resulting `scfres` might not be fully consistent.

!!! warning "No compatibility guarantees"
    No guarantees are made with respect to this function at this point.
    It may change incompatibly between DFTK versions (including patch versions)
    or stop working / be removed in the future.
"""
@timing function load_scfres(filename::AbstractString, basis=nothing;
                             skip_hamiltonian=false, strict=true)
    _, ext = splitext(filename)
    ext = Symbol(ext[2:end])
    if !(ext in (:jld2, :hdf5))
        error("Extension '$ext' not supported by DFTK.")
    end
    load_scfres(Val(ext), filename, basis; skip_hamiltonian, strict)
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
  Stores the density, spin density, optionally bands and some metadata.
- **json**: A JSON file with basic information about the SCF run. Stores for example
   the number of iterations, occupations, some information about the basis,
   eigenvalues, Fermi level etc.

Keyword arguments:
- `save_ψ`: Save the orbitals as well (may lead to larger files). This is the default
  for `jld2`, but `false` for all other formats, where this is considerably more
  expensive.
- `save_ρ`: Save the density as well (may lead to larger files). This is the default
  for all but `json`.
- `extra_data`: Additional data to place into the file. The data is just copied
  like `fp["key"] = value`, where `fp` is a `JLD2.JLDFile`, `WriteVTK.vtk_grid`
  and so on.
- `compress`: Apply compression to array data. Requires the `CodecZlib` package
  to be available.

!!! warning "Changes to data format reserved"
    No guarantees are made with respect to the format of the keys at this point.
    We may change this incompatibly between DFTK versions (including patch versions).
    In particular changes with respect to the ψ structure are planned.
"""
@timing function save_scfres(filename::AbstractString, scfres::NamedTuple;
                             save_ψ=nothing, extra_data=Dict{String,Any}(),
                             compress=false, save_ρ=true)
    filename = MPI.bcast(filename, 0, MPI.COMM_WORLD)

    _, ext = splitext(filename)
    ext = Symbol(ext[2:end])
    if !(ext in (:jld2, :vts, :json))
        error("Extension '$ext' not supported by DFTK.")
    end
    save_ψ = something(save_ψ, (ext == :jld2))
    save_scfres(Val(ext), filename, scfres; save_ψ, save_ρ, extra_data, compress)
end
function save_scfres(::Any, filename::AbstractString, ::NamedTuple; kwargs...)
    error("The extension $(last(splitext(filename))) is currently not available. " *
          "A required package (e.g. JLD2 or JSON3) is not yet loaded.")
end
