function _run_wannier90_jll(fileprefix::AbstractString, postproc::Bool=false)
    dir, prefix = dirname(fileprefix), basename(fileprefix)
    options = postproc ? ["-pp"] : []
    wannier90_jll.wannier90(exe -> run(Cmd(`$exe $options $prefix`; dir)))
end


@doc raw"""
Wannerize the obtained bands using Wannier90.

# Arguments
- `scfres`: the result of scf.
- `fileprefix`: the prefix of the Wannier90 files to be saved.
- `exclude_bands`: the bands to be excluded from Wannierization.
    By default all converged bands from the `scfres` are employed.
- `n_wann`: the number of Wannier functions to be computed.
    By default use random Gaussians as initial guesses.
- `spin`: for `collinear` calculation: 1 for spin up, 2 for spin down.

# Keyword arguments
- `kwargs`: will be passed to [`save_wannier`](@ref). All keyword arguments
    supported by Wannier90 can be added as keyword arguments.

# Return
- `fileprefixes`: for `collinear` calculation:
    `["$(fileprefix)_up", "$(fileprefix)_dn"]`, otherwise `["$(fileprefix)"]`.

!!! warning "Experimental feature"
    Currently this is an experimental feature, which has not yet been tested
    to full depth. The interface is considered unstable and may change
    incompatibly in the future. Use at your own risk and please report bugs
    in case you encounter any.
"""
function run_wannier90(
    scfres::NamedTuple;
    fileprefix::AbstractString="wannier/wannier",
    exclude_bands::AbstractArray{<:Integer}=_default_exclude_bands(scfres),
    kwargs...,
)
    basis, ψ, eigenvalues = unfold_scfres_wannier(scfres, exclude_bands)

    n_bands = length(eigenvalues[1])
    @assert haskey(kwargs, :n_wann) "Must specify `n_wann` in `kwargs`"
    n_wann = kwargs[:n_wann]

    # Save files for Wannierization
    if basis.model.spin_polarization == :collinear
        prefixes = ["$(fileprefix)_up", "$(fileprefix)_dn"]
    else
        prefixes = [fileprefix]
    end

    unk = get(kwargs, :wannier_plot, false)
    εF = auconvert(u"eV", scfres.εF).val

    for (spin, prefix) in enumerate(prefixes)
        @timing "Compute b-vectors" begin
            win = get_wannier90_win(basis; num_wann=n_wann, num_bands=n_bands)
            dir = dirname(prefix)
            isdir(dir) || mkpath(dir)
            WannierIO.write_win("$(prefix).win"; win...)
            _run_wannier90_jll(prefix, true)
            nnkp = WannierIO.read_nnkp("$(prefix).nnkp")
            # TODO I am naming kpb_G as kpb_b in WannierIO.jl for the moment,
            # probably going to rename it in the future.
            kpb_k, kpb_G = nnkp.kpb_k, nnkp.kpb_b
            nnkp = (; kpb_k, kpb_G)
        end

        @timing "Compute & save Wannier matrices" save_wannier(
            basis, ψ, eigenvalues; fileprefix=prefix, nnkp, spin, unk,
            fermi_energy=εF, kwargs...)

        @timing "Run Wannierization" _run_wannier90_jll(prefix)
    end
    prefixes
end
