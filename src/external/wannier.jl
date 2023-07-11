"""
Call `Wannier.jl` to Wannierize the results.

# Arguments
- `scfres`: the result of scf.

`kwargs` will be passed to [`save_wannier`](@ref).
"""
function run_wannier(
    scfres::NamedTuple;
    fileprefix::AbstractString="wannier/wannier",
    exclude_bands::AbstractArray{<:Integer}=_default_exclude_bands(scfres),
    kwargs...,
)
    basis, ψ, eigenvalues = unfold_scfres_wannier(scfres, exclude_bands)

    n_bands = length(eigenvalues[1])
    @assert haskey(kwargs, :n_wann) "Must specify `n_wann` in `kwargs`"
    n_wann = kwargs[:n_wann]

    # Although with Wannier.jl we can pass matrices in-memory,
    # however I still save the files so that we can restart later.
    if basis.model.spin_polarization == :collinear
        prefixes = ["$(fileprefix)_up", "$(fileprefix)_dn"]
    else
        prefixes = [fileprefix]
    end

    @timing "Compute b-vectors" begin
        win = get_wannier90_win(basis; num_wann=n_wann, num_bands=n_bands)
        # Note I am not using directly `basis.model.recip_lattice` here since
        # that one is in Bohr unit, while `win.unit_cell_cart` is in Angstrom.
        recip_lattice = compute_recip_lattice(win.unit_cell_cart)
        bvectors = Wannier.get_bvectors(win.kpoints, recip_lattice)
        # TODO I am naming kpb_G as kpb_b in WannierIO.jl for the moment,
        # probably going to rename it in the future.
        kpb_k, kpb_G = bvectors.kpb_k, bvectors.kpb_b
        nnkp = (; kpb_k, kpb_G)
    end

    unk = get(kwargs, :wannier_plot, false)
    εF = auconvert(u"eV", scfres.εF).val

    models = []
    for (spin, prefix) in enumerate(prefixes)
        @timing "Compute & save Wannier matrices" win, M, A, E = save_wannier(
            basis, ψ, eigenvalues; fileprefix=prefix, nnkp, spin, unk,
            fermi_energy=εF, kwargs...)

        @timing "Run Wannierization" begin
            # I simply call disentangle which works for both isolated and
            # entangled cases. Also if no frozen window provided, just use
            # 0.5 eV above Fermi.
            dis_froz_max = get(win, :dis_froz_max, εF + 0.5)
            dis_froz_min = get(win, :dis_froz_min, -Inf)
            frozen_bands = Wannier.get_frozen_bands(E, dis_froz_max, dis_froz_min)

            model = Wannier.Model(
                win.unit_cell_cart, win.atoms_frac, win.atom_labels,
                win.mp_grid, win.kpoints, bvectors, frozen_bands, M, A, E)
            model.U .= Wannier.disentangle(model)

            push!(models, model)
        end
    end
    models
end
