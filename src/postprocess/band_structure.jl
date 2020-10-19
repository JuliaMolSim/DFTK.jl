using PyCall

# Functionality for computing band structures, mostly using pymatgen

function high_symmetry_kpath(model; kline_density=20)
    pystructure = pymatgen_structure(model.lattice, model.atoms)
    symm_kpath = pyimport("pymatgen.symmetry.bandstructure").HighSymmKpath(pystructure)

    kcoords, labels = symm_kpath.get_kpoints(kline_density, coords_are_cartesian=false)

    labels_dict = Dict{String, Vector{eltype(kcoords[1])}}()
    for (ik, k) in enumerate(kcoords)
        if length(labels[ik]) > 0
            labels_dict[labels[ik]] = k
        end
    end

    (kcoords=kcoords, klabels=labels_dict, kpath=symm_kpath.kpath["path"])
end

@timing function compute_bands(basis, ρ, ρspin, kcoords, n_bands;
                               eigensolver=lobpcg_hyper,
                               tol=1e-3,
                               show_progress=true,
                               kwargs...)
    # Create basis with new kpoints, where we cheat by using any symmetry operations.
    ksymops = [[identity_symop()] for _ in 1:length(kcoords)]
    # For some reason rationalize(2//3) isn't supported (julia 1.4)
    myrationalize(x::T) where {T <: AbstractFloat} = rationalize(x, tol=10eps(T))
    myrationalize(x) = x
    bs_basis = PlaneWaveBasis(basis, [myrationalize.(k) for k in kcoords], ksymops)
    ham = Hamiltonian(bs_basis; ρ=ρ, ρspin=ρspin)

    band_data = diagonalize_all_kblocks(eigensolver, ham, n_bands + 3;
                                        n_conv_check=n_bands,
                                        tol=tol, show_progress=show_progress, kwargs...)
    if !band_data.converged
        @warn "Eigensolver not converged" iterations=band_data.iterations
    end
    merge((basis=bs_basis, ), select_eigenpairs_all_kblocks(band_data, 1:n_bands))
end

function prepare_band_data(band_data; datakeys=[:λ, :λerror],
                           klabels=Dict{String, Vector{Float64}}())
    # Get pymatgen to compute kpoint distances and get it to split quantities
    # from the band_data object into nicely plottable branches
    # This is a bit of abuse of the routines in pymatgen, but it works ...
    plotter = pyimport("pymatgen.electronic_structure.plotter")
    basis = band_data.basis

    # Read and parse pymatgen version
    mg_version = parse.(Int, split(pyimport("pymatgen").__version__, ".")[1:3])

    ret = Dict{Symbol, Any}(:basis => basis)
    for key in datakeys
        hasproperty(band_data, key) || continue

        # Compute dummy "Fermi level" for pymatgen to be happy
        allfinite = [filter(isfinite, x) for x in band_data[key]]
        eshift = sum(sum, allfinite) / sum(length, allfinite)
        bs = pymatgen_bandstructure(basis, band_data[key], eshift, klabels)
        data = plotter.BSPlotter(bs).bs_plot_data(zero_to_efermi=false)

        # Check number of k-Points agrees
        n_kcoords = div(length(basis.kpoints), basis.model.n_spin_components)
        @assert n_kcoords == sum(length, data["distances"])

        ret[:spins] = [:up]
        spinmap = [("1", :up)]
        if bs.is_spin_polarized
            ret[:spins] = [:up, :down]
            spinmap = [("1", :up), ("-1", :down)]
        end

        ret[:n_branches] = size(data["distances"], 1)
        ret[:kdistances] = data["distances"]
        ret[:ticks]      = data["ticks"]
        if mg_version[1:2] > [2020, 9]
            # New interface: {Spin:[np.array(nb_bands,kpoints),...]}
            ret[:n_bands] = size(data["energy"]["1"][1], 1)
            ret[key] = [Dict(spinsym => data["energy"][spin][ibranch]
                             for (spin, spinsym) in spinmap)
                        for ibranch = 1:ret[:n_branches]]
        else
            # Old interface: [{Spin:[band_index][k_point_index]}]
            ret[:n_bands] = size(data["energy"][1]["1"], 1)
            ret[key] = [Dict(spinsym => data["energy"][ibranch][spin]
                             for (spin, spinsym) in spinmap)
                        for ibranch = 1:ret[:n_branches]]
        end
    end

    (; ret...)  # Make it a named tuple and return
end

"""
    is_metal(band_data, εF, tol)

Determine whether the provided bands indicate the material is a metal,
i.e. where bands are cut by the Fermi level.
"""
function is_metal(band_data, εF, tol=1e-4)
    # This assumes no spin polarization
    @assert band_data.basis.model.spin_polarization in (:none, :spinless, :collinear)

    n_bands = length(band_data.λ[1])
    n_kpoints = length(band_data.λ)
    for ib in 1:n_bands
        some_larger = any(band_data.λ[ik][ib] - εF < -tol for ik in 1:n_kpoints)
        some_smaller = any(εF - band_data.λ[ik][ib] < -tol for ik in 1:n_kpoints)
        some_larger && some_smaller && return true
    end
    false
end

function detexify_kpoint(string)
    # For some reason Julia doesn't support this naively: https://github.com/JuliaLang/julia/issues/29849
    replacements = ("\\Gamma" => "Γ",
                    "\\Delta" => "Δ",
                    "\\Sigma" => "Σ",
                    "_1"      => "₁")
    for r in replacements
        string = replace(string, r)
    end
    string
end

"""
Compute and plot the band structure. `n_bands` selects the number of bands to compute.
If this value is absent and an `scfres` is used to start the calculation a default of
`n_bands_scf + 5sqrt(n_bands_scf)` is used. Unlike the rest of DFTK bands energies
are plotted in `:eV` unless a different `unit` is selected.
"""
function plot_bandstructure(basis, ρ, ρspin, n_bands;
                            εF=nothing, kline_density=20, unit=:eV, kwargs...)
    if !isdefined(DFTK, :PLOTS_LOADED)
        error("Plots not loaded. Run 'using Plots' before calling plot_bandstructure.")
    end

    # Band structure calculation along high-symmetry path
    kcoords, klabels, kpath = high_symmetry_kpath(basis.model; kline_density=kline_density)
    println("Computing bands along kpath:")
    println("       ", join(join.(detexify_kpoint.(kpath), " -> "), "  and  "))
    band_data = compute_bands(basis, ρ, ρspin, kcoords, n_bands; kwargs...)

    plotargs = ()
    if kline_density ≤ 10
        plotargs = (markersize=2, markershape=:circle)
    end

    plot_band_data(band_data; εF=εF, klabels=klabels, unit=unit, plotargs...)
end
function plot_bandstructure(scfres; n_bands=nothing, kwargs...)
    # Convenience wrapper for scfres named tuples
    n_bands_scf = length(scfres.occupation[1])
    isnothing(n_bands) && (n_bands = ceil(Int, n_bands_scf + 5sqrt(n_bands_scf)))
    plot_bandstructure(scfres.basis, scfres.ρ, scfres.ρspin, n_bands; εF=scfres.εF, kwargs...)
end
