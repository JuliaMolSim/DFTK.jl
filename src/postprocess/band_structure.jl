using PyCall
import Plots

# Functionality for computing band structures, mostly using pymatgen

function high_symmetry_kpath(model; kline_density=20)
    bandstructure = pyimport("pymatgen.symmetry.bandstructure")
    pystructure = pymatgen_structure(model.lattice, model.atoms)
    symm_kpath = bandstructure.HighSymmKpath(pystructure)

    kcoords, labels = symm_kpath.get_kpoints(kline_density, coords_are_cartesian=false)

    labels_dict = Dict{String, Vector{eltype(kcoords[1])}}()
    for (ik, k) in enumerate(kcoords)
        if length(labels[ik]) > 0
            labels_dict[labels[ik]] = k
        end
    end

    (kcoords=kcoords, klabels=labels_dict, kpath=symm_kpath.kpath["path"])
end

function compute_bands(basis, ρ, kcoords, n_bands;
                       eigensolver=lobpcg_hyper, tol=1e-3, show_progress=true, kwargs...)
    # Create basis with new kpoints, where we cheat by using any symmetry operations.
    ksymops = [[(Mat3{Int}(I), Vec3(zeros(3)))] for _ in 1:length(kcoords)]
    bs_basis = PlaneWaveBasis(basis, kcoords, ksymops)
    ham = Hamiltonian(bs_basis; ρ=ρ)

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

    ret = Dict{Symbol, Any}(:basis => band_data.basis)
    for key in datakeys
        hasproperty(band_data, key) || continue

        # Compute dummy "Fermi level" for pymatgen to be happy
        allfinite = [filter(isfinite, x) for x in band_data[key]]
        eshift = sum(sum, allfinite) / sum(length, allfinite)
        bs = pymatgen_bandstructure(band_data.basis, band_data[key], eshift, klabels)
        data = plotter.BSPlotter(bs).bs_plot_data(zero_to_efermi=false)

        # Check number of k-Points agrees
        @assert length(band_data.basis.kpoints) == sum(length, data["distances"])

        ret[:spins] = [:up]
        spinmap = [("1", :up)]
        if bs.is_spin_polarized
            ret[:spins] = [:up, :down]
            spinmap = [("1", :up), ("-1", :down)]
        end

        ret[:n_branches] = length(data["energy"])
        ret[:n_bands] = size(data["energy"][1]["1"], 1)
        ret[:kdistances] = data["distances"]
        ret[:ticks] = data["ticks"]
        ret[key] = [Dict(spinsym => data["energy"][ibranch][spin]
                         for (spin, spinsym) in spinmap)
                    for ibranch = 1:length(data["energy"])]
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
    @assert band_data.basis.model.spin_polarization in (:none, :spinless)

    n_bands = length(band_data.λ[1])
    n_kpoints = length(band_data.λ)
    for ib in 1:n_bands
        some_larger = any(band_data.λ[ik][ib] - εF < -tol for ik in 1:n_kpoints)
        some_smaller = any(εF - band_data.λ[ik][ib] < -tol for ik in 1:n_kpoints)
        some_larger && some_smaller && return true
    end
    false
end


function plot_band_data(band_data; εF=nothing,
                        klabels=Dict{String, Vector{Float64}}(), unit=:eV, kwargs...)
    eshift = isnothing(εF) ? 0.0 : εF
    data = prepare_band_data(band_data, klabels=klabels)

    # For each branch, plot all bands, spins and errors
    p = Plots.plot(xlabel="wave vector")
    for ibranch = 1:data.n_branches
        kdistances = data.kdistances[ibranch]
        for spin in data.spins, iband = 1:data.n_bands
            yerror = nothing
            if hasproperty(data, :λerror)
                yerror = data.λerror[ibranch][spin][iband, :] ./ unit_to_au(unit)
            end
            energies = (data.λ[ibranch][spin][iband, :] .- eshift) ./ unit_to_au(unit)

            color = (spin == :up) ? :blue : :red
            Plots.plot!(p, kdistances, energies; color=color, label="", yerror=yerror,
                        kwargs...)
        end
    end

    # X-range: 0 to last kdistance value
    Plots.xlims!(p, (0, data.kdistances[end][end]))
    Plots.xticks!(p, data.ticks["distance"],
                  [replace(l, raw"$\mid$" => " | ") for l in data.ticks["label"]])

    ylims = [-4, 4]
    is_metal(band_data, εF) && (ylims = [-10, 10])
    ylims = round.(ylims * units.eV ./ unit_to_au(unit), sigdigits=2)
    if isnothing(εF)
        Plots.ylabel!(p, "eigenvalues  ($(string(unit))")
    else
        Plots.ylabel!(p, "eigenvalues - ε_f  ($(string(unit)))")
        Plots.ylims!(p, ylims...)
    end

    p
end

function detexify_kpoint(string)
    # For some reason Julia doesn't support this naively: https://github.com/JuliaLang/julia/issues/29849
    replacements = ("\\Gamma" => "Γ",
                    "\\Delta" => "Δ",
                    "\\Sigma" => "Σ")
    for r in replacements
        string = replace(string, r)
    end
    string
end

# TODO This is the top-level function, which should be documented
function plot_bandstructure(basis, ρ, n_bands;
                            εF=nothing, kline_density=20, unit=:eV, kwargs...)
    # Band structure calculation along high-symmetry path
    kcoords, klabels, kpath = high_symmetry_kpath(basis.model; kline_density=kline_density)
    println("Computing bands along kpath:")
    println("       ", join(join.(detexify_kpoint.(kpath), " -> "), "  and  "))
    band_data = compute_bands(basis, ρ, kcoords, n_bands; kwargs...)

    plotargs = ()
    if kline_density ≤ 10
        plotargs = (markersize=2, markershape=:circle)
    end
    plot_band_data(band_data; εF=εF, klabels=klabels, unit=unit, plotargs...)
end
function plot_bandstructure(scfres, n_bands; kwargs...)
    # Convenience wrapper for scfres named tuples
    plot_bandstructure(scfres.ham.basis, scfres.ρ, n_bands; εF=scfres.εF, kwargs...)
end
