using PyCall
import Plots
include("../external/pymatgen.jl")

# Functionality for computing band structures, mostly using pymatgen

function high_symmetry_kpath(basis; kline_density=20)
    bandstructure = pyimport("pymatgen.symmetry.bandstructure")
    pystructure = pymatgen_structure(basis.model.lattice, basis.model.atoms)
    symm_kpath = bandstructure.HighSymmKpath(pystructure)

    kcoords, labels = symm_kpath.get_kpoints(kline_density, coords_are_cartesian=false)
    kpoints = build_kpoints(basis, kcoords)

    labels_dict = Dict{String, Vector{eltype(kcoords[1])}}()
    for (ik, k) in enumerate(kcoords)
        if length(labels[ik]) > 0
            labels_dict[labels[ik]] = k
        end
    end

    (kpoints=kpoints, klabels=labels_dict, kpath=symm_kpath.kpath["path"])
end

function compute_bands(ham::Hamiltonian, kpoints, n_bands;
                       eigensolver=lobpcg_hyper, tol=1e-5, show_progress=true)
    band_data = diagonalise_all_kblocks(eigensolver, ham, n_bands + 3;
                                        kpoints=kpoints, n_conv_check=n_bands,
                                        interpolate_kpoints=false, tol=tol,
                                        show_progress=show_progress)
    band_data.converged || (@warn "LOBPCG not converged" iterations=eigres.iterations)

    select_eigenpairs_all_kblocks(band_data, 1:n_bands)
end

function plot_band_data(basis, band_data; εF=nothing,
                        klabels=Dict{String, Vector{Float64}}(), unit=:eV)
    eshift = isnothing(εF) ? 0.0 : εF

    # Get pymatgen to computed kpoint distances and other useful info
    bs = pymatgen_bandstructure(basis, band_data, eshift, klabels)
    plotter = pyimport("pymatgen.electronic_structure.plotter")
    data = plotter.BSPlotter(bs).bs_plot_data(zero_to_efermi=false)

    # Collect some useful info in a more useful way
    if bs.is_spin_polarized
        spins = [("1", :blue), ("-1", :red)]
    else
        spins = [("1", :blue)]
    end
    n_branches = length(data["energy"])
    n_bands = size(data["energy"][1]["1"], 1)
    n_kpoints = sum(length, data["distances"])
    @assert length(band_data.kpoints) == n_kpoints

    # For each branch, plot all bands, spins and errors
    ikstart = 0
    p = Plots.plot(xlabel="wave vector")
    for ibr in 1:n_branches
        kdistances = data["distances"][ibr]
        for (spin, color) in spins, iband in 1:n_bands
            energies = data["energy"][ibr][spin][iband, :] .- eshift

            yerror = nothing
            if hasfield(typeof(band_data), :λerror)
                yerror = [band_data.λerror[ik + ikstart][iband] ./ unit_to_au(unit)
                          for ik in 1:length(kdistances)]
            end
            Plots.plot!(p, kdistances, energies / unit_to_au(unit),
                        color=color, label="", yerror=yerror)
        end
        ikstart += length(kdistances)
    end

    # X-range: 0 to last kdistance value
    Plots.xlims!(p, (0, data["distances"][end][end]))
    Plots.xticks!(p, data["ticks"]["distance"],
                  [replace(l, raw"$\mid$" => " | ") for l in data["ticks"]["label"]])

    ylims = [-4, 4]
    bs.is_metal() && (ylims = [-10, 10])
    ylims = ceil.(Int, ylims * units.eV ./ unit_to_au(unit))
    if isnothing(εF)
        Plots.ylabel!(p, "eigenvalues  ($(string(unit))")
    else
        Plots.ylabel!(p, "eigenvalues - \\epsilon_f  ($(string(unit)))")
        Plots.ylims!(p, ylims...)
    end

    p
end


# TODO This is the top-level function, which should be documented
function plot_bandstructure(ham::Hamiltonian, n_bands;
                            εF=nothing, kline_density=20, unit=:eV, kwargs...)
    # Band structure calculation along high-symmetry path
    kpoints, klabels, kpath = high_symmetry_kpath(ham.basis; kline_density=kline_density)
    println("Computing bands along kpath:")
    println("       ", join(join.(kpath, " -> "), "  and  "))
    band_data = compute_bands(ham, kpoints, n_bands; kwargs...)
    plot_band_data(ham.basis, band_data, εF=εF, klabels=klabels, unit=unit)
end
function plot_bandstructure(scfres, n_bands; kwargs...)
    plot_bandstructure(scfres.ham, n_bands; εF=scfres.εF, kwargs...)
end
