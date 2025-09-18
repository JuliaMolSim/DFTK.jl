module DFTKPlotsExt
using AtomsBase
using Brillouin: KPath
using DFTK
using DFTK: is_metal, data_for_plotting, spin_components, default_band_εrange
import DFTK: plot_dos, plot_bandstructure, plot_ldos, plot_pdos
using Plots
using Unitful
using UnitfulAtomic


function plot_bandstructure(basis::PlaneWaveBasis,
                            kpath::KPath=irrfbz_path(basis.model);
                            unit=u"hartree", kwargs_plot=(; ), kwargs...)
    @warn("Calling plot_bandstructure without first computing the band data " *
          "is deprecated and will be removed in the next minor version bump.")
    band_data = compute_bands(basis; kwargs...)
    plot_bandstructure(band_data; unit, kwargs_plot...)
end
function plot_bandstructure(band_data::NamedTuple;
                            unit=u"hartree", kwargs_plot=(; ), kwargs...)
    # TODO Replace by a plot recipe once BandData is its own type.

    mpi_nprocs() > 1 && error("Band structure plotting with MPI not supported yet")

    if !haskey(band_data, :kinter)
        @warn("Calling plot_bandstructure without first computing the band data " *
              "is deprecated and will be removed in the next minor version bump.")
        band_data = compute_bands(band_data; kwargs...)
        kwargs = kwargs_plot
    end

    eshift = something(band_data.εF, 0.0)
    data = data_for_plotting(band_data)

    # Constant to convert from AU to the desired unit
    to_unit = ustrip(auconvert(unit, 1.0))

    # Plot all bands, spins and errors
    p = Plots.plot(; xlabel="wave vector")
    margs = length(band_data.kinter) < 70 ? (; markersize=2, markershape=:circle) : (; )
    for σ = 1:data.n_spin, iband = 1:data.n_bands, branch in data.kbranches
        yerror = nothing
        if hasproperty(data, :eigenvalues_error)
            yerror = data.eigenvalues_error[:, iband, σ][branch] .* to_unit
        end
        energies = (data.eigenvalues[:, iband, σ][branch] .- eshift) .* to_unit
        Plots.plot!(p, data.kdistances[branch], energies;
                    label="", yerror, color=(:blue, :red)[σ], margs..., kwargs...)
    end

    # Delimiter for branches
    for branch in data.kbranches[1:end-1]
        Plots.vline!(p, [data.kdistances[last(branch)]], color=:black, label="")
    end

    # X-range: 0 to last kdistance value
    Plots.xlims!(p, (0, data.kdistances[end]))
    Plots.xticks!(p, data.ticks.distances, data.ticks.labels)
    if !isnothing(band_data.εF)
        Plots.hline!(p, [0.0], label="εF", color=:green, lw=1.5)
    end

    ylims = to_unit .* (default_band_εrange(band_data.eigenvalues; band_data.εF) .- eshift)
    Plots.ylims!(p, round.(ylims, sigdigits=2)...)
    if isnothing(band_data.εF)
        Plots.ylabel!(p, "eigenvalues  ($(string(unit)))")
    else
        Plots.ylabel!(p, "eigenvalues - εF  ($(string(unit)))")
    end

    p

end

function plot_dos(basis, eigenvalues; εF=nothing, unit=u"hartree",
                  temperature=basis.model.temperature,
                  smearing=basis.model.smearing,
                  εrange=default_band_εrange(eigenvalues; εF), n_points=1000, kwargs...)
    # TODO Should also split this up into one stage doing the DOS computation
    #      and one stage doing the DOS plotting (like now for the bands.)

    n_spin = basis.model.n_spin_components
    eshift = something(εF, 0.0)
    εs = range(austrip.(εrange)..., length=n_points)

    # Constant to convert from AU to the desired unit
    to_unit = ustrip(auconvert(unit, 1.0))

    p = Plots.plot(; kwargs...)
    spinlabels = spin_components(basis.model)
    colors = [:blue, :red]
    Dεs = compute_dos.(εs, Ref(basis), Ref(eigenvalues); smearing, temperature)
    for σ = 1:n_spin
        D = [Dσ[σ] for Dσ in Dεs]
        label = n_spin > 1 ? "DOS $(spinlabels[σ]) spin" : "DOS"
        Plots.plot!(p, (εs .- eshift) .* to_unit, D; label, color=colors[σ])
    end
    if !isnothing(εF)
        Plots.vline!(p, [0.0], label="εF", color=:green, lw=1.5)
    end

    if isnothing(εF)
        Plots.xlabel!(p, "eigenvalues  ($(string(unit)))")
    else
        Plots.xlabel!(p, "eigenvalues -ε_F  ($(string(unit)))")
    end
    p
end
plot_dos(scfres; kwargs...) = plot_dos(scfres.basis, scfres.eigenvalues; scfres.εF, kwargs...)

function plot_ldos(basis, eigenvalues, ψ; εF=nothing, unit=u"hartree",
                   temperature=basis.model.temperature,
                   smearing=basis.model.smearing,
                   εrange=default_band_εrange(eigenvalues; εF),
                   n_points=1000, ldos_xyz=[:, 1, 1], kwargs...)
    eshift = something(εF, 0.0)
    εs = range(austrip.(εrange)..., length=n_points)

    # Constant to convert from AU to the desired unit
    to_unit = ustrip(auconvert(unit, 1.0))

    # LDε has three dimensions (x, y, z)
    # map on a single axis to plot the variation with εs
    LDεs = dropdims.(compute_ldos.(εs, Ref(basis), Ref(eigenvalues), Ref(ψ); smearing, temperature); dims=4)
    LDεs_slice = similar(LDεs[1], n_points, length(LDεs[1][ldos_xyz...]))
    for (i, LDε) in enumerate(LDεs)
        LDεs_slice[i, :] = LDε[ldos_xyz...]
    end
    p = heatmap(1:size(LDεs_slice, 2), (εs .- eshift) .* to_unit, LDεs_slice; kwargs...)
    if !isnothing(εF)
        Plots.hline!(p, [0.0], label="εF", color=:green, lw=1.5)
    end

    if isnothing(εF)
        Plots.ylabel!(p, "eigenvalues  ($(string(unit)))")
    else
        Plots.ylabel!(p, "eigenvalues -ε_F  ($(string(unit)))")
    end
    p
end
plot_ldos(scfres; kwargs...) = plot_ldos(scfres.basis, scfres.eigenvalues, scfres.ψ; scfres.εF, kwargs...)

function plot_pdos(basis::PlaneWaveBasis{T}, eigenvalues, ψ; iatom, label=nothing,
                   positions=basis.model.positions,
                   εF=nothing, unit=u"hartree",
                   temperature=basis.model.temperature,
                   smearing=basis.model.smearing,
                   colors = [:blue, :red],
                   εrange=default_band_εrange(eigenvalues; εF),
                   n_points=1000, p=nothing, kwargs...) where {T}
    eshift = something(εF, 0.0)
    εs = range(austrip.(εrange)..., length=n_points)
    n_spin = basis.model.n_spin_components
    isnothing(iatom) ? species = "all atoms" : species = basis.model.atoms[iatom].species
    isnothing(label) ? orb_name = "all orbitals" : orb_name = label

    to_unit = ustrip(auconvert(unit, 1.0))

    # Plot pdos
    isnothing(p) && (p = Plots.plot(; kwargs...))
    p = Plots.plot(p; kwargs...)
    spinlabels = spin_components(basis.model)
    pdos = DFTK.sum_pdos(compute_pdos(εs, basis, ψ, eigenvalues;
                                      positions, temperature, smearing), 
                         [DFTK.OrbitalManifold(;iatom, label)])
    for σ = 1:n_spin
        plot_label = n_spin > 1 ? "$(species) $(orb_name) $(spinlabels[σ]) spin" : "$(species) $(orb_name)"
        Plots.plot!(p, (εs .- eshift) .* to_unit, pdos[σ]*2/n_spin; label=plot_label, color=colors[σ])
    end
    if !isnothing(εF)
        Plots.vline!(p, [0.0], label="εF", color=:green, lw=1.5)
    end

    if isnothing(εF)
        Plots.xlabel!(p, "eigenvalues  ($(string(unit)))")
    else
        Plots.xlabel!(p, "eigenvalues - ε_F  ($(string(unit)))")
    end
    p
end
plot_pdos(scfres; kwargs...) = plot_pdos(scfres.basis, scfres.eigenvalues, scfres.ψ; scfres.εF, kwargs...)

end