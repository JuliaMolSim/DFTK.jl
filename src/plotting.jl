import Plots

# This is needed to flag that the plots-dependent code has been loaded
const PLOTS_LOADED = true

"""
Plot the trace of an SCF, i.e. the absolute error of the total energy at
each iteration versus the converged energy in a semilog plot. By default
a new plot canvas is generated, but an existing one can be passed and reused
along with `kwargs` for the call to `plot!`.
"""
function ScfPlotTrace(plt=Plots.plot(yaxis=:log); kwargs...)
    energies = Float64[]
    function callback(info)
        if info.stage == :finalize
            minenergy = minimum(energies[max(1, end-5):end])
            error = abs.(energies .- minenergy)
            error[error .== 0] .= NaN
            extra = ifelse(:mark in keys(kwargs), (), (mark=:x, ))
            Plots.plot!(plt, error; extra..., kwargs...)
            display(plt)
        else
            push!(energies, info.energies.total)
        end
    end
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
    !isnothing(εF) && is_metal(band_data, εF) && (ylims = [-10, 10])
    ylims = round.(ylims * units.eV ./ unit_to_au(unit), sigdigits=2)
    if isnothing(εF)
        Plots.ylabel!(p, "eigenvalues  ($(string(unit))")
    else
        Plots.ylabel!(p, "eigenvalues - ε_f  ($(string(unit)))")
        Plots.ylims!(p, ylims...)
    end

    p
end


function plot_dos(basis, eigenvalues; εF=nothing)
    n_spin = basis.model.n_spin_components
    εs = range(minimum(minimum(eigenvalues)) - .5,
               maximum(maximum(eigenvalues)) + .5, length=1000)

    p = Plots.plot()
    spinlabels = spin_components(basis.model)
    colors = [:blue, :red]
    for σ in 1:n_spin
        D = DOS.(εs, Ref(basis), Ref(eigenvalues), spins=(σ, ))
        label = n_spin > 1 ? "DOS $(spinlabels[σ]) spin" : "DOS"
        Plots.plot!(p, εs, D, label=label, color=colors[σ])
    end
    if !isnothing(εF)
        Plots.vline!(p, [εF], label="εF", color=:green, lw=1.5)
    end
    p
end
plot_dos(scfres; kwargs...) = plot_dos(scfres.basis, scfres.eigenvalues; εF=scfres.εF, kwargs...)
