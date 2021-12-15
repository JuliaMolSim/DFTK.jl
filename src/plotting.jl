# This is needed to flag that the plots-dependent code has been loaded
const PLOTS_LOADED = true

function ScfPlotTrace(plt=Plots.plot(yaxis=:log); kwargs...)
    energies = nothing
    function callback(info)
        if info.stage == :finalize
            minenergy = minimum(energies[max(1, end-5):end])
            error = abs.(energies .- minenergy)
            error[error .== 0] .= NaN
            extra = ifelse(:mark in keys(kwargs), (), (mark=:x, ))
            Plots.plot!(plt, error; extra..., kwargs...)
            display(plt)
        elseif info.n_iter == 1
            energies = [info.energies.total]
        else
            push!(energies, info.energies.total)
        end
        info
    end
end


function plot_band_data(band_data; εF=nothing, klabels=Dict{String, Vector{Float64}}(),
                        kbranches=[1:length(band_data.λ)], unit=u"hartree", kwargs...)
    eshift = isnothing(εF) ? 0.0 : εF
    data = prepare_band_data(band_data; klabels, kbranches)

    # Constant to convert from AU to the desired unit
    to_unit = ustrip(auconvert(unit, 1.0))

    markerargs = ()
    if !(:markersize in keys(kwargs)) && !(:markershape in keys(kwargs))
        if length(krange_spin(band_data.basis, 1)) < 70
            markerargs = (markersize=2, markershape=:circle)
        end
    end

    # Plot all bands, spins and errors
    p = Plots.plot(xlabel="wave vector")
    for σ in 1:data.n_spin, iband = 1:data.n_bands
        yerror = nothing
        if hasproperty(data, :λerror)
            yerror = data.λerror[:, iband, σ] .* to_unit
        end
        energies = (data.λ[:, iband, σ] .- eshift) .* to_unit
        color = (:blue, :red)[σ]
        for branch in data.kbranches
            Plots.plot!(p, data.kdistances[branch], energies[branch]; color, label="",
                        yerror=yerror[branch], markerargs..., kwargs...)
        end
    end

    # X-range: 0 to last kdistance value
    Plots.xlims!(p, (0, data.kdistances[end]))
    Plots.xticks!(p, data.ticks.distances, data.ticks.labels)

    ylims = [-0.147, 0.147]
    !isnothing(εF) && is_metal(band_data, εF) && (ylims = [-0.367, 0.367])
    ylims = round.(ylims .* to_unit, sigdigits=2)
    if isnothing(εF)
        Plots.ylabel!(p, "eigenvalues  ($(string(unit)))")
    else
        Plots.ylabel!(p, "eigenvalues - ε_f  ($(string(unit)))")
        Plots.ylims!(p, ylims...)
    end

    p
end


function plot_dos(basis, eigenvalues; εF=nothing, kwargs...)
    n_spin = basis.model.n_spin_components
    εs = range(minimum(minimum(eigenvalues)) - .5,
               maximum(maximum(eigenvalues)) + .5, length=1000)

    p = Plots.plot(;kwargs...)
    spinlabels = spin_components(basis.model)
    colors = [:blue, :red]
    Dεs = compute_dos.(εs, Ref(basis), Ref(eigenvalues))
    for σ in 1:n_spin
        D = [Dσ[σ] for Dσ in Dεs]
        label = n_spin > 1 ? "DOS $(spinlabels[σ]) spin" : "DOS"
        Plots.plot!(p, εs, D, label=label, color=colors[σ])
    end
    if !isnothing(εF)
        Plots.vline!(p, [εF], label="εF", color=:green, lw=1.5)
    end
    p
end
plot_dos(scfres; kwargs...) = plot_dos(scfres.basis, scfres.eigenvalues; εF=scfres.εF, kwargs...)
