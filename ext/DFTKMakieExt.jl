module DFTKMakieExt

using DFTK
using DFTK: is_metal, data_for_plotting, spin_components, default_band_εrange
using Makie
using Unitful
using UnitfulAtomic
using LinearAlgebra

# Import DFTK stub functions so we can extend them natively
import DFTK: plot_bandstructure, plot_dos, plot_bandstructure!, plot_dos!, plot_unfolded_bands!, plot_unfolded_bands, plot_folded_bands, plot_folded_bands!, plot_dos_rotated!

@info "DFTKMakieExt successfully loaded!"

function plot_bandstructure!(pos, band_data; title_text="Band Structure", y_limits=nothing)
    data = DFTK.data_for_plotting(band_data)
    eshift = something(band_data.εF, 0.0)
    to_unit = ustrip(auconvert(u"eV", 1.0)) 
    
    ax = Makie.Axis(pos, title=title_text, xlabel="Wave Vector", 
                    ylabel=isnothing(band_data.εF) ? "Energy (eV)" : "Energy - εF (eV)")
    
    for σ = 1:data.n_spin, iband = 1:data.n_bands, branch in data.kbranches
        energies = (data.eigenvalues[:, iband, σ][branch] .- eshift) .* to_unit
        Makie.lines!(ax, data.kdistances[branch], energies, 
                     color = σ == 1 ? :blue : :red, linewidth=2)
    end
    
    for branch in data.kbranches[1:end-1]
        Makie.vlines!(ax, [data.kdistances[last(branch)]], color=:black, linewidth=1)
    end
    
    ax.xticks = (data.ticks.distances, data.ticks.labels)
    Makie.xlims!(ax, 0, data.kdistances[end])
    
    if !isnothing(band_data.εF)
        Makie.hlines!(ax, [0.0], color=:green, linewidth=2, linestyle=:dash)
    end

    # --- NEW: Apply custom Y-axis limits if provided ---
    if !isnothing(y_limits)
        Makie.ylims!(ax, y_limits[1], y_limits[2])
    end
    
    return ax
end

function plot_bandstructure(band_data; kwargs...)
    fig = Figure(size=(800, 600), fontsize=18)
    plot_bandstructure!(fig[1, 1], band_data; kwargs...)
    return fig
end

function plot_dos!(pos, scfres; n_points=500, title_text="Density of States")
    εF = scfres.εF
    eshift = something(εF, 0.0)
    to_unit = ustrip(auconvert(u"eV", 1.0))
    
    εrange = DFTK.default_band_εrange(scfres.eigenvalues; εF=εF)
    εs = range(εrange[1], εrange[2], length=n_points)
    
    Dεs = DFTK.compute_dos.(εs, Ref(scfres.basis), Ref(scfres.eigenvalues); 
                            smearing=scfres.basis.model.smearing, 
                            temperature=scfres.basis.model.temperature)
    
    energies = (εs .- eshift) .* to_unit
    n_spin = scfres.basis.model.n_spin_components
    
    ax = Axis(pos, title=title_text, 
              xlabel=isnothing(εF) ? "Energy (eV)" : "Energy - εF (eV)", 
              ylabel="Density of States")
    
    for σ = 1:n_spin
        D = [Dσ[σ] for Dσ in Dεs]
        lines!(ax, energies, D, label="Spin $σ", color = σ == 1 ? :blue : :red, linewidth=2)
    end
    
    if !isnothing(εF)
        vlines!(ax, [0.0], color=:green, linewidth=2, linestyle=:dash)
    end
    if n_spin > 1
        axislegend(ax)
    end
    return ax
end

function plot_dos(scfres; kwargs...)
    fig = Figure(size=(800, 600), fontsize=18)
    plot_dos!(fig[1, 1], scfres; kwargs...)
    return fig
end

function plot_dos_rotated!(pos, scfres; n_points=500, title_text="Total DOS")
    εF = scfres.εF
    eshift = something(εF, 0.0)
    to_unit = ustrip(auconvert(u"eV", 1.0))
    
    εrange = DFTK.default_band_εrange(scfres.eigenvalues; εF=εF)
    εs = range(εrange[1], εrange[2], length=n_points)
    
    Dεs = DFTK.compute_dos.(εs, Ref(scfres.basis), Ref(scfres.eigenvalues); 
                            smearing=scfres.basis.model.smearing, 
                            temperature=scfres.basis.model.temperature)
    
    energies = (εs .- eshift) .* to_unit
    n_spin = scfres.basis.model.n_spin_components
    
    ax = Makie.Axis(pos, title=title_text, xlabel="DOS (states/eV)")
    
    for σ = 1:n_spin
        D = [Dσ[σ] for Dσ in Dεs]
        # SWAP AXES: Plot DOS on X, Energy on Y
        Makie.lines!(ax, D, energies, color = :blue, linewidth=2)
    end
    
    if !isnothing(εF)
        Makie.hlines!(ax, [0.0], color=(:black, 0.4), linewidth=1, linestyle=:dash)
    end
    
    Makie.xlims!(ax, low=0.0) # Force DOS axis to start at 0
    return ax
end

function plot_unfolded_bands!(pos, unfold_data; title_text="Unfolded Bands", colormap=:turbo)
    eshift = something(unfold_data.εF, 0.0)
    to_unit = ustrip(auconvert(u"eV", 1.0))
    
    ax = Makie.Axis(pos, title=title_text, xlabel="Wavevector", 
                    ylabel=isnothing(unfold_data.εF) ? "Energy (eV)" : "Energy (eV - Ef)")
    
    sc = nothing # Initialize variable to hold the scatter object
    
    for spin in 1:length(unfold_data.k_indices)
        k_idxs = unfold_data.k_indices[spin]
        x_vals = unfold_data.kdistances[k_idxs]
        y_vals = (unfold_data.eigenvalues[spin] .- eshift) .* to_unit
        weights = unfold_data.spectral_weights[spin]
        
        # --- FIX: Sort data so high-weight points are drawn last (on top) ---
        perm = sortperm(weights)
        x_sorted = x_vals[perm]
        y_sorted = y_vals[perm]
        weights_sorted = weights[perm]
        
        # Plot with sorted arrays
        sc = Makie.scatter!(ax, x_sorted, y_sorted;
                            color=weights_sorted, colormap=colormap, 
                            colorrange=(0.0, 1.0), # Lock scale from 0 to 1
                            markersize=4, strokewidth=0)
    end
    
    # Formatting X-axis with high-symmetry points
    ax.xticks = (unfold_data.ticks.distances, unfold_data.ticks.labels)
    for dist in unfold_data.ticks.distances[1:end-1]
        Makie.vlines!(ax, [dist], color=(:black, 0.2), linewidth=1)
    end
    
    Makie.xlims!(ax, 0, unfold_data.kdistances[end])
    if !isnothing(unfold_data.εF)
        Makie.hlines!(ax, [0.0], color=(:black, 0.4), linewidth=1, linestyle=:dash)
    end
    
    # Return both the axis AND the scatter object so we can build a colorbar
    return ax, sc 
end

function plot_unfolded_bands(unfold_data; kwargs...)
    fig = Figure(size=(800, 600), fontsize=18)
    plot_unfolded_bands!(fig[1, 1], unfold_data; kwargs...)
    return fig
end

function plot_folded_bands!(pos, band_data_raw, unfold_data; title_text="Raw Supercell Bands (Folded)")
    eshift = something(unfold_data.εF, 0.0)
    to_unit = ustrip(auconvert(u"eV", 1.0))
    
    ax = Makie.Axis(pos, title=title_text, xlabel="Wave Vector", 
                    ylabel=isnothing(unfold_data.εF) ? "Energy (eV)" : "Energy (eV - Ef)")
    
    n_kpoints = length(unfold_data.kdistances)
    # Get the number of bands from the first k-point
    n_bands = length(band_data_raw.eigenvalues[1]) 
    
    # Trace a continuous line for each individual band
    for ib in 1:n_bands
        # Extract the energy of band `ib` across all k-points
        band_energies = [(band_data_raw.eigenvalues[ik][ib] - eshift) * to_unit for ik in 1:n_kpoints]
        
        # Plot as a solid line
        Makie.lines!(ax, unfold_data.kdistances, band_energies, 
                     color=(:black, 0.5), linewidth=1.5)
    end
    
    # Format X-axis with high-symmetry points
    ax.xticks = (unfold_data.ticks.distances, unfold_data.ticks.labels)
    for dist in unfold_data.ticks.distances[1:end-1]
        Makie.vlines!(ax, [dist], color=(:black, 0.3), linewidth=1, linestyle=:dash)
    end
    
    Makie.xlims!(ax, 0, unfold_data.kdistances[end])
    if !isnothing(unfold_data.εF)
        Makie.hlines!(ax, [0.0], color=:green, linewidth=2, linestyle=:dash)
    end
    
    return ax
end

function plot_folded_bands(band_data_raw; kwargs...)
    fig = Figure(size =(800, 600), fontsize=18)
    plot_folded_bands!(fig[1,1], band_data_raw; kwargs...)
    return fig
end

end # module