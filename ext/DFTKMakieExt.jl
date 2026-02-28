module DFTKMakieExt

using DFTK
using DFTK: is_metal, data_for_plotting, spin_components, default_band_εrange
using Makie
using Unitful
using UnitfulAtomic
using LinearAlgebra

# Import DFTK stub functions so we can extend them natively
import DFTK: plot_bandstructure, plot_dos, plot_spin_3d, plot_spin_slice, plot_bandstructure!, plot_dos!, plot_spin_3d!, plot_spin_slice!

@info "DFTKMakieExt successfully loaded!"

function plot_spin_3d!(pos, scfres; density_threshold=0.05, arrow_scale=1.0, stride=1.0, 
                       title_text="Magnetization", tip_len=0.8, tip_rad=0.35, line_rad=0.25)
    
    data = DFTK.get_spin_3d_data(scfres; density_threshold=density_threshold, stride=stride)

    ax = Axis3(pos[1, 1], title=title_text, aspect=:data, 
               xlabel="x (Bohr)", ylabel="y (Bohr)", zlabel="z (Bohr)",
               elevation=pi/6, azimuth=pi/4) 

    if !isempty(data.X)
        arrows3d!(ax, data.X, data.Y, data.Z, data.U, data.V, data.W;
                lengthscale=arrow_scale, color=data.mags, colormap=:plasma, 
                colorrange=(0, data.max_mag), 
                tiplength=tip_len, 
                tipradius=tip_rad, 
                shaftradius=line_rad)
    end
    Colorbar(pos[1, 2], limits=(0, data.max_mag), colormap=:plasma, label="Magnetization (μB)")
    return ax
end

function plot_spin_3d(scfres; kwargs...)
    fig = Figure(size=(1000, 800), fontsize=20)
    plot_spin_3d!(fig[1, 1], scfres; kwargs...)
    return fig
end

function plot_spin_slice!(pos, scfres; axis=:z, stride=2, scale=1.5, 
                          title_text="Spin Slice", tip_len=14, tip_wid=12, line_wid=2.0)
    
    data = DFTK.get_spin_slice_data(scfres; axis=axis, stride=stride, scale=scale)
    if !data.has_spin; error("No spin data found."); end

    ax = Axis(pos[1, 1], title=title_text, xlabel=data.xl, ylabel=data.yl, aspect=DataAspect())
    hm = heatmap!(ax, data.X_axis, data.Y_axis, data.h_data, 
                  colormap=:balance, colorrange=(-data.clim_val, data.clim_val))
    Colorbar(pos[1, 2], hm, label="Out-of-Plane Spin")

    if !isempty(data.X_ar)
        arrows2d!(ax, data.X_ar, data.Y_ar, data.U_ar, data.V_ar;
                  tiplength=tip_len, tipwidth=tip_wid, shaftwidth=line_wid, color=:black)
    end
    return ax
end

function plot_spin_slice(scfres; kwargs...)
    fig = Figure(size=(800, 700), fontsize=18)
    plot_spin_slice!(fig[1, 1], scfres; kwargs...)
    return fig
end

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

end # module