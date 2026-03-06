module DFTKPlotsExt
using AtomsBase
using Brillouin: KPath
using DFTK
using DFTK: is_metal, data_for_plotting, spin_components, default_band_εrange
import DFTK: plot_dos, plot_bandstructure, plot_ldos, plot_pdos, plot_spin_slice
using Plots
using Unitful
using UnitfulAtomic
using LinearAlgebra


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
                  colors=[:blue, :red], p=nothing,
                  εrange=default_band_εrange(eigenvalues; εF), n_points=1000, kwargs...)
    # TODO Should also split this up into one stage doing the DOS computation
    #      and one stage doing the DOS plotting (like now for the bands.)

    n_spin = basis.model.n_spin_components
    eshift = something(εF, 0.0)
    εs = range(austrip.(εrange)..., length=n_points)

    # Constant to convert from AU to the desired unit
    to_unit = ustrip(auconvert(unit, 1.0))

    isnothing(p) && (p = Plots.plot())
    p = Plots.plot(p; kwargs...)
    spinlabels = spin_components(basis.model)
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

function plot_pdos(basis::PlaneWaveBasis{T}, eigenvalues, ψ; iatom=nothing, label=nothing,
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
    to_unit = ustrip(auconvert(unit, 1.0))

    species = isnothing(iatom) ? "all atoms" : "atom $(iatom) ($(basis.model.atoms[iatom].species))"
    orb_name = isnothing(label) ? "all orbitals" : label

    # Plot pdos
    isnothing(p) && (p = Plots.plot())
    p = Plots.plot(p; kwargs...)
    spinlabels = spin_components(basis.model)
    pdos = DFTK.sum_pdos(compute_pdos(εs, basis, ψ, eigenvalues;
                                      positions, temperature, smearing),
                         [l -> ((isnothing(iatom) || l.iatom == iatom)
                             && (isnothing(label) || l.label == label))])
    for σ = 1:n_spin
        plot_label = n_spin > 1 ? "$(species) $(orb_name) $(spinlabels[σ]) spin" : "$(species) $(orb_name)"
        Plots.plot!(p, (εs .- eshift) .* to_unit, pdos[:, σ]; label=plot_label, color=colors[σ])
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

"""
    plot_spin_slice(scfres; axis=:z, slice_index=nothing, stride=1, scale=1.0, title="")

Plots a 2D slice of the spin density.
- `axis`: The normal axis to the slice (`:x`, `:y`, or `:z`).
- `slice_index`: The grid index of the slice (defaults to the middle of the cell).
- `stride`: Subsampling factor for arrows (use 2 or 3 to reduce clutter).
- `scale`: Length multiplier for the magnetic arrows.
"""
function plot_spin_slice(basis, ρ; axis=:z, slice_index=nothing, scale=0.5, stride=1, title="", kwargs...)
    model = basis.model
    
    # 1. Handle Non-Spin Cases
    if model.spin_polarization in (:none, :spinless)
        return Plots.plot(title="No Spin Polarization", grid=false, showaxis=false)
    end

    # 2. Extract Components
    if model.spin_polarization == :collinear
        # For collinear, we only have Z-component magnetization (Up - Down)
        # We set Mx and My to zero so the code structure remains generic.
        mx = zeros(size(ρ,1), size(ρ,2), size(ρ,3))
        my = zeros(size(ρ,1), size(ρ,2), size(ρ,3))
        mz = ρ[:, :, :, 1] .- ρ[:, :, :, 2]
    else
        mx, my, mz = ρ[:, :, :, 2], ρ[:, :, :, 3], ρ[:, :, :, 4]
    end
    
    # 3. Slice the Data based on the requested axis
    dims = size(mx)
    
    if axis == :z
        k = isnothing(slice_index) ? dims[3]÷2 : slice_index
        # Heatmap = Out-of-plane (Mz), Arrows = In-plane (Mx, My)
        h_data = mz[:, :, k]
        u_data = mx[:, :, k]
        v_data = my[:, :, k]
        xl, yl = "x (Bohr)", "y (Bohr)"
        heatmap_title = "Color: Mz (Out-of-Plane)"
        
    elseif axis == :y
        j = isnothing(slice_index) ? dims[2]÷2 : slice_index
        # Heatmap = Out-of-plane (My), Arrows = In-plane (Mx, Mz)
        h_data = my[:, j, :]
        u_data = mx[:, j, :]
        v_data = mz[:, j, :]
        xl, yl = "x (Bohr)", "z (Bohr)"
        heatmap_title = "Color: My (Out-of-Plane)"

    elseif axis == :x
        i = isnothing(slice_index) ? dims[1]÷2 : slice_index
        # Heatmap = Out-of-plane (Mx), Arrows = In-plane (My, Mz)
        h_data = mx[i, :, :]
        u_data = my[i, :, :]
        v_data = mz[i, :, :]
        xl, yl = "y (Bohr)", "z (Bohr)"
        heatmap_title = "Color: Mx (Out-of-Plane)"
    end
    
    nx, ny = size(h_data)
    
    # 4. Generate the Heatmap (The "Background" Scalar Field)
    # We use :balance (Blue-White-Red) to clearly show Positive vs Negative domains
    limit = maximum(abs.(h_data))
    if limit < 1e-6; limit = 1.0; end
    
    p = Plots.heatmap(1:nx, 1:ny, h_data', 
        c=:balance, 
        clims=(-limit, limit),
        title=title, 
        xlabel=xl, ylabel=yl, 
        aspect_ratio=:equal,
        colorbar_title=heatmap_title,
        right_margin=15Plots.mm,
        size=(800, 700),
        dpi=300,
        kwargs...
    )

    # 5. Generate the Quiver Arrows (The "In-Plane" Vector Field)
    # We subsample using 'stride' to prevent the plot from becoming a black blob
    X, Y, U, V = Float64[], Float64[], Float64[], Float64[]
    
    for x in 1:stride:nx, y in 1:stride:ny
        u, v = u_data[x, y], v_data[x, y]
        mag = sqrt(u^2 + v^2)
        
        # Only draw arrows if there is significant magnetization
        if mag > 1e-4
            push!(X, x)
            push!(Y, y)
            push!(U, u * scale * 5) # Scale factor for visibility
            push!(V, v * scale * 5)
        end
    end

    if !isempty(X)
        Plots.quiver!(p, X, Y, quiver=(U, V), color=:black, linewidth=1.2)
    end
    
    return p
end

# Tuple Catcher
plot_spin_slice(scfres; kwargs...) = plot_spin_slice(scfres.basis, scfres.ρ; kwargs...)

end
