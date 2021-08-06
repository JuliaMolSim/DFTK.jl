import Brillouin

# Functionality for computing band structures
function high_symmetry_kpath(model; kline_density=20)
    # Change units from Number of kpoints per inverse Angström (i.e. unit of length)
    # to number of kpoints per inverse bohrs.
    # TODO Change interface to atomic units ... 
    kline_density = kline_density * austrip(1u"Å")

    if model.n_dim == 1  # Return fast for 1D model
        # Length of the kpath is recip_lattice[1, 1] in 1D
        n_points = ceil(Int, kline_density * model.recip_lattice[1, 1])
        return (
            kcoords = [[coord, 0, 0] for coord in range(-1//2, 1//2, length=1+n_points)],
            klabels=Dict("Γ" => zeros(3), "-1/2" => [-0.5, 0.0, 0.0], "1/2" => [0.5, 0, 0]),
            kpath=[[raw"-1/2", raw"1/2"]],
        )
    end

    conv_latt      = get_spglib_lattice(model; to_primitive=false)
    primitive_latt = get_spglib_lattice(model; to_primitive=true)
    # TODO This produces a segfault ... unify functions later
    # conv_latt, _      = spglib_standardize_cell(model.lattice, model.atoms, primitive=false)
    # primitive_latt, _ = spglib_standardize_cell(model.lattice, model.atoms, primitive=true)
    primitive_latt ≈ model.lattice || @warn "The DFTK lattice and spglib's primitive lattice do not agree."

    # Get International Tables for Crystallography (ITA) space group number
    sgnum = spglib_spacegroup_number(model)

    Rs = collect(eachcol(conv_latt))
    kp = Brillouin.irrfbz_path(sgnum, Rs)
    kcoords = collect(Brillouin.interpolate(kp, density=kline_density))

    # Need to double the points whenever a new path starts
    # (for temporary compatibility with pymatgen)
    idcs = findall(k -> any(sum(abs2, k - kcomp) < 1e-5 for kcomp in values(kp.points)), kcoords)
    @assert length(idcs) ≥ 2
    idcs = idcs[2:end-1]  # Don't duplicate first and last
    idcs = sort(append!(idcs, 1:length(kcoords)))
    kcoords = kcoords[idcs]

    T = eltype(kcoords[1])
    klabels = Dict{String,Vector{T}}(string(key) => val for (key, val) in kp.points)
    kpath = [[string(el) for el in path] for path in kp.paths]
    (; kcoords, klabels, kpath)
end

@timing function compute_bands(basis, ρ, kcoords, n_bands;
                               eigensolver=lobpcg_hyper,
                               tol=1e-3,
                               show_progress=true,
                               kwargs...)
    # Create basis with new kpoints, where we cheat by using any symmetry operations.
    ksymops = [[identity_symop()] for _ in 1:length(kcoords)]
    # For some reason rationalize(2//3) isn't supported (julia 1.6)
    myrationalize(x::T) where {T <: AbstractFloat} = rationalize(x, tol=10eps(T))
    myrationalize(x) = x
    bs_basis = PlaneWaveBasis(basis, [myrationalize.(k) for k in kcoords], ksymops)
    ham = Hamiltonian(bs_basis; ρ=ρ)

    band_data = diagonalize_all_kblocks(eigensolver, ham, n_bands + 3;
                                        n_conv_check=n_bands,
                                        tol=tol, show_progress=show_progress, kwargs...)
    if !band_data.converged
        @warn "Eigensolver not converged" iterations=band_data.iterations
    end
    merge((basis=bs_basis, ), select_eigenpairs_all_kblocks(band_data, 1:n_bands))
end


function split_into_branches(kcoords, data::Dict, klabels::Dict)
    # kcoords in cartesian coordinates, klabels uses cartesian coordinates
    function getlabel(kcoord; tol=1e-4)
        findfirst(c -> norm(c - kcoord) < tol, klabels)
    end

    branches = Any[(kindices = [0], kdistances=[0.0], ), ]
    for (ik, kcoord) in enumerate(kcoords)
        previous_kcoord = ik == 1 ? kcoords[1] : kcoords[ik - 1]
        if !isnothing(getlabel(kcoord)) && !isnothing(getlabel(previous_kcoord))
            # New branch encountered
            previous_distance = branches[end].kdistances[end]
            push!(branches, (kindices=[ik], kdistances=[previous_distance]))
        else
            # Keep adding to current branch
            distance = branches[end].kdistances[end] + norm(kcoord - kcoords[ik - 1])
            push!(branches[end].kdistances, distance)
            push!(branches[end].kindices, ik)
        end
    end

    map(branches[2:end]) do branch
        branch_data = Dict(key => data[key][branch.kindices, :, :] for key in keys(data))
        ret = (klabels=(getlabel(kcoords[branch.kindices[1]]),
                        getlabel(kcoords[branch.kindices[end]])),
               kdistances=branch.kdistances,
               kindices=branch.kindices)
        merge(ret, (; branch_data...))
    end
end


function prepare_band_data(band_data; datakeys=[:λ, :λerror],
                           klabels=Dict{String, Vector{Float64}}())
    basis = band_data.basis
    n_spin   = basis.model.n_spin_components
    n_kcoord = length(basis.kpoints) ÷ n_spin
    n_bands  = nothing

    # Convert coordinates to Cartesian
    kcoords_cart = [basis.kpoints[ik].coordinate_cart for ik in krange_spin(basis, 1)]
    klabels_cart = Dict(lal => basis.model.recip_lattice * vec for (lal, vec) in klabels)

    # Split data into branches
    data = Dict{Symbol, Any}()
    for key in datakeys
        hasproperty(band_data, key) || continue
        n_bands = length(band_data[key][1])
        data_per_kσ = similar(band_data[key][1], n_kcoord, n_bands, n_spin)
        for σ in 1:n_spin, (ito, ik) in enumerate(krange_spin(basis, σ))
            data_per_kσ[ito, :, σ] = band_data[key][ik]
        end
        data[key] = data_per_kσ
    end
    @assert !isnothing(n_bands)
    branches = split_into_branches(kcoords_cart, data, klabels_cart)

    tick_labels    = String[branches[1].klabels[1]]
    tick_distances = Float64[branches[1].kdistances[1]]
    for (i, br) in enumerate(branches)
        # Ignore branches with a single k-point
        branches[i].klabels[1] == branches[i].klabels[2] && continue

        label    = branches[i].klabels[2]
        distance = branches[i].kdistances[end]
        if i != length(branches) && branches[i+1].klabels[1] != label
            # Next branch is not continuous from the current
            label = label * " | " * branches[i+1].klabels[1]
        end
        push!(tick_labels, label)
        push!(tick_distances, distance)
    end

    (branches=branches, ticks=(distances=tick_distances, labels=tick_labels),
     n_bands=n_bands, n_kcoord=n_kcoord, n_spin=n_spin, basis=basis)
end

"""
    is_metal(band_data, εF, tol)

Determine whether the provided bands indicate the material is a metal,
i.e. where bands are cut by the Fermi level.
"""
function is_metal(band_data, εF, tol=1e-4)
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
are plotted in eV unless a different `unit` (any Unitful unit) is selected.
"""
function plot_bandstructure(basis, ρ, n_bands;
                            εF=nothing, kline_density=20, unit=u"eV", kwargs...)
    mpi_nprocs() > 1 && error("Band structures with MPI not supported yet")
    if !isdefined(DFTK, :PLOTS_LOADED)
        error("Plots not loaded. Run 'using Plots' before calling plot_bandstructure.")
    end

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
function plot_bandstructure(scfres; n_bands=nothing, kwargs...)
    # Convenience wrapper for scfres named tuples
    n_bands_scf = length(scfres.occupation[1])
    isnothing(n_bands) && (n_bands = ceil(Int, n_bands_scf + 5sqrt(n_bands_scf)))
    plot_bandstructure(scfres.basis, scfres.ρ, n_bands; εF=scfres.εF, kwargs...)
end
