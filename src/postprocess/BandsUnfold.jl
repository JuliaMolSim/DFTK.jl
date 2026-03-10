import Base.Threads
using StaticArrays
using DFTK

"""
    unfold_bands(scfres_sc, band_data_prim; kwargs...)

Memory-optimized and thread-safe band unfolding using k-point batching.
"""
function unfold_bands(scfres_sc, band_data_prim; n_bands=scfres_sc.basis.model.n_bands_converge, weight_threshold=1e-4, batch_size=20)
    
    DFTK.@timing "Total Unfold Bands" begin
        
        DFTK.@timing "Setup S-Matrices" begin
            basis_sc = scfres_sc.basis
            is_full = basis_sc.model.spin_polarization == :full
            n_spin = basis_sc.model.n_spin_components

            lattice_prim = band_data_prim.basis.model.lattice
            recip_prim = band_data_prim.basis.model.recip_lattice
            recip_sc_S = SMatrix{3, 3, Float64}(basis_sc.model.recip_lattice)
            inv_prim_S = SMatrix{3, 3, Float64}(lattice_prim' / 2π)

            G_mapper_S = inv_prim_S * recip_sc_S
            
            prim_k_idxs = krange_spin(band_data_prim.basis, 1)
            kpath_cart = [SMatrix{3, 3, Float64}(recip_prim) * SVector{3, Float64}(band_data_prim.basis.kpoints[ik].coordinate) for ik in prim_k_idxs]
            kpath_sc_frac = [recip_sc_S \ k for k in kpath_cart]
        end

        n_kpoints = length(kpath_sc_frac)
        
        unfolded_k_idx = [Int[] for _ in 1:n_spin]
        unfolded_evals = [Float32[] for _ in 1:n_spin]
        unfolded_weights = [Float32[] for _ in 1:n_spin]

        DFTK.@timing "Batched Calculation & Unfolding" begin
            for batch_start in 1:batch_size:n_kpoints
                batch_end = min(batch_start + batch_size - 1, n_kpoints)
                batch_range = batch_start:batch_end
                batch_len = length(batch_range)
                
                @info "Unfolding batch $(batch_start) to $(batch_end) of $(n_kpoints)..."
                
                kgrid_batch = ExplicitKpoints(kpath_sc_frac[batch_range])
                band_data_sc = compute_bands(scfres_sc, kgrid_batch; n_bands=n_bands)
                basis_bands = band_data_sc.basis

                # ---------------------------------------------------------
                # BUG FIX: Storage is now per-kpoint, totally immune to thread migration
                # ---------------------------------------------------------
                t_k_idx   = [[Int[] for _ in 1:n_spin] for _ in 1:batch_len]
                t_evals   = [[Float32[] for _ in 1:n_spin] for _ in 1:batch_len]
                t_weights = [[Float32[] for _ in 1:n_spin] for _ in 1:batch_len]

                Threads.@threads for local_idx in 1:batch_len
                    global_idx_in_path = batch_range[local_idx]
                    
                    valid_iG = Int[]
                    sizehint!(valid_iG, 100)
                    
                    for spin in 1:n_spin
                        ik = krange_spin(basis_bands, spin)[local_idx]
                        Kpoint_sc = basis_bands.kpoints[ik]
                        coeffs = band_data_sc.ψ[ik]    
                        evals  = band_data_sc.eigenvalues[ik]
                        n_evals = length(evals)
                        
                        K_sc_cart_S = recip_sc_S * SVector{3, Float64}(Kpoint_sc.coordinate)
                        G_sc_vecs = G_vectors(basis_bands, Kpoint_sc)
                        n_G = length(G_sc_vecs)
                        
                        empty!(valid_iG)
                        delta_k_frac = inv_prim_S * (K_sc_cart_S - kpath_cart[global_idx_in_path])

                        @fastmath for (iG, G_int) in enumerate(G_sc_vecs)
                            G_S = SVector{3, Float64}(G_int[1], G_int[2], G_int[3])
                            n_check = delta_k_frac + G_mapper_S * G_S
                            nx, ny, nz = n_check[1], n_check[2], n_check[3]
                            if abs(nx - round(nx)) < 1e-4 && abs(ny - round(ny)) < 1e-4 && abs(nz - round(nz)) < 1e-4
                                push!(valid_iG, iG)
                            end
                        end
                        
                        if !isempty(valid_iG)
                            for ib in 1:n_evals
                                w = 0.0
                                @simd for i in eachindex(valid_iG)
                                    iG_idx = valid_iG[i]
                                    w += abs2(coeffs[iG_idx, ib])
                                end
                                
                                if is_full
                                    @simd for i in eachindex(valid_iG)
                                        iG_idx = valid_iG[i]
                                        w += abs2(coeffs[iG_idx + n_G, ib])
                                    end
                                end
                                
                                if w > weight_threshold
                                    # We now index safely using local_idx instead of threadid()
                                    push!(t_k_idx[local_idx][spin], global_idx_in_path)
                                    push!(t_evals[local_idx][spin], Float32(evals[ib]))
                                    push!(t_weights[local_idx][spin], Float32(min(w, 1.0)))
                                end
                            end
                        end
                    end 
                end 

                # Safely combine the batch results into the global array
                for local_idx in 1:batch_len
                    for spin in 1:n_spin
                        append!(unfolded_k_idx[spin], t_k_idx[local_idx][spin])
                        append!(unfolded_evals[spin], t_evals[local_idx][spin])
                        append!(unfolded_weights[spin], t_weights[local_idx][spin])
                    end
                end

                band_data_sc = nothing
                kgrid_batch = nothing
                basis_bands = nothing
                GC.gc() 
            end
        end

        plot_data = DFTK.data_for_plotting(band_data_prim)
    end 

    return (k_indices=unfolded_k_idx, eigenvalues=unfolded_evals, spectral_weights=unfolded_weights,
            kdistances=plot_data.kdistances, ticks=plot_data.ticks, εF=scfres_sc.εF)
end

"""
    compute_folded_band(scfres_sc, band_data_prim; kwargs...)

Computes the raw bands but strips out heavy wavefunctions, returning only 
the eigenvalues needed for plotting.
"""
function compute_folded_bands(scfres_sc, band_data_prim; kwargs...)
    recip_prim = band_data_prim.basis.model.recip_lattice
    recip_sc   = scfres_sc.basis.model.recip_lattice

    kpath_cart = [recip_prim * k.coordinate for k in band_data_prim.basis.kpoints]
    kpath_sc_frac = [recip_sc \ k for k in kpath_cart]

    # Compute bands (this allocates memory temporarily)
    band_data_sc = compute_bands(scfres_sc, ExplicitKpoints(kpath_sc_frac); kwargs...)
    
    # Extract only the lightweight plotting data
    evals = band_data_sc.eigenvalues
    εF = scfres_sc.εF

    # FIX: Use the primitive band data to extract the path metadata!
    plot_data = DFTK.data_for_plotting(band_data_prim)

    # Free the massive wavefunctions
    band_data_sc = nothing
    GC.gc()

    return (eigenvalues=evals, kdistances=plot_data.kdistances, ticks=plot_data.ticks, εF=εF)
end

# --- Plotting Stubs for Extensions ---
function plot_bandstructure! end
function plot_dos! end
function plot_unfolded_bands end
function plot_unfolded_bands! end
function plot_folded_bands end
function plot_folded_bands! end
function plot_dos_rotated! end