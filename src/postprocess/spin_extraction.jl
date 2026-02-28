using LinearAlgebra

"""
    get_spin_3d_data(scfres; density_threshold=0.01, stride=1, arrow_scale=0.8, head_frac=0.35, head_width=0.2)

Extracts 3D magnetization data. Returns a NamedTuple containing:
1. Raw vector arrays (X, Y, Z, U, V, W) optimized for Makie.jl natively rendered arrows.
2. NaN-separated coordinate wireframes (Lx, Ly, Lz, C_vals) with explicit arrowhead geometry for Plots.jl.
"""
function get_spin_3d_data(scfres; density_threshold=0.01, stride=1, arrow_scale=0.8, head_frac=0.35, head_width=0.2)
    basis = scfres.basis
    model = basis.model
    ρ = scfres.ρ
    
    # 1. Check for Spin
    if model.spin_polarization in (:none, :spinless)
        return (; has_spin=false)
    end

    # 2. Extract Components
    if model.spin_polarization == :collinear
        n_tot = ρ[:, :, :, 1] .+ ρ[:, :, :, 2]
        mx = zeros(size(n_tot))
        my = zeros(size(n_tot))
        mz = ρ[:, :, :, 1] .- ρ[:, :, :, 2]
    else
        n_tot = ρ[:, :, :, 1]
        mx, my, mz = ρ[:, :, :, 2], ρ[:, :, :, 3], ρ[:, :, :, 4]
    end

    r_vecs = DFTK.r_vectors_cart(basis)
    
    # 3. Masking (Threshold & Stride)
    density_mask = n_tot .> (density_threshold * maximum(n_tot))
    stride_mask = falses(size(n_tot))
    stride_mask[1:stride:end, 1:stride:end, 1:stride:end] .= true
    final_mask = density_mask .& stride_mask
    
    # ---------------------------------------------------------
    # PART A: RAW VECTOR DATA (For Makie.jl)
    # ---------------------------------------------------------
    X = [r[1] for r in r_vecs][final_mask]
    Y = [r[2] for r in r_vecs][final_mask]
    Z = [r[3] for r in r_vecs][final_mask]
    U, V, W = mx[final_mask], my[final_mask], mz[final_mask]
    
    mags = sqrt.(U.^2 .+ V.^2 .+ W.^2)
    max_mag = isempty(mags) ? 1.0 : maximum(mags)

    # ---------------------------------------------------------
    # PART B: WIREFRAME GEOMETRY (For Plots.jl)
    # ---------------------------------------------------------
    Lx, Ly, Lz, C_vals = Float64[], Float64[], Float64[], Float64[]
    
    if !isempty(X)
        for i in 1:length(X)
            start_pt = [X[i], Y[i], Z[i]]
            vec = [U[i], V[i], W[i]]
            m = mags[i]
            
            end_pt = start_pt .+ vec .* arrow_scale
            vec_dir = end_pt .- start_pt
            vnorm = norm(vec_dir)
            
            if vnorm < 1e-5; continue; end
            dir = vec_dir ./ vnorm

            # Draw the Main Shaft
            push!(Lx, start_pt[1], end_pt[1], NaN)
            push!(Ly, start_pt[2], end_pt[2], NaN)
            push!(Lz, start_pt[3], end_pt[3], NaN)
            push!(C_vals, m, m, NaN)

            # Calculate Orthogonal Vectors for Arrowhead Base
            if abs(dir[3]) < 0.99
                n1 = [dir[2], -dir[1], 0.0]
            else
                n1 = [0.0, dir[3], -dir[2]]
            end
            n1 ./= norm(n1)
            n2 = [dir[2]*n1[3] - dir[3]*n1[2], dir[3]*n1[1] - dir[1]*n1[3], dir[1]*n1[2] - dir[2]*n1[1]]

            # Draw 4 Arrowhead "Wings"
            for angle in (0.0, π/2, π, 3π/2)
                wing = end_pt .- (head_frac .* vec_dir) .+ (head_width * vnorm) .* (cos(angle) .* n1 .+ sin(angle) .* n2)
                push!(Lx, end_pt[1], wing[1], NaN)
                push!(Ly, end_pt[2], wing[2], NaN)
                push!(Lz, end_pt[3], wing[3], NaN)
                push!(C_vals, m, m, NaN)
            end
        end
    end
    
    # Return EVERYTHING!
    return (; has_spin=true, 
              X, Y, Z, U, V, W, mags, max_mag,  # Makie Needs
              Lx, Ly, Lz, C_vals)               # Plots.jl Needs
end

"""
    get_spin_slice_data(scfres; axis=:z, slice_index=nothing, stride=2, scale=1.0)

Extracts a 2D slice of the magnetization. Returns a NamedTuple containing:
1. `X_axis`, `Y_axis`, and `h_data` (2D matrix) for Heatmap plotting.
2. Flattened, masked 1D arrays (`X_ar`, `Y_ar`, `U_ar`, `V_ar`) for Quiver/Arrow plotting.
"""
function get_spin_slice_data(scfres; axis=:z, slice_index=nothing, stride=2, scale=1.0)
    basis = scfres.basis
    model = basis.model
    ρ = scfres.ρ
    
    if model.spin_polarization in (:none, :spinless)
        return (; has_spin=false)
    end

    if model.spin_polarization == :collinear
        mx, my = zeros(size(ρ,1), size(ρ,2), size(ρ,3)), zeros(size(ρ,1), size(ρ,2), size(ρ,3))
        mz = ρ[:, :, :, 1] .- ρ[:, :, :, 2]
    else
        mx, my, mz = ρ[:, :, :, 2], ρ[:, :, :, 3], ρ[:, :, :, 4]
    end
    
    dims = size(mx)
    if axis == :z  
        k = isnothing(slice_index) ? dims[3]÷2 : slice_index
        h_data, u_data, v_data = mz[:, :, k], mx[:, :, k], my[:, :, k]
        xl, yl = "x (grid)", "y (grid)"
    elseif axis == :y 
        j = isnothing(slice_index) ? dims[2]÷2 : slice_index
        h_data, u_data, v_data = my[:, j, :], mx[:, j, :], mz[:, j, :]
        xl, yl = "x (grid)", "z (grid)"
    elseif axis == :x 
        i = isnothing(slice_index) ? dims[1]÷2 : slice_index
        h_data, u_data, v_data = mx[i, :, :], my[i, :, :], mz[i, :, :]
        xl, yl = "y (grid)", "z (grid)"
    end
    
    nx, ny = size(h_data)
    X_axis, Y_axis = 1:nx, 1:ny
    
    clim_val = maximum(abs.(h_data))
    if clim_val < 1e-5; clim_val = 0.01; end 
    
    # Pre-calculate the masked, flattened arrays for Quiver/Arrows
    X_ar, Y_ar, U_ar, V_ar = Float64[], Float64[], Float64[], Float64[]
    for x in 1:stride:nx, y in 1:stride:ny
        u, v = u_data[x, y], v_data[x, y]
        mag = sqrt(u^2 + v^2)
        
        if mag > 1e-4 # Only push if there is actual in-plane magnetization
            push!(X_ar, x)
            push!(Y_ar, y)
            push!(U_ar, u * scale * 5)
            push!(V_ar, v * scale * 5)
        end
    end
    
    return (; has_spin=true, X_axis, Y_axis, h_data, X_ar, Y_ar, U_ar, V_ar, clim_val, xl, yl)
end

function plot_spin_3d end
function plot_spin_3d! end
function plot_spin_slice end
function plot_spin_slice! end
