@doc raw"""
    compute_periodic_green_function(basis, y, E; alpha=0.1, deltaE=0.1, n_bands=10, tol=1e-6)

Compute periodic Green's function G(x,y;E) via complex k-point deformation.
Algorithm from https://hal.science/hal-03611185/document

NOTE: This is a simplified implementation that outlines the structure.
Full implementation requires extending DFTK to support complex k-points.
"""
function compute_periodic_green_function(basis::PlaneWaveBasis, y, E; 
                                        alpha=0.1, deltaE=0.1, n_bands=10,
                                        tol=1e-6, maxiter=100)
    @assert basis.model.n_spin_components == 1 "Only spinless systems supported"
    
    # Compute eigenfunctions
    ham = Hamiltonian(basis)
    eigres = diagonalize_all_kblocks(diag_full, ham, n_bands)
    
    # Compute h(k) and nabla h
    h_values = compute_h_values(basis, eigres, E, alpha, deltaE)
    nabla_h_values = compute_nabla_h_finite_diff(basis, h_values)
    
    # Build and solve at complex k-points
    G = zeros(ComplexF64, basis.fft_size)
    for (ik, kpt) in enumerate(basis.kpoints)
        # Complex k-point: k_c = k + im*h(k)
        k_complex = (kpt.coordinate, h_values[ik])
        
        # Weight with determinant factor
        det_factor = det(I + im * nabla_h_values[ik])
        weight = basis.kweights[ik] * det_factor
        
        # Solve (E - H) u = delta_y (simplified - needs complex k support)
        # For now, use real k-point as placeholder
        u_k = solve_at_kpoint(basis, kpt, y, E, tol)
        
        # Add contribution to Green's function
        add_green_contribution!(G, basis, k_complex, u_k, weight)
    end
    
    return G
end

"""
Compute h(k) = -alpha * sum_n grad_k lambda_nk * chi(lambda_nk - E)
Using Hellmann-Feynman theorem: grad_k lambda = 2 * <psi | (k+G) | psi>
"""
function compute_h_values(basis, eigres, E, alpha, deltaE)
    recip_lattice = basis.model.recip_lattice
    n_dim = count(!iszero, eachcol(basis.model.lattice))
    h_values = Vector{Vec3{Float64}}(undef, length(basis.kpoints))
    
    for (ik, kpt) in enumerate(basis.kpoints)
        eigenvalues = eigres.λ[ik]
        eigenvectors = eigres.X[ik]
        n_bands = length(eigenvalues)
        
        h_k = zeros(3)
        for n in 1:n_bands
            λ_n = eigenvalues[n]
            psi_n = eigenvectors[:, n]
            chi_n = exp(-(λ_n - E)^2 / deltaE^2)
            
            # Hellmann-Feynman: grad_k lambda_n in reciprocal space
            grad_lambda_n = zeros(3)
            for (iG, G) in enumerate(kpt.G_vectors)
                k_plus_G = recip_lattice * (kpt.coordinate + G)
                grad_lambda_n += abs2(psi_n[iG]) * k_plus_G
            end
            
            h_k -= alpha * chi_n * grad_lambda_n
        end
        h_values[ik] = Vec3(h_k)
    end
    return h_values
end

"""Compute nabla h using finite differences"""
function compute_nabla_h_finite_diff(basis, h_values)
    n_dim = count(!iszero, eachcol(basis.model.lattice))
    nabla_h_values = Vector{Mat3{Float64}}(undef, length(basis.kpoints))
    
    # Simple approximation: use neighboring k-points in grid
    kgrid_size = basis.kgrid isa MonkhorstPack ? basis.kgrid.kgrid_size : Vec3(1, 1, 1)
    
    for (ik, kpt) in enumerate(basis.kpoints)
        # Approximate nabla h as identity (simplified)
        # Full implementation would compute proper finite differences
        nabla_h_k = Matrix{Float64}(I, 3, 3) * 1e-2  # Small default
        nabla_h_values[ik] = Mat3(nabla_h_k)
    end
    return nabla_h_values
end

"""Solve at a single k-point (simplified version)"""
function solve_at_kpoint(basis, kpt, y, E, tol)
    # Build periodized delta function
    delta_y = build_periodized_delta(basis, kpt, y)
    
    # For now, return delta as placeholder
    # Full implementation needs GMRES with (E*I - H) operator
    return delta_y
end

"""Build periodized delta function"""
function build_periodized_delta(basis, kpt, y)
    n_G = length(kpt.G_vectors)
    delta_fourier = zeros(ComplexF64, n_G)
    Omega = basis.model.unit_cell_volume
    recip_lattice = basis.model.recip_lattice
    
    for (iG, G) in enumerate(kpt.G_vectors)
        G_cart = recip_lattice * (G + kpt.coordinate)
        y_cart = basis.model.lattice * y
        delta_fourier[iG] = exp(-im * dot(G_cart, y_cart)) / sqrt(Omega)
    end
    return delta_fourier
end

"""Add contribution to Green's function"""
function add_green_contribution!(G, basis, k_complex, u_k, weight)
    k_real, k_imag = k_complex
    kpt_real = basis.kpoints[1]  # Use first as template
    recip_lattice = basis.model.recip_lattice
    
    # Transform u to real space
    u_real = ifft(basis.fft_grid, kpt_real, u_k)
    
    # Add weighted contribution with phase
    for idx in CartesianIndices(basis.fft_size)
        x_frac = Vec3([idx[1]-1, idx[2]-1, idx[3]-1]) ./ Vec3(basis.fft_size)
        x_cart = basis.model.lattice * x_frac
        
        k_real_cart = recip_lattice * k_real
        k_imag_cart = recip_lattice * k_imag
        
        phase = exp(im * dot(k_real_cart, x_cart) - dot(k_imag_cart, x_cart))
        G[idx] += weight * phase * u_real[idx]
    end
end
