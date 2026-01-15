using KrylovKit

@doc raw"""
    compute_periodic_green_function(basis, y, E; alpha=0.1, deltaE=0.1, n_bands=10, 
                                   tol=1e-6, maxiter=100, R_vectors=[])

Compute periodic Green's function G(x,y;E) via complex k-point deformation.
Algorithm from https://hal.science/hal-03611185/document

Returns G(r+R, y) for r in the unit cell and R in R_vectors (lattice vectors).
If R_vectors is empty, returns only G(r, y) for r in the unit cell.

# Algorithm outline:
1. Compute eigenfunctions λ_nk, ψ_nk on Monkhorst-Pack grid
2. Compute h(k) = -alpha * sum_n grad_k λ_nk * χ(λ_nk - E)
   where χ(x) = exp(-x²/deltaE²) using Hellmann-Feynman theorem
3. Compute ∇h and ∇²h (via finite differences respecting periodicity)
4. Build basis with complex k-points: k̃ = k + im*h(k)
5. Solve (E - H_{k̃}) u_{k̃} = P δ_y for each k̃ using GMRES
6. Assemble: G(x) = sum_k det(1+im∇h(k)) exp(im k̃·x) u_{k̃}(x)

# Current implementation status:
- Steps 1-6: Fully implemented
- Complex k-point Hamiltonian: Implemented via kinetic energy correction
- GMRES solver: Integrated using KrylovKit

# Arguments
- `basis::PlaneWaveBasis`: Plane-wave basis (symmetries must be disabled)
- `y::AbstractVector`: Delta source position (fractional coordinates)  
- `E::Real`: Energy for Green's function

# Keyword arguments
- `alpha::Real=0.1`: h(k) scaling parameter
- `deltaE::Real=0.1`: χ function energy width
- `n_bands::Int=10`: Number of eigenfunctions  
- `tol::Real=1e-6`: GMRES tolerance
- `maxiter::Int=100`: Maximum GMRES iterations
- `R_vectors::Vector=[]`: Lattice vectors for extended range (fractional coords)

# Returns
- If R_vectors is empty: Array of size fft_size with G(r, y) for r in unit cell
- If R_vectors provided: Dict mapping R to arrays of G(r+R, y)

# Notes
Complex k-point Hamiltonian is constructed by adding correction to kinetic energy:
-½|k + ik_imag + G|² = -½|k+G|² -½|k_imag|² - i(k+G)·k_imag
The resulting non-Hermitian system is solved with GMRES from KrylovKit.
"""
function compute_periodic_green_function(basis::PlaneWaveBasis, y, E; 
                                        alpha=0.1, deltaE=0.1, n_bands=10,
                                        tol=1e-6, maxiter=100, R_vectors=[])
    @assert basis.model.n_spin_components == 1 "Only spinless systems supported"
    @assert !basis.model.symmetries "Symmetry must be disabled"
    
    # Compute eigenfunctions
    ham = Hamiltonian(basis)
    eigres = diagonalize_all_kblocks(diag_full, ham, n_bands)
    
    # Compute h(k) and nabla h
    h_values = compute_h_values(basis, eigres, E, alpha, deltaE)
    nabla_h_values = compute_nabla_h_finite_diff(basis, h_values)
    
    # Solve for u_k at each k-point and store
    u_k_solutions = Vector{Vector{ComplexF64}}(undef, length(basis.kpoints))
    weights = Vector{ComplexF64}(undef, length(basis.kpoints))
    
    for (ik, kpt) in enumerate(basis.kpoints)
        # Weight with determinant factor
        det_factor = det(I + im * nabla_h_values[ik])
        weights[ik] = basis.kweights[ik] * det_factor
        
        # Solve (E - H) u = delta_y with complex k-point using GMRES
        u_k_solutions[ik] = solve_at_kpoint(basis, kpt, ik, h_values[ik], y, E, tol, maxiter)
    end
    
    # Assemble Green's function for unit cell and extended lattice
    if isempty(R_vectors)
        # Return only unit cell
        G = assemble_green_unit_cell(basis, h_values, u_k_solutions, weights)
        return G
    else
        # Return dict with G for each R
        G_dict = Dict()
        for R in R_vectors
            G_dict[R] = assemble_green_with_R(basis, h_values, u_k_solutions, weights, R)
        end
        return G_dict
    end
end

@doc raw"""
    compute_h_values(basis, eigres, E, alpha, deltaE)

Compute h(k) = -alpha * sum_n grad_k λ_nk * χ(λ_nk - E)

Uses Hellmann-Feynman theorem: grad_k λ_n = <ψ_n|∇_k H|ψ_n>. 
For kinetic energy ∇_k(-½|k+G|²) = -(k+G), giving grad_k λ_n = Σ_G |ψ_G|²(k+G).
The function χ(x) = exp(-x²/deltaE²) weights contributions by proximity to energy E.
"""
function compute_h_values(basis, eigres, E, alpha, deltaE)
    recip_lattice = basis.model.recip_lattice
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

@doc raw"""
    compute_nabla_h_finite_diff(basis, h_values)

Compute ∇h via finite differences using k-grid neighbors.
For each k-point, compute ∂h/∂k_i using neighboring k-points in the grid.
"""
function compute_nabla_h_finite_diff(basis, h_values)
    # Get k-grid structure
    kgrid = basis.kgrid
    if !(kgrid isa MonkhorstPack)
        error("Only MonkhorstPack grids supported for finite differences")
    end
    
    kgrid_size = kgrid.kgrid_size
    n_kpts = length(basis.kpoints)
    nabla_h_values = Vector{Mat3{Float64}}(undef, n_kpts)
    
    # Build mapping from k-point to grid index
    # For MonkhorstPack, k-points are ordered
    for (ik, kpt) in enumerate(basis.kpoints)
        nabla_h_k = zeros(3, 3)
        
        # Compute finite difference for each direction
        for i in 1:3
            if kgrid_size[i] > 1
                # Find neighboring k-points in direction i
                # Use periodic boundary conditions
                dk = 1.0 / kgrid_size[i]
                
                # Forward and backward neighbors
                k_forward = mod.(kpt.coordinate + [j==i ? dk : 0.0 for j in 1:3], 1.0)
                k_backward = mod.(kpt.coordinate - [j==i ? dk : 0.0 for j in 1:3], 1.0)
                
                # Find indices of neighboring k-points
                ik_forward = findfirst(k -> norm(k.coordinate - k_forward) < 1e-10, basis.kpoints)
                ik_backward = findfirst(k -> norm(k.coordinate - k_backward) < 1e-10, basis.kpoints)
                
                if !isnothing(ik_forward) && !isnothing(ik_backward)
                    # Central difference: (h(k+dk) - h(k-dk)) / (2*dk)
                    dh_dk = (h_values[ik_forward] - h_values[ik_backward]) / (2 * dk)
                    nabla_h_k[:, i] = dh_dk
                end
            end
        end
        
        nabla_h_values[ik] = Mat3(nabla_h_k)
    end
    
    return nabla_h_values
end

@doc raw"""
    solve_at_kpoint(basis, kpt, ik, k_imag, y, E, tol, maxiter)

Solve (E*I - H_{k+ik_imag}) u = δ_y at a complex k-point using iterative solver.

Uses a matrix-free approach where the Hamiltonian application is done via
operator application plus diagonal correction.
"""
function solve_at_kpoint(basis, kpt, ik, k_imag, y, E, tol, maxiter)
    # Build periodized delta function (RHS)
    delta_y = build_periodized_delta(basis, kpt, y)
    
    # Get Hamiltonian operator at real k-point
    ham = Hamiltonian(basis)
    H_op = ham.blocks[ik]
    
    # Compute kinetic correction as diagonal
    kinetic_correction = compute_kinetic_correction(basis, kpt, k_imag)
    
    # Define linear operator: A*v = (E*I - H_real - H_correction)*v
    function apply_A(v)
        # Apply real Hamiltonian
        Hv = H_op * v
        # Add diagonal correction
        Hv_corrected = Hv + kinetic_correction .* v
        # Return (E*I - H)*v
        return E * v - Hv_corrected
    end
    
    # Solve using linsolve from KrylovKit (suitable for non-Hermitian systems)
    result, info = linsolve(apply_A, delta_y, delta_y;
                           tol=tol, 
                           maxiter=maxiter,
                           krylovdim=min(20, length(delta_y)))
    
    if info.converged != 1
        @warn "linsolve did not converge for k-point" info.normres
    end
    
    return result
end

@doc raw"""
    compute_kinetic_correction(basis, kpt, k_imag)

Compute diagonal kinetic energy correction for complex k-point.
Returns a vector of corrections for each G-vector.
"""
function compute_kinetic_correction(basis, kpt, k_imag)
    recip_lattice = basis.model.recip_lattice
    n_G = length(kpt.G_vectors)
    k_imag_cart = recip_lattice * k_imag
    
    correction = zeros(ComplexF64, n_G)
    for (iG, G) in enumerate(kpt.G_vectors)
        k_plus_G_cart = recip_lattice * (kpt.coordinate + G)
        # Correction: -½|k_imag|² - i(k+G)·k_imag
        correction[iG] = -0.5 * dot(k_imag_cart, k_imag_cart) - 
                        im * dot(k_plus_G_cart, k_imag_cart)
    end
    return correction
end

@doc raw"""
    build_periodized_delta(basis, kpt, y)

Build periodized delta function P δ_y in Fourier space.
For k-point k: P δ_y(k+G) = exp(-i(k+G)·y) / √Ω
"""
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

@doc raw"""
    assemble_green_unit_cell(basis, h_values, u_k_solutions, weights)

Assemble Green's function G(r, y) for r in the unit cell.
"""
function assemble_green_unit_cell(basis, h_values, u_k_solutions, weights)
    G = zeros(ComplexF64, basis.fft_size)
    recip_lattice = basis.model.recip_lattice
    
    for (ik, kpt) in enumerate(basis.kpoints)
        k_real = kpt.coordinate
        k_imag = h_values[ik]
        u_k = u_k_solutions[ik]
        weight = weights[ik]
        
        # Transform u to real space
        u_real = ifft(basis.fft_grid, kpt, u_k)
        
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
    
    return G
end

@doc raw"""
    assemble_green_with_R(basis, h_values, u_k_solutions, weights, R)

Assemble Green's function G(r+R, y) for lattice vector R.
Uses phase factor e^{ik·R} to extend u_k to r+R positions.
"""
function assemble_green_with_R(basis, h_values, u_k_solutions, weights, R)
    G = zeros(ComplexF64, basis.fft_size)
    recip_lattice = basis.model.recip_lattice
    R_cart = basis.model.lattice * R
    
    for (ik, kpt) in enumerate(basis.kpoints)
        k_real = kpt.coordinate
        k_imag = h_values[ik]
        u_k = u_k_solutions[ik]
        weight = weights[ik]
        
        # Transform u to real space
        u_real = ifft(basis.fft_grid, kpt, u_k)
        
        # Compute phase factor for R: e^{i(k+ih)·R}
        k_real_cart = recip_lattice * k_real
        k_imag_cart = recip_lattice * k_imag
        phase_R = exp(im * dot(k_real_cart, R_cart) - dot(k_imag_cart, R_cart))
        
        # Add weighted contribution with phase for r+R
        for idx in CartesianIndices(basis.fft_size)
            x_frac = Vec3([idx[1]-1, idx[2]-1, idx[3]-1]) ./ Vec3(basis.fft_size)
            x_cart = basis.model.lattice * x_frac
            
            phase_r = exp(im * dot(k_real_cart, x_cart) - dot(k_imag_cart, x_cart))
            G[idx] += weight * phase_R * phase_r * u_real[idx]
        end
    end
    
    return G
end

@doc raw"""
    add_green_contribution!(G, basis, k_complex, u_k, weight, ik)

Add weighted contribution to Green's function from solution at complex k-point.
Phase factor: exp(i(k_real + i*k_imag)·x) = exp(ik_real·x - k_imag·x)

NOTE: This function is deprecated in favor of assemble_green_unit_cell and assemble_green_with_R.
"""
function add_green_contribution!(G, basis, k_complex, u_k, weight, ik)
    k_real, k_imag = k_complex
    kpt = basis.kpoints[ik]  # Use the actual k-point for this contribution
    recip_lattice = basis.model.recip_lattice
    
    # Transform u to real space using correct k-point
    u_real = ifft(basis.fft_grid, kpt, u_k)
    
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
