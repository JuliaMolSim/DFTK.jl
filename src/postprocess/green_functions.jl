using KrylovKit

function compute_periodic_green_function(basis::PlaneWaveBasis, y, E;
                                        alpha=0.1, deltaE=0.1, n_bands=10,
                                        tol=1e-6, maxiter=100, R_vectors=[])
    @assert basis.model.n_spin_components == 1 "Only spinless systems supported"
    @assert length(basis.model.symmetries) == 1 "Symmetry must be disabled"
    
    # Compute eigenfunctions
    ham = Hamiltonian(basis)
    eigres = diagonalize_all_kblocks(lobpcg_hyper, ham, n_bands)
    
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

    # Assemble Green's function
    # Return dict with G for each R
    G_dict = Dict()
    for R in R_vectors
        G_dict[R] = assemble_green_with_R(basis, h_values, u_k_solutions, weights, R)
    end
    return G_dict
end
function compute_h_values(basis, eigres, E, alpha, deltaE)
    recip_lattice = basis.model.recip_lattice
    T = typeof((alpha))
    h_values = Vector{Vec3{T}}(undef, length(basis.kpoints))
    
    for (ik, kpt) in enumerate(basis.kpoints)
        eigenvalues = eigres.λ[ik]
        eigenvectors = eigres.X[ik]
        n_bands = length(eigenvalues)
        
        h_k = zeros(T, 3)
        for n in 1:n_bands
            λ_n = eigenvalues[n]
            psi_n = eigenvectors[:, n]
            chi_n = exp(-(λ_n - E)^2 / deltaE^2)
            
            # Hellmann-Feynman: grad_k lambda_n in reciprocal space
            k_plus_G_vectors = Gplusk_vectors_cart(basis, kpt)
            grad_lambda_n = sum(abs2.(psi_n) .* k_plus_G_vectors)
            
            h_k -= alpha * chi_n * grad_lambda_n
        end
        h_values[ik] = Vec3(h_k...)
    end
    return h_values
end

function compute_nabla_h_finite_diff(basis, h_values)
    # Get k-grid structure
    kgrid = basis.kgrid
    if !(kgrid isa MonkhorstPack)
        error("Only MonkhorstPack grids supported for finite differences")
    end
    
    kgrid_size = kgrid.kgrid_size
    n_kpts = length(basis.kpoints)
    # Infer type from h_values (may be complex)
    T = eltype(eltype(h_values))
    nabla_h_values = Vector{Mat3{T}}(undef, n_kpts)
    
    # Build a mapping from fractional k-coordinates to k-point indices
    # Handle periodicity correctly by wrapping to [-0.5, 0.5)
    function wrap_k(k)
        k_wrapped = mod.(k .+ 0.5, 1.0) .- 0.5
        return k_wrapped
    end
    
    function find_kpoint(k_target)
        k_wrapped = wrap_k(k_target)
        for (ik, kpt) in enumerate(basis.kpoints)
            k_current = wrap_k(kpt.coordinate)
            if norm(k_current - k_wrapped) < 1e-10
                return ik
            end
        end
        return nothing
    end
    
    # Compute finite difference for each k-point
    for (ik, kpt) in enumerate(basis.kpoints)
        nabla_h_k = zeros(T, 3, 3)
        
        # Compute finite difference for each direction
        for i in 1:3
            if kgrid_size[i] > 1
                # Grid spacing in direction i
                dk = 1.0 / kgrid_size[i]
                
                # Forward and backward neighbors in fractional coordinates
                shift = [j==i ? dk : 0.0 for j in 1:3]
                k_forward = kpt.coordinate .+ shift
                k_backward = kpt.coordinate .- shift
                
                # Find indices of neighboring k-points
                ik_forward = find_kpoint(k_forward)
                ik_backward = find_kpoint(k_backward)
                
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
operator application plus diagonal correction. Includes kinetic energy preconditioning
to improve convergence.
"""
function solve_at_kpoint(basis, kpt, ik, k_imag, y, E, tol, maxiter)
    # Build periodized delta function (RHS) at shifted k-point k̃ = k + im·h
    delta_y = build_periodized_delta(basis, kpt, k_imag, y)
    
    # Get Hamiltonian operator at real k-point
    ham = Hamiltonian(basis)
    H_op = ham.blocks[ik]
    
    # Compute kinetic correction as diagonal
    kinetic_correction = compute_kinetic_correction(basis, kpt, k_imag)
    
    # Build kinetic energy preconditioner
    # Kinetic energies: 1/2|k+G|^2
    kinetic_term = [t for t in basis.model.term_types if t isa Kinetic]
    if !isempty(kinetic_term)
        kinetic_term = only(kinetic_term)
        kin = kinetic_energy(kinetic_term, basis.Ecut, Gplusk_vectors_cart(basis, kpt))
    else
        # Fallback: compute manually if no Kinetic term
        kin = [norm2(p) / 2 for p in Gplusk_vectors_cart(basis, kpt)]
    end
    
    # Preconditioner: P ≈ diag(kinetic + shift)
    # This approximates the dominant diagonal part of the operator
    default_shift = 1.0
    precond_diag = kin .+ default_shift
    
    # Define linear operator: A*v = (E*I - H_real - H_correction)*v
    function apply_A(v)
        # Apply real Hamiltonian
        Hv = H_op * v
        # Add diagonal correction
        Hv_corrected = Hv + kinetic_correction .* v
        # Return (E*I - H)*v
        return E * v - Hv_corrected
    end
    
    # Left-preconditioned operator: P^{-1} * A
    function apply_preconditioned_A(v)
        Av = apply_A(v)
        return Av ./ precond_diag
    end
    
    # Preconditioned RHS: P^{-1} * b
    preconditioned_rhs = delta_y ./ precond_diag
    
    # Solve using linsolve from KrylovKit with preconditioning
    # Solves: (P^{-1} A) x = P^{-1} b
    result, info = linsolve(apply_preconditioned_A, preconditioned_rhs, preconditioned_rhs;
                           tol=tol, 
                           maxiter=maxiter,
                           krylovdim=min(20, length(delta_y)))
    
    if info.converged != 1
        @warn "linsolve did not converge for k-point" info.normres
    end
    
    return result
end

@doc raw"""
    compute_kinetic_correction(basis, kpt, h_k)

Compute diagonal kinetic energy correction for complex k-point.
Returns a vector of corrections for each G-vector.

When k → k̃ = k + im·h, kinetic energy becomes:
  T_{k̃} = ½|k+G + im·h|² = ½|k+G|² + im·h·(k+G) - ½h²
  ΔT = T_{k̃} - T_k = im·h·(k+G) - ½h²
"""
function compute_kinetic_correction(basis, kpt, h_k)
    recip_lattice = basis.model.recip_lattice
    h_k_cart = recip_lattice * h_k
    k_plus_G_vectors = Gplusk_vectors_cart(basis, kpt)
    
    h_dot_kG = [im * dot(h_k_cart, kG) for kG in k_plus_G_vectors]
    h_squared = dot(h_k_cart, h_k_cart)
    
    return h_dot_kG .- 0.5 * h_squared
end

@doc raw"""
    build_periodized_delta(basis, kpt, h_k, y)

Build the source term for the Green's function equation in the periodic-part basis.

In DFTK, we work with periodic parts u_k, not full Bloch functions ψ_k = e^{ik·r} u_k.
The delta function source for the equation (E - H_k) g_k = b_k should be:
  b_k(G) = e^{-iG·y} / Ω
The e^{-ik·y} factor is handled in the assembly step.

For the deformed k-point k̃ = k + im·h, we need an additional factor e^{h·y}
from the im·h part of the shift.
"""
function build_periodized_delta(basis, kpt, h_k, y)
    Omega = basis.model.unit_cell_volume
    recip_lattice = basis.model.recip_lattice
    h_k_cart = recip_lattice * h_k
    y_cart = basis.model.lattice * y
    
    G_vectors_cart = [recip_lattice * G for G in G_vectors(basis, kpt)]
    
    return [exp(-im * dot(G_cart, y_cart)) * exp(dot(h_k_cart, y_cart)) / Omega for G_cart in G_vectors_cart]
end

@doc raw"""
    assemble_green_with_R(basis, h_values, u_k_solutions, weights, R)

Assemble Green's function G(r+R, y) for lattice vector R.
Uses phase factor e^{ik̃·(r+R)} where k̃ = k + im·h.
"""
function assemble_green_with_R(basis, h_values, u_k_solutions, weights, R)
    G = zeros(ComplexF64, basis.fft_size)
    recip_lattice = basis.model.recip_lattice
    R_cart = basis.model.lattice * R
    r_vecs = r_vectors(basis)
    
    for (ik, kpt) in enumerate(basis.kpoints)
        k = kpt.coordinate
        h_k = h_values[ik]
        u_k = u_k_solutions[ik]
        weight = weights[ik]
        
        # Transform u to real space
        u_real = ifft(basis.fft_grid, kpt, u_k)
        
        # Deformed k-point: k̃ = k + im·h
        kdef = k + im * h_k
        kdef_cart = recip_lattice * kdef
        
        # Phase factor for lattice vector R: e^{ik̃·R}
        phase_R = exp(im * dot(kdef_cart, R_cart))
        
        # Add weighted contribution with phase for each r
        for idx in CartesianIndices(basis.fft_size)
            r_cart = r_vecs[idx]
            phase_r = exp(im * dot(kdef_cart, r_cart))
            G[idx] += weight * phase_R * phase_r * u_real[idx]
        end
    end
    
    return G
end
