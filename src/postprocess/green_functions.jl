using KrylovKit

# The Green function is
# ∫_k e^ik(x-y) u_nk(x) u_nk(y) / (E-εnk) dk
# ∫_k e^ik(x-y) 1/(E-Hk) (δ_y)_k dk
# where Hk is the periodic hamiltonian
# δ_y has decomposition (up to normalizations)
# δ_y = ∫_k ∑_G e^i(k+G)(x-y) dk = ∫_k e^ikx ∑_G e^-i(k+G)y e^iG x
# so Bloch transform ∑_G e^-i(k+G)y e^iG x = P δ_y

function compute_periodic_green_function(basis::PlaneWaveBasis, y, E;
                                        alpha=0.1, deltaE=0.1, n_bands=10,
                                        tol=1e-6, maxiter=100, 
                                        Rmin=[0, 0, 0], Rmax=[0, 0, 0])
    @assert basis.model.n_spin_components == 1 "Only spinless systems supported"
    @assert length(basis.model.symmetries) == 1 "Symmetry must be disabled"
    
    # Generate R_vectors from Rmin and Rmax
    R_vectors = []
    for i in Rmin[1]:Rmax[1]
        for j in Rmin[2]:Rmax[2]
            for k in Rmin[3]:Rmax[3]
                push!(R_vectors, [i, j, k])
            end
        end
    end
    
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

    # Assemble Green's function - optimized version
    # Loop over k-points first, then R vectors, so iffts are only done once per k-point
    recip_lattice = basis.model.recip_lattice
    r_vecs = r_vectors(basis)
    nx, ny, nz = basis.fft_size
    
    # Initialize dictionary to store Green's function for each R
    G_dict = Dict{Vector{Int}, Array{ComplexF64, 3}}()
    for R in R_vectors
        G_dict[R] = zeros(ComplexF64, basis.fft_size)
    end
    
    # Loop over k-points (outer loop)
    for (ik, kpt) in enumerate(basis.kpoints)
        k = kpt.coordinate
        h_k = h_values[ik]
        u_k = u_k_solutions[ik]
        weight = weights[ik]
        
        # Transform u to real space once per k-point
        u_real = ifft(basis.fft_grid, kpt, u_k)
        
        # Deformed k-point: k̃ = k + im·h
        kdef = k .+ im .* h_k

        # Vectorize over R vectors (inner loop)
        for R in R_vectors
            for (ir, r) in enumerate(r_vecs)
                G_dict[R][ir] += weight * cis2pi(sum(kdef .* (r .+ R))) * u_real[ir]
            end
        end
    end
    
    # Build extended Green's function array
    ncell = Tuple(Rmax .- Rmin .+ 1)
    G_extended = zeros(ComplexF64, ncell[1] * nx, ncell[2] * ny, ncell[3] * nz)
    
    for R in R_vectors
        i, j, k = Int(R[1]), Int(R[2]), Int(R[3])
        G_R = G_dict[R]
        
        # Place in extended grid
        ix_start = (i - Rmin[1]) * nx + 1
        iy_start = (j - Rmin[2]) * ny + 1
        iz_start = (k - Rmin[3]) * nz + 1
        G_extended[ix_start:ix_start+nx-1, iy_start:iy_start+ny-1, iz_start:iz_start+nz-1] = G_R
    end
    
    # Create fractional coordinate arrays for extended grid
    # These are in fractional coordinates (relative to lattice vectors)
    r_frac_coords = range(Rmin[1], Rmax[1]+1, length=ncell[1]*nx+1)[1:end-1]
    s_frac_coords = range(Rmin[2], Rmax[2]+1, length=ncell[2]*ny+1)[1:end-1]
    t_frac_coords = range(Rmin[3], Rmax[3]+1, length=ncell[3]*nz+1)[1:end-1]
    
    return (; G_dict, G_extended, r_frac_coords, s_frac_coords, t_frac_coords)
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
    kgrid = basis.kgrid
    !(kgrid isa MonkhorstPack) && error("Only MonkhorstPack grids supported")
    
    T = eltype(eltype(h_values))
    kgrid_size = kgrid.kgrid_size
    n_kpts = length(basis.kpoints)
    
    # Initialize output
    nabla_h_values = [(zeros(T, 3, 3)) for _ in 1:n_kpts]
    
    for α in 1:3
        kgrid_size[α] == 1 && continue
        
        dk = 1.0 / kgrid_size[α]
        eα = [β == α ? dk : 0.0 for β in 1:3]
        
        # Compute permutations once per direction
        ik_fwd = k_to_kpq_permutation(basis, eα)
        ik_bwd = k_to_kpq_permutation(basis, -eα)
        
        # Update all k-points for this direction
        for ik in 1:n_kpts
            nabla_h_values[ik][α, :] = (h_values[ik_fwd[ik]] - h_values[ik_bwd[ik]]) / (2dk)
        end
    end
    
    return Mat3.(nabla_h_values)
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
    kinetic_term = only([t for t in basis.model.term_types if t isa Kinetic])
    kin = kinetic_energy(kinetic_term, basis.Ecut, Gplusk_vectors_cart(basis, kpt))
    
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
    
    h_dot_kG = [im * sum(h_k_cart .* kG) for kG in k_plus_G_vectors]
    h_squared = sum(h_k_cart .^ 2)
    
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
    
    G_vecs = G_vectors(basis, kpt)
    
    return [cis2pi(-dot(G, y)) * exp(dot(h_k_cart, y_cart)) / Omega for G in G_vecs]
end
