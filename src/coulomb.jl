raw"""
    CoulombKernelModel

Abstract type for different methods of computing the 
discretised Coulomb kernel ``v(G+q) = 4π/|G+q|²``.

Available models:
- [`ProbeCharge`](@ref): Gygi-Baldereschi probe charge method (default)
- [`NeglectSingularity`](@ref): Set G+q=0 component to zero
- [`SphericallyTruncated`](@ref): Spherical truncation at radius Rcut
- [`WignerSeitzTruncated`](@ref): Wigner-Seitz cell truncation

See also: [`compute_coulomb_kernel`](@ref)
"""
abstract type CoulombKernelModel end


raw"""
    compute_coulomb_kernel(basis; q=0, coulomb_kernel_model=ProbeCharge())

Compute Coulomb kernel, i.e. essentially ``v(G+q) = 4π/|G+q|²``, on spherical plane-wave grid.

Returns the Fourier-space Coulomb interaction for momentum transfer `q`,
evaluated only on the spherical cutoff |G+q|² < 2Ecut (not the full cubic FFT grid).

!!! note "Gamma-point only"
    Currently only works for single k-point calculations (Gamma-only).
    For general k-points, a q-dependent basis would be needed.

# Arguments
- `basis::PlaneWaveBasis`: Plane-wave basis defining the grid
- `q=zero(Vec3)`: Momentum transfer vector in fractional coordinates
- `coulomb_kernel_model::CoulombKernelModel=ProbeCharge()`: Method for treating singularity

# Returns
Vector of Coulomb kernel values for each G-vector in the spherical cutoff.
"""
function compute_coulomb_kernel(basis::PlaneWaveBasis{T};
                                q=zero(Vec3{T}),
                                coulomb_kernel_model::CoulombKernelModel=ProbeCharge()) where {T}
    if length(basis.kpoints) > 1 || !iszero(basis.kpoints[1].coordinate)
        throw(ArgumentError(
            "Currently only Gamma-point calculations are supported in " *
            "compute_coulomb_kernel, respectively Hartree-Fock and " *
            "calculations involving exact exchange."))
    end

    # currently only works for Gamma-only (need correct q-point otherwise)
    qpt = basis.kpoints[1] 
    coulomb_kernel =  _compute_coulomb_kernel(basis, qpt, q, coulomb_kernel_model)

    # TODO: if q=0, symmetrize Fourier coeffs to have real iFFT 

    coulomb_kernel = to_device(basis.architecture, coulomb_kernel)
end


"""
    ProbeCharge <: CoulombKernelModel

Probe charge Ewald method for treating the Coulomb singularity.

Uses a Gaussian probe charge with width parameter α = π²/Ecut to regularize
the G+q=0 component. Well-tested in VASP for production calculations.

# Reference
Phys. Rev. B 48, 5058 (1993), doi.org/10.1103/PhysRevB.48.5058
"""
struct ProbeCharge <: CoulombKernelModel end
function _compute_coulomb_kernel(basis::PlaneWaveBasis{T},
                                 qpt::Kpoint,
                                 q::Vec3{T},
                                 coulomb_kernel_model::ProbeCharge) where {T}
    model = basis.model
    NG = length(qpt.G_vectors)
    coulomb_kernel = zeros(T, NG)
    
    # well tested in VASP such that e^(-α*G²) is localized 
    # charge with full support on G grid
    α = π^2 / basis.Ecut  
    
    probe_charge_sum = zero(T)
    for (iG, G) in enumerate(to_cpu(qpt.G_vectors))
        G_cart = model.recip_lattice * (G+q)
        Gnorm2 = sum(abs2, G_cart)
        found_singularity = (iG==1 && iszero(q))
        if !found_singularity
            coulomb_kernel[iG] = 4T(π) / Gnorm2
            probe_charge_sum += coulomb_kernel[iG] * exp(-α*Gnorm2)
        end
    end

    # calculate coulomb_kernel[1]
    Ω = model.unit_cell_volume  # volume of cell 

    #  = Ω/(2π)^3 ∫ 4π/q² ρ(q) dq  with  ρ(q)=e^(-αq²)
    probe_charge_integral = 8*π^2*sqrt(π/α) * Ω/(2π)^3 
    
    coulomb_kernel[1] = probe_charge_integral - probe_charge_sum
    coulomb_kernel
end

"""
    NeglectSingularity <: CoulombKernelModel

Simply set the G+q=0 Coulomb kernel component to zero.

This is the simplest approach but leads to brutally slow convergence with system size.
Useful for testing or comparison purposes.
"""
struct NeglectSingularity <: CoulombKernelModel end
function _compute_coulomb_kernel(basis::PlaneWaveBasis{T},
                                 qpt::Kpoint,
                                 q::Vec3{T},
                                 coulomb_kernel_model::NeglectSingularity) where {T}
    model = basis.model
    NG = length(qpt.G_vectors)
    coulomb_kernel = zeros(T, NG)
    for (iG, G) in enumerate(to_cpu(qpt.G_vectors))
        G_cart = model.recip_lattice * (G+q)
        Gnorm2 = sum(abs2, G_cart)
        found_singularity = (iG==1 && iszero(q))
        if !found_singularity
            coulomb_kernel[iG] = 4T(π) / Gnorm2
        end
    end
    coulomb_kernel
end


"""
    SphericallyTruncated(; Rcut=nothing) <: CoulombKernelModel

Spherical truncation of Coulomb interaction at radius Rcut.

If Rcut < 0 (default), uses Rcut = ∛(3V/(4π)) => V = 4/3π*Rcut^3
where V is the BvK cell volume.

# Reference
Phys. Rev. B 77, 193110 (2008), doi.org/10.1103/PhysRevB.77.193110
"""
struct SphericallyTruncated <: CoulombKernelModel 
    Rcut::Union{Float64, Nothing}
end
SphericallyTruncated(; Rcut=nothing) = SphericallyTruncated(Rcut)
function _compute_coulomb_kernel(basis::PlaneWaveBasis{T},
                                 qpt::Kpoint,
                                 q::Vec3{T},
                                 coulomb_kernel_model::SphericallyTruncated) where {T}
    model = basis.model
    NG = length(qpt.G_vectors)
    coulomb_kernel = zeros(T, NG)

    Rcut = @something(coulomb_kernel_model.Rcut, cbrt(basis.model.unit_cell_volume*3/4/π))
    for (iG, G) in enumerate(to_cpu(qpt.G_vectors))
        G_cart = model.recip_lattice * (G+q)
        Gnorm2 = sum(abs2, G_cart)
        found_singularity = (iG==1 && iszero(q))
        if !found_singularity
            coulomb_kernel[iG] = 4T(π) / Gnorm2 * (1-cos(Rcut*Gnorm2^0.5))
        else
            coulomb_kernel[iG] = 2T(π)*Rcut^2
        end
    end
    coulomb_kernel
end

"""
    WignerSeitzTruncated <: CoulombKernelModel

Truncate Coulomb interaction at the Wigner-Seitz cell boundary.

# Reference
Phys. Rev. B 87, 165122, 2013 (doi.org/10.1103/PhysRevB.87.165122)
"""
struct WignerSeitzTruncated <: CoulombKernelModel end
function _compute_coulomb_kernel(basis::PlaneWaveBasis{T},
                                 qpt::Kpoint,
                                 q::Vec3{T},
                                 coulomb_kernel_model::WignerSeitzTruncated) where {T}
    model = basis.model
    NG = length(qpt.G_vectors)
    coulomb_kernel = zeros(T, NG)
    # calculate inradius R_in of Wigner-Seitz cell
    # R_in is given by the largest possible 
    # R_in = (sum_i n_i * a*i) / 2 
    # with integers n_i and |R_in| <= a_min where a_min is the length of the smallest lattice vector.
    # The inequality allows us to restrict n_i by exploiting Cauchy-Schwarz, leading to 
    # |n_i| <= a_min * |b_i| / 2π 
    # where b_i are the reciprocal lattice vectors.
    L_min = minimum(norm, eachcol(model.lattice))
    n_bounds = zeros(Int, 3)
    for i in 1:3
        b_vec = model.recip_lattice[:, i]
        b_len = norm(b_vec)
        N = (L_min * b_len) / (2π)
        n_bounds[i] = ceil(Int, N)
    end
    nx, ny, nz = n_bounds # in case of a cubic cell nx=ny=nz=1
    # construct R_in
    R_in = T(Inf)
    for ix in -nx:nx, iy in -ny:ny, iz in -nz:nz # loop through all necessary integers 
        ix == 0 && iy == 0 && iz == 0 && continue
        R = model.lattice * [ix, iy, iz]
        d = norm(R) / 2 # distance from origin to perpendicular bisector plane = |R|/2
        R_in = min(R_in, d)
    end
    
    # Nyquist frequency of FFT grid
    G_Nyquist = minimum(basis.fft_size[d] / 2 * norm(model.recip_lattice[:, d]) for d in 1:3) 

    ε_target = 1e-12
    ε_min = exp(-0.5*G_Nyquist*R_in)  # required: G_Nyquist > -2*log(ε)/R_in (Appendix A.1 in paper)
    ε = max(ε_target, ε_min)
    if ε > 1e-8
        @warn "Grid too coarse for Wigner-Seitz truncation. Effective truncation error: $ε"
    end
    α = sqrt(-log(ε)) / R_in          # range separation parameter

    # FFT of long-range term erf(αr)/r restricted to Wigner-Seitz cell 
    r_vectors = DFTK.r_vectors(basis)
    V_lr_real = zeros(Complex{T}, basis.fft_size...)
    for idx in CartesianIndices(V_lr_real)
        r_frac = r_vectors[idx]
        r_centered = r_frac .- round.(r_frac) # MIC
        r_cart = model.lattice * r_centered
        d_min = norm(r_cart)
        for dx in -nx:nx, dy in -ny:ny, dz in -nz:nz # Check neighbors for non-orthorhombic cells
            dx == 0 && dy == 0 && dz == 0 && continue 
            r_shifted = r_centered - T[dx, dy, dz]
            d = norm(model.lattice * r_shifted)
            d_min = min(d_min, d)
        end
        if d_min > sqrt(eps(T))
            V_lr_real[idx] = erf(α * d_min) / d_min
        else
            V_lr_real[idx] = 2*α / sqrt(T(π))
        end
        V_lr_real[idx] *= exp(-im * 2*T(π) * dot(q, r_frac)) # add phase e^{-2πiqr}
    end
    coulomb_kernel_lr = real.(fft(basis, qpt, V_lr_real))
    coulomb_kernel_lr .*= sqrt(model.unit_cell_volume)
    
    # analytic short-range term + long-range term 
    for (iG, G) in enumerate(to_cpu(qpt.G_vectors))
        G_cart = model.recip_lattice * (G+q)
        Gnorm2 = sum(abs2, G_cart)
        found_singularity = (iG==1 && iszero(q))
        Rcut = cbrt(basis.model.unit_cell_volume*3/4/π)
        if !found_singularity
            coulomb_kernel[iG] = 4T(π) / Gnorm2 * (1 - exp(-Gnorm2/(4α^2))) + coulomb_kernel_lr[iG]
        else
            coulomb_kernel[iG] = T(π)/α^2 + coulomb_kernel_lr[iG]
        end
    end
    coulomb_kernel
end


"""
    VoxelAveraged <: CoulombKernelModel
Calculates the average of the Coulomb kernel over the Brillouin zone voxel associated
with each grid point. It is particularly well suited for highly anisotropic cells.
- **G=0 (Singularity):** Uses an exact mathematical reduction of the volume integral 
  ``∫ 1/G^2 dV`` to a smooth surface integral over the voxel faces (surface reduction).
- **G≠0 (Smooth):** Uses high-order Gaussian quadrature.
It is conceptually equivalent to the HFMEANPOT flag in VASP but uses improved integration
techniques to calcualte the average in the voxel.
# Reference
J. Chem. Phys. 160, 051101 (2024) (doi.org/10.1063/5.0182729)
"""
struct VoxelAveraged <: CoulombKernelModel end
function _compute_coulomb_kernel(basis::PlaneWaveBasis{T},
                                 qpt::Kpoint,
                                 q::Vec3{T},
                                 ::VoxelAveraged) where {T}
    model = basis.model

    # Get kgrid_size
    if isnothing(basis.kgrid)
        kgrid_size = Vec3{Int}(1, 1, 1)
    elseif basis.kgrid isa AbstractVector
        kgrid_size = Vec3{Int}(basis.kgrid)
    elseif basis.kgrid isa MonkhorstPack
        kgrid_size = Vec3{Int}(basis.kgrid.kgrid_size)
    else
        @error "Cannot determine kgrid_size for VoxelAveraged Coulomb model."
    end

    # Define Voxels as reciprocal cell deivided by k-mesh
    voxel_basis = model.recip_lattice * Diagonal(1 ./ Vec3{T}(kgrid_size))
    voxel_vol = abs(det(voxel_basis))

    # Hard coded: 12-point Gauss-Legendre nodes and weights
    nodes, weights = gauss_legendre_nodes_weights(T, 12)

    coulomb_kernel = zeros(T, length(qpt.G_vectors))

    for (iG, G) in enumerate(to_cpu(qpt.G_vectors))
        G_cart = model.recip_lattice * (G+q)

        found_singularity = (iG==1 && iszero(q))

        if found_singularity 
            # === At Singularity (G+q=0) use surface reduction method ===

            # Transforms volume integral ∫ 1/G^2 dV to surface integral Σ h * ∫ 1/G^2 dS
            integral = zero(T)
            for i in 1:3
                u_i = voxel_basis[:, i]
                u_j = voxel_basis[:, mod1(i+1, 3)]
                u_k = voxel_basis[:, mod1(i+2, 3)]

                # Height of the face from origin
                normal = cross(u_j, u_k)
                area_norm = norm(normal)

                # Calculate distance h from center to face.
                # Since face is at u_i/2, h = |(u_i/2) . n|
                h = abs(dot(u_i, normal)) / (2 * area_norm)

                # Integrate 1/G^2 over the face parallelogram
                face_integral = zero(T)
                for (wa, a) in zip(weights, nodes)
                    for (wb, b) in zip(weights, nodes)
                        # Parametrization of the face: r = u_i/2 + a*u_j + b*u_k
                        r_vec = 0.5f0 * u_i + a * u_j + b * u_k 
                        r_sq  = dot(r_vec, r_vec)
                        face_integral += wa * wb / r_sq
                    end
                end

                # Add contribution: 2 faces * h * Area * Mean(1/r^2)
                # Area factor (area_norm) comes from the Jacobian.
                integral += 2 * h * area_norm * face_integral
            end

            coulomb_kernel[iG] = 4T(π) * (integral / voxel_vol)

        else
            # === Otherwise use smooth 3D Gaussian Quadrature ===
            integral = zero(T)
            for (wx, x) in zip(weights, nodes)
                for (wy, y) in zip(weights, nodes)
                    for (wz, z) in zip(weights, nodes)
                        # q vector inside voxel
                        q_local = x * voxel_basis[:, 1] + 
                                  y * voxel_basis[:, 2] + 
                                  z * voxel_basis[:, 3]

                        G_total = G_cart + q_local
                        integral += wx * wy * wz / dot(G_total, G_total)
                    end
                end
            end
            coulomb_kernel[iG] = 4T(π) * integral
        end
    end

    return coulomb_kernel
end
# Hard coded Gauss-Legendre nodes and weights (12-point, scaled to [-0.5, 0.5])
function gauss_legendre_nodes_weights(::Type{T}, N::Int) where T
    x_std = [
        0.1252334085114689, 0.3678314989981802, 0.5873179542866683,
        0.7699026741943050, 0.9037070154876170, 0.9815606342467191
    ]
    w_std = [
        0.2491470458134029, 0.2334925365383547, 0.2031674267230659,
        0.1600783285433464, 0.1069393259953183, 0.0471753363865117
    ]
    nodes   = zeros(T, 12)
    weights = zeros(T, 12)
    for i in 1:6
        nodes[i]      = -x_std[i] / 2
        nodes[13-i]   =  x_std[i] / 2
        weights[i]    = w_std[i]  / 2
        weights[13-i] = w_std[i]  / 2
    end
    (nodes, weights)
end
