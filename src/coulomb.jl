using FastGaussQuadrature

#
# Interaction models
#

@doc raw"""
Abstract type for different interaction models

Available models:
- [`Coulomb`](@ref): 1/r
- [`ShortRangeCoulomb`](@ref): erfc(μr)/r
- [`LongRangeCoulomb`](@ref): erf(μr)/r
- [`SphericallyTruncatedCoulomb`](@ref): θ(R-r)/r
- [`WignerSeitzTruncatedCoulomb`](@ref): χ(r)/r where χ(r)=1 inside Wigner-Seitz cell, otherwise 0.

If an interaction model features a singularity, that requires some special treatment,
the following are available:
- [`ProbeCharge`](@ref): Gygi-Baldereschi probe charge method
- [`ReplaceSingularity`](@ref): Set G+q=0 component to given value (default is zero)

See also: [`compute_kernel_fourier`](@ref)
"""
abstract type InteractionKernel end
Base.Broadcast.broadcastable(k::InteractionKernel) = Ref(k)

# Each InteractionKernel should support the following functions:
#   eval_kernel_fourier(::InteractionKernel, Gsq)
#   eval_probe_charge_integral(::InteractionKernel, α)
#      Should return ∫_{BZ}  kernel(q) * e^(-α * q^2) dq
#      This is needed for the ProbeCharge regularisation. Note, that no factor 1/Γ
#      (where Γ is BZ volume) is used.
#   _compute_kernel_fourier(::InteractionKernel, basis, qpt, q)
#      The single q-point version of compute_kernel_fourier


"""
Coulomb interaction: 1/r 
"""
@kwdef struct Coulomb{R} <: InteractionKernel 
    regularization::R = ProbeCharge()
end
eval_kernel_fourier(::Coulomb, Gsq::T) where {T} = 4T(π) / Gsq
eval_probe_charge_integral(::Coulomb, α) = 8π^2 * sqrt(π / α)
_compute_kernel_fourier(k::Coulomb, basis, qpt, q) = _compute_kernel_fourier(k, k.regularization, basis, qpt, q)


"""
Short-range Coulomb interaction via error function: erfc(μr)/r
"""
struct ShortRangeCoulomb <: InteractionKernel 
    μ::Float64  # Cutoff parameter in inverse length units
end
ShortRangeCoulomb(; μ=0.2/u"Å") = ShortRangeCoulomb(austrip(μ))
ShortRangeCoulomb(μ::Quantity) = ShortRangeCoulomb(austrip(μ))
function eval_kernel_fourier(k::ShortRangeCoulomb, Gsq::T) where {T}
    -(4T(π) / Gsq) * expm1(-Gsq / (4 * T(k.μ)^2))
end
function _compute_kernel_fourier(k::ShortRangeCoulomb, basis, qpt, q)
    # Use ReplaceSingularity regularisation to explicitly set as the G==0
    # component the exact limit of the kernel for G->0, namely π/μ^2
    _compute_kernel_fourier(k, ReplaceSingularity(π/k.μ^2), basis, qpt, q)
end


"""
Long-range Coulomb interaction via error function: erf(μr)/r
"""
struct LongRangeCoulomb{R} <: InteractionKernel 
    μ::Float64  # Cutoff parameter in inverse length units
    regularization::R
end
function LongRangeCoulomb(; μ=0.2/u"Å", regularization=ProbeCharge())
    LongRangeCoulomb(austrip(μ), regularization)
end
function eval_kernel_fourier(k::LongRangeCoulomb, Gsq::T) where {T}
    (4T(π) / Gsq) * exp(-Gsq / (4 * T(k.μ)^2))
end
function eval_probe_charge_integral(k::LongRangeCoulomb, α::T) where {T}
    8T(π)^2 * sqrt(T(π) / (α + 1/(4 * T(k.μ)^2)))
end
function _compute_kernel_fourier(k::LongRangeCoulomb, basis, qpt, q)
    _compute_kernel_fourier(k, k.regularization, basis, qpt, q)
end


#
# Evaluation of interaction kernels
#

@doc raw"""
Returns the Fourier-space Coulomb kernel for momentum transfer `q`,
evaluated only on the spherical cutoff |G+q|² < 2Ecut (not the full cubic FFT grid).

In the most simple case this is essentially 4π/(G+q)².

!!! note "Gamma-point only"
    Currently only works for single k-point calculations (Gamma-only).
    For general k-points, a q-dependent basis would be needed.

## Arguments
- `basis::PlaneWaveBasis`: Plane-wave basis defining the grid
- `q`: Momentum transfer vector in fractional coordinates
- `kernel::InteractionKernel`: The physical operator defining the electron-electron interaction 

## Returns
Vector of Coulomb kernel values for each G-vector in the spherical cutoff.
"""
function compute_kernel_fourier(kernel::InteractionKernel, basis::PlaneWaveBasis{T};
                                q=zero(Vec3{T})) where {T}
    is_gamma_only = all(iszero(kpt.coordinate) for kpt in basis.kpoints)
    if !is_gamma_only
        throw(ArgumentError("Currently only Gamma-point calculations are supported in " *
                            "compute_kernel_fourier, respectively Hartree-Fock and " *
                            "calculations involving exact exchange."))
    end
    if mpi_nprocs(basis.comm_kpts) > 1
        error("MPI parallelisation not yet supported for coulomb kernel")
    end
    @assert iszero(q)

    # currently only works for Gamma-only (need correct q-point otherwise)
    qpt = basis.kpoints[1] 
    kernel_fourier =  _compute_kernel_fourier(kernel, basis, qpt, q)

    # TODO: if q=0, symmetrize Fourier coeffs to have real iFFT 

    to_device(basis.architecture, kernel_fourier)
end


"""
Spherically truncated Coulomb interaction: θ(Rcut-r)/r
If Rcut is nothing, it uses `Rcut = cbrt(3Ω / (4π))` where `Ω` is the unit cell volume.

## References
- [J. Spencer, A. Alavi. Phys. Rev. B **77**, 193110 (2008)](https://doi.org/10.1103/PhysRevB.77.193110)
"""
@kwdef struct SphericallyTruncatedCoulomb{T} <: InteractionKernel
    Rcut::T = nothing
end   
function eval_kernel_fourier(k::SphericallyTruncatedCoulomb, Gsq::T) where {T}
    4T(π) / Gsq * (1 - cos(T(k.Rcut) * sqrt(Gsq)))
end
function _compute_kernel_fourier(k::SphericallyTruncatedCoulomb, basis, qpt, q)
    # TODO: This is a bit hackish as the parameter needs to be re-computed every kernel
    #       evaluation. Cleaner would be to move this further up in the call hierarchy,
    #       such that compute_kernel_fourier is never called without Rcut being set to
    #       not nothing
    Ω = basis.model.unit_cell_volume  
    Rcut = @something k.Rcut cbrt(3Ω/(4π))
    kRcut = SphericallyTruncatedCoulomb(Rcut)

    # Use ReplaceSingularity regularisation to explicitly set as the G==0
    # component the exact limit of the kernel for G->0
    _compute_kernel_fourier(kRcut, ReplaceSingularity(2π*Rcut^2), basis, qpt, q)
end


"""
Truncate Coulomb interaction at the Wigner-Seitz cell boundary.

Computational approach: We expand 1/r = erfc(ωr)/r + erf(ωr)/r and chose ω such that the
short-range part erfc(ωr)/r is virtually unaffected by the truncation.

First the inradius R_in of the Wigner-Seitz cell is calculated.

By chosing ω = sqrt(-log(ε))/R_in we can be sure that erfc(ω*R_in) < ε.

At the same time the long-range part needs to be representable on the Fourier grid
which implies G_Nyquist >= -2 log(ε) / R_in (see Appendix A.1 in reference).
Hence we simply define ε through the given grid via ε = exp(-G_Nyquist*R_in/2).

The short-range contribution is then given by the analytical expression
4π/G^2*(1-exp(-G^2/(4ω^2)))
while the long-range contributiom is obtained through an FFT of the real-space function
erf(ωr)/r
to reciprocal space.

## Reference
- [R. Sundararaman, T. A. Arias. Phys. Rev. B **87**, 165122 (2013)](https://doi.org/10.1103/PhysRevB.87.165122)
"""
struct WignerSeitzTruncatedCoulomb <: InteractionKernel end 
@views function _compute_kernel_fourier(k::WignerSeitzTruncatedCoulomb, 
                                        basis::PlaneWaveBasis{T}, qpt, q) where {T}
    model = basis.model
    NG = length(qpt.G_vectors)
    kernel_fourier = zeros(T, NG)
    q = qpt.coordinate
    
    # === Calculate inradius R_in of Wigner-Seitz cell ===
    
    # R_in is largest possible R_in = (sum_i n_i * a*i) / 2 with integers n_i 
    # and |R_in| <= a_min where a_min is the length of the smallest lattice vector. 
    # The inequality allows to restrict n_i by exploiting Cauchy-Schwarz, leading 
    # to |n_i| <= a_min * |b_i| / 2π where b_i are reciprocal lattice vectors.

    L_min = minimum(norm, eachcol(model.lattice))
    n_bounds = zeros(Int, 3)
    for i in 1:3
        b_vec = model.recip_lattice[:, i]
        b_len = norm(b_vec)
        N = (L_min * b_len) / (2π)
        n_bounds[i] = ceil(Int, N)
    end
    nx, ny, nz = n_bounds # in case of a cubic cell nx=ny=nz=1

    # finally compute R_in
    R_in = T(Inf)
    for ix in -nx:nx, iy in -ny:ny, iz in -nz:nz 
        ix == 0 && iy == 0 && iz == 0 && continue
        R = model.lattice * [ix, iy, iz]
        d = norm(R) / 2 # distance from origin to perpendicular bisector plane = |R|/2
        R_in = min(R_in, d)
    end
    
    # Nyquist frequency of FFT grid
    G_Nyquist = minimum(basis.fft_size[d] / 2 * norm(model.recip_lattice[:, d]) for d in 1:3) 

    ε = exp(-0.5*G_Nyquist*R_in)  # required: G_Nyquist >= -2*log(ε)/R_in (Appendix A.1 in paper)
    ω = sqrt(-log(ε)) / R_in  # range separation parameter
    ε_actual = erfc(ω*R_in)
    if ε_actual > 1e-8
        @warn "Coarse grid for Wigner-Seitz truncation. Effective error: $ε_actual"
    end

    # == FFT of long-range term erf(ωr)/r restricted to Wigner-Seitz cell ===

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
            V_lr_real[idx] = erf(ω * d_min) / d_min
        else
            V_lr_real[idx] = 2*ω / sqrt(T(π))
        end
        V_lr_real[idx] *= exp(-im * 2*T(π) * dot(q, r_frac)) # add phase e^{-2πiqr}
    end
    kernel_fourier_lr = real.(fft(basis, qpt, V_lr_real))
    kernel_fourier_lr .*= sqrt(model.unit_cell_volume)
    
    # === Analytic short-range term + long-range term ===

    for (iG, G) in enumerate(to_cpu(qpt.G_vectors))
        G_cart = model.recip_lattice * (G+q)
        Gnorm2 = sum(abs2, G_cart)
        found_singularity = (iG==1 && iszero(q))
        Rcut = cbrt(basis.model.unit_cell_volume*3/4/π)
        if !found_singularity
            kernel_fourier[iG] = 4T(π) / Gnorm2 * (1 - exp(-Gnorm2/(4ω^2))) + kernel_fourier_lr[iG]
        else
            kernel_fourier[iG] = T(π)/ω^2 + kernel_fourier_lr[iG]
        end
    end
    kernel_fourier
end


"""
Probe charge Ewald method for treating the Coulomb singularity.

Regularize the G+q=0 component of the kernel by adding and subtracting the potential
generated by an array of unit Gaussian charges of width `sqrt(2α)` placed at the
grid points of the supercell with the exception to the origin itself. Here `α` should be
chosen as a localised charge that is well-representable in the chosen plane-wave basis.
We take `α = π²/Ecut` (VASP default). Convergence is `O(1/L³) = O(1 / Nk)` with `Nk`
the number of k-points.

The rationale of this method is that these artificial charges screen the Coulomb interactions
between the unit cell with the origin and the displaced unit cells of the supercell due to
the k-point sampling, such that the G+q=0 term only has contributions from these Gaussian
charges, which can be computed using an Ewald sum.

## References
- [S. Massidda, M. Posternak, A. Baldereschi. Phys. Rev. B **48**, 5058 (1993)](https://doi.org/10.1103/PhysRevB.48.5058)
"""
@kwdef struct ProbeCharge
    α::Union{Float64, Nothing} = nothing  # Width of the probe charge
end
@views function _compute_kernel_fourier(kernel, regularization::ProbeCharge,
                                        basis::PlaneWaveBasis{T}, qpt, q) where {T}
    # Default value well-tested in VASP; ensures that e^(-α*G²) is localized
    # charge with full support on G grid
    α::T = @something regularization.α   π^2/basis.Ecut

    Ω = basis.model.unit_cell_volume  # volume of unit cell 
    Gpq = map(Gpq -> sum(abs2, Gpq), Gplusk_vectors_cart(basis, qpt))

    # Note: q+G = 0 component is not special-cased, i.e. may be NaN or otherwise wrong
    kernel_fourier = eval_kernel_fourier.(kernel, Gpq)

    # Potential of Gaussian charges (skipping term at G+q=0)
    probe_charge_sum = sum((kernel_fourier .* exp.(-α*Gpq))[2:end])

    # Interaction of Gaussian charges with uniform background (i.e. integral of charges)
    # = 1/Γ ∫_{BZ} kernel(q) e^(-αq²) dq, where the integral is computed by the
    # eval_probe_charge_integral function.
    Γ = basis.model.recip_cell_volume
    probe_charge_integral = eval_probe_charge_integral(kernel, α) / Γ

    if iszero(qpt.coordinate)
        GPUArraysCore.@allowscalar begin
            kernel_fourier[1] = probe_charge_integral - probe_charge_sum
        end
    end
    kernel_fourier
end


"""
Simply set the G+q=0 Coulomb kernel component to Gpq_zero_value.
This is useful for interaction models with an analytic G+q=0 component
or for testing/comparison purposes.

For Coulomb and the case of Gpq_zero_value=0 this leads to slow `O(1/L) = O(1 / ∛(Nk))`
convergence where `L` is the size of the supercell,`Nk` is the number of k-points.
"""
struct ReplaceSingularity
    Gpq_zero_value::Float64
end
@views function _compute_kernel_fourier(kernel, regularization::ReplaceSingularity,
                                        basis::PlaneWaveBasis{T}, qpt, q) where {T}
    # Compute truncated Coulomb kernel without special-casing singularity at G+q=0 
    kernel_fourier = map(Gplusk_vectors_cart(basis, qpt)) do Gpq
        eval_kernel_fourier(kernel, sum(abs2, Gpq))
    end
    if iszero(qpt.coordinate)  # Neglect the singularity
        GPUArraysCore.@allowscalar kernel_fourier[1] = T(regularization.Gpq_zero_value)
    end
    kernel_fourier
end


"""
Calculates the average of the Coulomb kernel K(G+q) over the Brillouin zone voxel associated
with each grid point. It is particularly well suited for highly anisotropic cells.

Since the Coulomb kernel is not necessarily given by ``K(G+q)=1/(G+q)^2`` the 
following approach is used:
- G+q=0 (Singularity): Uses an exact mathematical reduction of the volume integral 
  ``∫ 1/(G+q)^2 dV`` to a smooth surface integral over the voxel faces (surface reduction).
  Then a high-order Gaussian quadrature is used to calculate ``∫ (K(G+q) - 1/(G+q)^2) dV``.
- G+q≠0 (Smooth): Uses high-order Gaussian quadrature for ``∫ K(G+q) dV``

It is conceptually equivalent to the HFMEANPOT flag in VASP but uses improved integration
techniques to calcualte the average in the voxel.

# Arguments
- `N_quadrature_points::Int`: The number of Gauss-Legendre quadrature points used per dimension. 
  Defaults to 12. For highly anisotropic cells or rigorous Thermodynamic Limit (TDL) extrapolations, 
    it is advisable to check if higher values (e.g., to 18 or 24) eliminate numerical noise.

## Reference
J. Chem. Phys. 160, 051101 (2024) (doi.org/10.1063/5.0182729)
"""
@kwdef struct VoxelAveraged
    n_quadrature_points = 12
end
@views function _compute_kernel_fourier(kernel, regularization::VoxelAveraged,
                                        basis::PlaneWaveBasis{T}, qpt, q) where {T}
    model = basis.model
    q = qpt.coordinate
    
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
    
    # get Gauss-Legendre nodes and weights
    nodes_std, weights_std = gausslegendre(regularization.n_quadrature_points)
    # Scale from [-1, 1] to [-0.5, 0.5]
    nodes = T.(nodes_std ./ 2)
    weights = T.(weights_std ./ 2)
    
    kernel_fourier = zeros(T, length(qpt.G_vectors))

    for (iG, G) in enumerate(to_cpu(qpt.G_vectors))
        G_cart = model.recip_lattice * (G+q)

        found_singularity = (iG==1 && iszero(q))
        
        if found_singularity 
            # === At Singularity (G+q=0) use surface reduction method ===
            
            # We only do that for the 4π/(G+q)^2 kernel, hence the quadrature below
            # covers the rest if kernel_fourier is different from Coulomb().
            
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
            
            kernel_fourier[iG] = 4T(π) * (integral / voxel_vol)
        end

        # === Use smooth 3D Gaussian Quadrature ===
        integral = zero(T)
        for (wx, x) in zip(weights, nodes)
            for (wy, y) in zip(weights, nodes)
                for (wz, z) in zip(weights, nodes)
                    # q vector inside voxel
                    q_local = x * voxel_basis[:, 1] + 
                              y * voxel_basis[:, 2] + 
                              z * voxel_basis[:, 3]
                    
                    G_total = G_cart + q_local
                    Gsq = dot(G_total, G_total)

                    # switch temporarily to BigFloat
                    Gsq_big = BigFloat(Gsq)
                    val_big = eval_kernel_fourier(kernel, Gsq_big)

                    # At singularity, already captured the 4π/(G+q)^2 contribution above: subtract
                    if found_singularity
                        val_big -= 4π/Gsq_big
                    end

                    # back to type T
                    val = T(val_big)

                    integral += wx * wy * wz * val
                end
            end
        end
        kernel_fourier[iG] += integral
    end
    
    kernel_fourier
end
