@doc raw"""
Abstract type for different interaction models.

### Architecture

Computing interaction kernels is split into two parts: the mathematical formula (e.g. 4\pi/G^2)
and the grid discretization. This split is primarily driven by the need to handle singularities
in long-range kernels. Note, that we already identified deficiencies with this design and a
refactoring is discussed, which will change this file's structure in the future.
# TODO (Issue #1322): Refactor file structure and kernels as discussed

1. **InteractionKernel:** Defines the pure mathematical formula (via `eval_kernel_fourier`).
2. **regularization:** Necessary for long-range kernels (like `Coulomb` and `LongRangeCoulomb`)
   diverge as ``G+q \to 0``. Evaluating them on a periodic grid requires a specific strategy
   to handle this divergence.
   
Because of this divergence, long-range `InteractionKernel`s contain a `regularization` field to
dictate how the ``G+q=0`` component is built via `_compute_kernel_fourier`. Short-range kernels
have a finite limit at `G+q \to 0``` and don't need a regularizatin.

Each InteractionKernel should support the following functions:
eval_kernel_fourier(::InteractionKernel, Gsq)
eval_probe_charge_integral(::InteractionKernel, α)
    Should return ∫_{BZ}  kernel(q) * e^(-α * q^2) dq
    This is needed for the ProbeCharge regularisation. Note, that no factor 1/Γ
    where Γ is BZ volume) is used.
_compute_kernel_fourier(::InteractionKernel, basis, qpt)
    The single q-point version of compute_kernel_fourier. Returns the kernel at G+q
    for all G of the full FFT cube `G_vectors(basis)` as an array of shape
    `basis.fft_size` (linear index 1 is G=0). Only `qpt.coordinate` is used.

### Available models:
- [`Coulomb`](@ref): 1/r
- [`ShortRangeCoulomb`](@ref): erfc(μr)/r
- [`LongRangeCoulomb`](@ref): erf(μr)/r
- [`SphericallyTruncatedCoulomb`](@ref): θ(R-r)/r
- [`WignerSeitzTruncatedCoulomb`](@ref): χ(r)/r (1 inside Wigner-Seitz cell, 0 otherwise)

### Available singularity corrections (regularizations):
- [`ProbeCharge`](@ref): Gygi-Baldereschi probe charge method
- [`ReplaceSingularity`](@ref): Set the G+q=0 component to a specific value
- [`VoxelAveraged`](@ref): Average the continuous kernel over the Brillouin zone voxel

See also: [`compute_kernel_fourier`](@ref)
"""
abstract type InteractionKernel end
Base.Broadcast.broadcastable(k::InteractionKernel) = Ref(k)

# |G+q|² for all G of the full FFT cube G_vectors(basis)
function _norm2_Gplusq_on_cube(basis::PlaneWaveBasis, qpt::Kpoint)
    recip_lattice = basis.model.recip_lattice  # hoist: avoid closure over basis / qpt
    q = qpt.coordinate
    map(G -> norm2(recip_lattice * (G + q)), G_vectors(basis))
end

# Evaluate the kernel at G+q for all G of the full FFT cube. The G+q=0 component is not
# special-cased (i.e. may be Inf or NaN) and is left to the regularization.
function _eval_kernel_on_cube(kernel, basis::PlaneWaveBasis, qpt::Kpoint)
    eval_kernel_fourier.(kernel, _norm2_Gplusq_on_cube(basis, qpt))
end

# TODO: should we have a eval_kernel_real?
# TODO: rename "k" in _compute_kernel_fourier(k...
# TODO: change notation: p instead of G, G+q, ...
# TODO: introduce a clever and AD-friendly way to deal with f(x)/x for x->0.
#       E.g. intoduce phi(x) = iszero(x) ? one(x) : expm1(x) / x 


"""
Coulomb interaction: 1/r 
"""
@kwdef struct Coulomb{R} <: InteractionKernel 
    regularization::R = ProbeCharge()
end
eval_kernel_fourier(::Coulomb, Gsq::T) where {T} = 4T(π) / Gsq
eval_probe_charge_integral(::Coulomb, α) = 8π^2 * sqrt(π / α)
_compute_kernel_fourier(k::Coulomb, basis, qpt) = _compute_kernel_fourier(k, k.regularization, basis, qpt)


"""
Short-range Coulomb interaction via error function: erfc(μr)/r
"""
struct ShortRangeCoulomb{T <: Real} <: InteractionKernel 
    μ::T  # Cutoff parameter in inverse length units
end
ShortRangeCoulomb(; μ=0.2/u"Å") = ShortRangeCoulomb(austrip(μ))
ShortRangeCoulomb(μ::Quantity) = ShortRangeCoulomb(austrip(μ))
function eval_kernel_fourier(k::ShortRangeCoulomb, Gsq::T) where {T}
    -(4T(π) / Gsq) * expm1(-Gsq / (4 * T(k.μ)^2))
end
function _compute_kernel_fourier(k::ShortRangeCoulomb, basis, qpt)
    # Use ReplaceSingularity regularisation to explicitly set as the G==0
    # component the exact limit of the kernel for G->0, namely π/μ^2
    _compute_kernel_fourier(k, ReplaceSingularity(π/k.μ^2), basis, qpt)
end


"""
Long-range Coulomb interaction via error function: erf(μr)/r
"""
struct LongRangeCoulomb{T <: Real, R} <: InteractionKernel 
    μ::T  # Cutoff parameter in inverse length units
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
function _compute_kernel_fourier(k::LongRangeCoulomb, basis, qpt)
    _compute_kernel_fourier(k, k.regularization, basis, qpt)
end


#
# Evaluation of interaction kernels
#

@doc raw"""
Returns the Fourier-space Coulomb kernel for momentum transfer `q`, evaluated at all
`G+q` with `G` on the full cubic FFT grid `G_vectors(basis)`.

In the most simple case this is essentially 4π/(G+q)².

## Arguments
- `kernel::InteractionKernel`: The physical operator defining the electron-electron interaction
- `basis::PlaneWaveBasis`: Plane-wave basis defining the grid
- `qpt::Kpoint`: Momentum transfer; only `qpt.coordinate` (fractional coordinates) is used

## Returns
Array of shape `basis.fft_size` with the kernel value for each `G` of `G_vectors(basis)`.
"""
function compute_kernel_fourier(kernel::InteractionKernel, basis::PlaneWaveBasis{T},
                                qpt::Kpoint) where {T}
    if mpi_nprocs(basis.comm_kpts) > 1
        error("MPI parallelisation not yet supported for coulomb kernel")
    end

    kernel_fourier = _compute_kernel_fourier(kernel, basis, qpt)

    # TODO: if q=0, symmetrize Fourier coeffs to have real iFFT 

    to_device(basis.architecture, kernel_fourier)
end


"""
Spherically truncated Coulomb interaction: θ(Rcut-r)/r
If Rcut is nothing, it uses `Rcut = cbrt(3 Nk Ω / (4π))`, i.e. the radius of the sphere
with the volume of the supercell corresponding to the k-point grid, where `Ω` is the
unit cell volume and `Nk` the number of (reducible) k-points.

## References
- [J. Spencer, A. Alavi. Phys. Rev. B **77**, 193110 (2008)](https://doi.org/10.1103/PhysRevB.77.193110)
"""
@kwdef struct SphericallyTruncatedCoulomb{T} <: InteractionKernel
    Rcut::T = nothing
end
function eval_kernel_fourier(k::SphericallyTruncatedCoulomb, Gsq::T) where {T}
    4T(π) / Gsq * (1 - cos(T(k.Rcut) * sqrt(Gsq)))
end
function _compute_kernel_fourier(k::SphericallyTruncatedCoulomb, basis, qpt)
    # TODO: This is a bit hackish as the parameter needs
    #       to be re-computed every kernel evaluation.
    Ω_supercell = basis.model.unit_cell_volume * length(basis.kgrid)  # Nk = length(kgrid)
    Rcut = @something k.Rcut cbrt(3Ω_supercell/(4π))
    kRcut = SphericallyTruncatedCoulomb(Rcut)

    # Use ReplaceSingularity regularisation to explicitly set as the G==0
    # component the exact limit of the kernel for G->0
    _compute_kernel_fourier(kRcut, ReplaceSingularity(2π*Rcut^2), basis, qpt)
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

# TODO: Evaluating erf(ωr)/r on a discrete real-space grid and performing 
# an FFT introduces aliasing errors, as the function is not strictly band-limited.
# For details on this discretization error, see Appendix A.1 of the Reference below.

## Reference
- [R. Sundararaman, T. A. Arias. Phys. Rev. B **87**, 165122 (2013)](https://doi.org/10.1103/PhysRevB.87.165122)
"""
struct WignerSeitzTruncatedCoulomb <: InteractionKernel end
@views function _compute_kernel_fourier(k::WignerSeitzTruncatedCoulomb,
                                        basis::PlaneWaveBasis{T}, qpt) where {T}
    model = basis.model
    q = qpt.coordinate
    kernel_fourier = zeros(T, basis.fft_size)

    # TODO: With k-points the truncation region has to be the Wigner-Seitz cell of the
    #       supercell corresponding to the k-point grid (not of the unit cell) and the
    #       long-range part below needs to be transformed on the supercell grid
    #       fft_size .* kgrid_size, which yields all G+q at once.
    #       Until this is implemented only Γ-only is supported.
    if length(basis.kgrid) > 1
        error("WignerSeitzTruncatedCoulomb currently only supports Γ-only calculations " *
              "(kgrid=(1, 1, 1)). Use SphericallyTruncatedCoulomb or a ProbeCharge / " *
              "VoxelAveraged regularization with k-points.")
    end

    # === Calculate inradius R_in of Wigner-Seitz cell ===
    
    # R_in is largest possible R_in = (sum_i n_i * a*i) / 2 with integers n_i 
    # and |R_in| <= a_min where a_min is the length of the smallest lattice vector. 
    # The inequality allows to restrict n_i by exploiting Cauchy-Schwarz, leading 
    # to |n_i| <= a_min * |b_i| / 2π where b_i are reciprocal lattice vectors.

    L_min = minimum(norm, eachcol(model.lattice))
    nx, ny, nz = estimate_integer_lattice_bounds(model.lattice, L_min)

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

        # Map point to the [-0.5, 0.5)³ fractional cell (Minimum Image Convention)
        r_centered = r_frac .- round.(r_frac) 
        r_cart = model.lattice * r_centered
        d_min = norm(r_cart)

        # For non-orthorhombic cells, the fractional Minimum Image Convention
        # does NOT guarantee the shortest Cartesian distance. A point in a
        # neighboring fractional cell could be physically closer. We exhaustively
        # check all nearby cells (bounds nx,ny,nz) to find the absolute minimum.
        for dx in -nx:nx, dy in -ny:ny, dz in -nz:nz 
            dx == 0 && dy == 0 && dz == 0 && continue 
            r_shifted = r_centered - T[dx, dy, dz]
            d = norm(model.lattice * r_shifted)
            d_min = min(d_min, d)
        end

        # Evaluate erf(ωr)/r 
        if d_min > sqrt(eps(T))
            V_lr_real[idx] = erf(ω * d_min) / d_min
        else
            V_lr_real[idx] = 2*ω / sqrt(T(π))
        end

        # Add the Bloch phase factor for q-point evaluation
        V_lr_real[idx] *= exp(-im * 2*T(π) * dot(q, r_frac)) # add phase e^{-2πiqr}
    end
    # Scale such that kernel_fourier_lr[G] ≈ ∫_Ω erf(ω|r|)/|r| e^{-i(G+q)r} dr
    kernel_fourier_lr = real.(fft(basis, V_lr_real))
    kernel_fourier_lr .*= sqrt(model.unit_cell_volume)

    # === Analytic short-range term + long-range term ===

    for (iG, G) in enumerate(to_cpu(G_vectors(basis)))
        Gnorm2 = norm2(model.recip_lattice * (G + q))
        if iG == 1 && iszero(q)  # G+q = 0: use the analytic limit of the short-range term
            kernel_fourier[iG] = T(π)/ω^2 + kernel_fourier_lr[iG]
        else
            kernel_fourier[iG] = 4T(π) / Gnorm2 * (1 - exp(-Gnorm2/(4ω^2))) + kernel_fourier_lr[iG]
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
                                        basis::PlaneWaveBasis{T}, qpt) where {T}
    # Default value well-tested in VASP; ensures that e^(-α*G²) is localized
    # charge with full support on G grid
    α::T = @something regularization.α   π^2/basis.Ecut

    kernel_fourier = _eval_kernel_on_cube(kernel, basis, qpt)

    if iszero(qpt.coordinate)
        # Interaction of Gaussian charges with uniform background (i.e. integral of charges)
        # = 1/Γ ∫_{BZ} kernel(q) e^(-αq²) dq, where the integral is computed by the
        # eval_probe_charge_integral function.
        Γ = basis.model.recip_cell_volume
        Nk = length(basis.kgrid)  # number of (reducible) k-points
        probe_charge_integral = eval_probe_charge_integral(kernel, α) * Nk / Γ

        # Potential of the Gaussian charges: sum over all G+Q with Q in the k-point grid,
        # i.e. over all momentum transfers Q = k - k', except G+Q=0.
        # Note: build_qpoints derives the Q-points from basis.kpoints. The uniform
        # weight below rescales the sum to Nk terms, which is exact if basis.kpoints
        # is the full (non-symmetry-reduced) k-grid.
        Q_points, _ = build_qpoints(basis)
        probe_charge_sum = sum(Q_points) do Qpt
            weight = Nk / length(Q_points)
            Gsq_Q = _norm2_Gplusq_on_cube(basis, Qpt)
            summands = eval_kernel_fourier.(kernel, Gsq_Q) .* exp.(-α .* Gsq_Q)
            if iszero(Qpt.coordinate)  # skip the singular G+Q=0 term
                GPUArraysCore.@allowscalar summands[1] = zero(T)
            end
            weight * sum(summands)
        end
        #probe_charge_sum = mpi_sum(probe_charge_sum, basis.comm_kpts)

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
struct ReplaceSingularity{T <: Real}
    Gpq_zero_value::T
end
@views function _compute_kernel_fourier(kernel, regularization::ReplaceSingularity,
                                        basis::PlaneWaveBasis{T}, qpt) where {T}
    kernel_fourier = _eval_kernel_on_cube(kernel, basis, qpt)
    if iszero(qpt.coordinate)  # Replace the singular G+q=0 component
        GPUArraysCore.@allowscalar kernel_fourier[1] = T(regularization.Gpq_zero_value)
    end
    kernel_fourier
end


"""
Calculates the average of the Coulomb kernel K(G+q) over the Brillouin zone voxel associated
with each grid point. It is particularly well suited for highly anisotropic cells.
Note that this kernel evaluation strategy only becomes available once the
`FastGaussQuadrature` module is explicitly loaded.

Since the Coulomb kernel is not necessarily given by ``K(G+q)=1/(G+q)^2`` the 
following approach is used:
- G+q=0 (Singularity): Uses an exact mathematical reduction of the volume integral 
  ``∫ 1/(G+q)^2 dV`` to a smooth surface integral over the voxel faces (surface reduction).
  Then a high-order Gaussian quadrature is used to calculate ``∫ (K(G+q) - 1/(G+q)^2) dV``.
- G+q≠0 (Smooth): Uses high-order Gaussian quadrature for ``∫ K(G+q) dV``

It is conceptually equivalent to the HFMEANPOT flag in VASP but uses improved integration
techniques to calcualte the average in the voxel.

## Arguments
- `N_quadrature_points::Int`: The number of Gauss-Legendre quadrature points used per dimension. 
  Defaults to 12. For highly anisotropic cells or rigorous Thermodynamic Limit (TDL) extrapolations, 
    it is advisable to check if higher values (e.g., to 18 or 24) eliminate numerical noise.

## Reference
J. Chem. Phys. 160, 051101 (2024) (doi.org/10.1063/5.0182729)
"""
@kwdef struct VoxelAveraged
    n_quadrature_points = 12
end

# For the implementation see DFTKFastGaussQuadratureExt.jl
