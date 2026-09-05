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
_compute_kernel_fourier(::InteractionKernel, basis, qpt::Kpoint)
    The single q-point version of compute_kernel_fourier. Returns the kernel at G+q
    for all G of the full FFT cube `G_vectors(basis)` as an array of shape
    `basis.fft_size` (linear index 1 is G=0). Only `qpt.coordinate` is used.
_compute_kernel_fourier(::InteractionKernel, basis, q_points::AbstractVector{<:Kpoint})
    Optional: The kernels for several momentum transfers at once (as a vector). Defaults
    to calling the single q-point version for each q-point. Kernels which are cheaper
    to compute for all q-points together (e.g. `WignerSeitzTruncatedCoulomb`) specialise it.

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

# Size of the k-point grid, i.e. of the supercell corresponding to the k-point sampling,
# inferred from the grid of momentum transfers q = k - k'. In contrast to the k-points
# themselves the momentum transfers always form a Γ-centred grid, such that this also
# works for shifted and for explicitly given k-point grids.
function _kgrid_size(basis::PlaneWaveBasis; tol=1e-8)
    q_points, _ = build_qpoints(basis)
    kgrid_size = map(1:3) do i
        q_i = sort([qpt.coordinate[i] for qpt in q_points])
        1 + count(>(tol), diff(q_i))
    end
    if prod(kgrid_size) != length(q_points)
        error("The k-point grid is not a regular (Monkhorst-Pack-like) grid.")
    end
    Vec3{Int}(kgrid_size)
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
_compute_kernel_fourier(k::Coulomb, basis, qpt::Kpoint) = _compute_kernel_fourier(k, k.regularization, basis, qpt)


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
function _compute_kernel_fourier(k::ShortRangeCoulomb, basis, qpt::Kpoint)
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
function _compute_kernel_fourier(k::LongRangeCoulomb, basis, qpt::Kpoint)
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
The kernels for several momentum transfers `q_points` at once, returned as a vector
with one entry per q-point. See [`compute_kernel_fourier`](@ref).
"""
function compute_kernel_fourier(kernel::InteractionKernel, basis::PlaneWaveBasis,
                                q_points::AbstractVector{<:Kpoint})
    if mpi_nprocs(basis.comm_kpts) > 1
        error("MPI parallelisation not yet supported for coulomb kernel")
    end
    kernels_fourier = _compute_kernel_fourier(kernel, basis, q_points)
    [to_device(basis.architecture, kernel_fourier) for kernel_fourier in kernels_fourier]
end
function _compute_kernel_fourier(kernel::InteractionKernel, basis::PlaneWaveBasis,
                                 q_points::AbstractVector{<:Kpoint})
    [_compute_kernel_fourier(kernel, basis, qpt) for qpt in q_points]
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
function _compute_kernel_fourier(k::SphericallyTruncatedCoulomb, basis, qpt::Kpoint)
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
Coulomb interaction truncated at the boundary of the Wigner-Seitz cell of the supercell
corresponding to the k-point grid (for Γ-only calculations this is the unit cell).

Computational approach: We expand 1/r = erfc(ωr)/r + erf(ωr)/r and chose ω such that the
short-range part erfc(ωr)/r is virtually unaffected by the truncation.

First the inradius R_in of the Wigner-Seitz cell is calculated.

By chosing ω = sqrt(-log(ε))/R_in we can be sure that erfc(ω*R_in) < ε.

At the same time the long-range part needs to be representable on the Fourier grid
which implies G_Nyquist >= -2 log(ε) / R_in (see Appendix A.1 in reference).
Hence we simply define ε through the given grid via ε = exp(-G_Nyquist*R_in/2).

The short-range contribution is then given by the analytical expression
4π/G^2*(1-exp(-G^2/(4ω^2)))
while the long-range contribution is obtained through an FFT of the real-space function
erf(ωr)/r, truncated to the Wigner-Seitz cell, on the real-space grid of the supercell.
Since this function is periodic on the supercell, one FFT (including the Bloch phase
``e^{-iqr}`` of the momentum transfer ``q``) yields its Fourier coefficients at all ``G+q``.

# TODO: Evaluating erf(ωr)/r on a discrete real-space grid and performing
# an FFT introduces aliasing errors, as the function is not strictly band-limited.
# For details on this discretization error, see Appendix A.1 of the Reference below.

## Reference
- [R. Sundararaman, T. A. Arias. Phys. Rev. B **87**, 165122 (2013)](https://doi.org/10.1103/PhysRevB.87.165122)
"""
struct WignerSeitzTruncatedCoulomb <: InteractionKernel end
function _compute_kernel_fourier(k::WignerSeitzTruncatedCoulomb, basis, qpt::Kpoint)
    only(_compute_kernel_fourier(k, basis, [qpt]))
end
function _compute_kernel_fourier(::WignerSeitzTruncatedCoulomb, basis::PlaneWaveBasis{T},
                                 q_points::AbstractVector{<:Kpoint}) where {T}
    model = basis.model

    # The truncated kernel is periodic on the supercell corresponding to the k-point grid
    kgrid_size = _kgrid_size(basis)
    lattice_supercell  = model.lattice .* kgrid_size'  # scale the i-th lattice vector by kgrid_size[i]
    fft_size_supercell = Tuple(basis.fft_size .* kgrid_size)
    Ω_supercell = model.unit_cell_volume * prod(kgrid_size)

    # === Calculate inradius R_in of the Wigner-Seitz cell of the supercell ===

    # R_in is largest possible R_in = (sum_i n_i * a*i) / 2 with integers n_i
    # and |R_in| <= a_min where a_min is the length of the smallest lattice vector.
    # The inequality allows to restrict n_i by exploiting Cauchy-Schwarz, leading
    # to |n_i| <= a_min * |b_i| / 2π where b_i are reciprocal lattice vectors.
    L_min = minimum(norm, eachcol(lattice_supercell))
    nx, ny, nz = estimate_integer_lattice_bounds(lattice_supercell, L_min)
    R_in = T(Inf)
    for ix in -nx:nx, iy in -ny:ny, iz in -nz:nz
        ix == 0 && iy == 0 && iz == 0 && continue
        # distance from origin to perpendicular bisector plane = |R|/2
        R_in = min(R_in, norm(lattice_supercell * Vec3(ix, iy, iz)) / 2)
    end

    # === Range separation parameter ω ===

    # Nyquist frequency of the FFT grid (the same for the unit cell and the supercell grid)
    G_Nyquist = minimum(basis.fft_size[d] / 2 * norm(model.recip_lattice[:, d]) for d in 1:3)
    ε = exp(-0.5*G_Nyquist*R_in)  # required: G_Nyquist >= -2*log(ε)/R_in (Appendix A.1 in paper)
    ω = sqrt(-log(ε)) / R_in
    ε_actual = erfc(ω*R_in)
    if ε_actual > 1e-8
        @warn "Coarse grid for Wigner-Seitz truncation. Effective error: $ε_actual"
    end

    # === Long-range term erf(ωr)/r truncated to the Wigner-Seitz cell on the supercell grid ===

    fft_grid_supercell = FFTGrid(fft_size_supercell, Ω_supercell, CPU())
    V_lr_real = map(r_vectors(fft_grid_supercell)) do r_frac  # supercell fractional coordinates
        # Distance to the closest lattice point of the supercell, i.e. |r| inside the
        # Wigner-Seitz cell. Mapping to [-0.5, 0.5)³ (minimum image convention) does not
        # guarantee the closest image for non-orthorhombic cells, so the neighbouring
        # cells (within the bounds nx, ny, nz) are checked as well.
        r_centered = r_frac - round.(r_frac)
        d_min = norm(lattice_supercell * r_centered)
        for dx in -nx:nx, dy in -ny:ny, dz in -nz:nz
            d_min = min(d_min, norm(lattice_supercell * (r_centered - Vec3(dx, dy, dz))))
        end
        d_min > sqrt(eps(T)) ? erf(ω * d_min) / d_min : 2ω / sqrt(T(π))
    end

    # === Analytic short-range term + long-range term for each q ===

    map(q_points) do qpt
        q = qpt.coordinate

        # Fourier coefficients ∫ V_lr(r) e^{-i(G+q)r} dr over the supercell for all G of the
        # unit cell grid: Bloch phase e^{-iqr} (r in unit cell fractional coordinates is
        # kgrid_size .* r_frac) and one FFT on the supercell grid, on which the unit cell
        # vector G is the supercell reciprocal lattice vector kgrid_size .* G.
        phase = map(r_frac -> cis2pi(-dot(q, kgrid_size .* r_frac)), r_vectors(fft_grid_supercell))
        V_lr_fourier = real.(fft(fft_grid_supercell, V_lr_real .* phase))
        V_lr_fourier .*= sqrt(Ω_supercell)

        kernel_fourier = zeros(T, basis.fft_size)
        for (iG, G) in enumerate(to_cpu(G_vectors(basis)))
            Gnorm2 = norm2(model.recip_lattice * (G + q))
            idx_supercell = CartesianIndex(Tuple(mod.(kgrid_size .* G, fft_size_supercell) .+ 1))
            if iG == 1 && iszero(q)  # G+q = 0: use the analytic limit of the short-range term
                kernel_fourier[iG] = T(π)/ω^2 + V_lr_fourier[idx_supercell]
            else
                kernel_fourier[iG] = (4T(π) / Gnorm2 * (1 - exp(-Gnorm2/(4ω^2)))
                                      + V_lr_fourier[idx_supercell])
            end
        end
        kernel_fourier
    end
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
