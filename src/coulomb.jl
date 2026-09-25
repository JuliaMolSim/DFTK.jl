@doc raw"""
Abstract type for electron-electron interaction kernels (Coulomb and Coulomb-like), used e.g.
by the [`ExactExchange`](@ref) term.

### Interface

A kernel is a function ``\hat{k}(p)`` of the Fourier vector ``p``. The two main methods are
```julia
eval_kernel_fourier(kernel, p)          # k̂(p) at one Cartesian Fourier vector p
eval_kernel_fourier(kernel, basis, q)   # k̂(G+q) for all G of the FFT cube G_vectors(basis)
```
where the momentum transfer `q` is a `Vec3` in reduced coordinates. The grid method returns
an array of shape `basis.fft_size` (linear index 1 is `G=0`) on the device of the basis. It falls
back to evaluating the pointwise method on all `G+q`, so kernels with a closed-form
``\hat{k}(p)`` only need to implement the pointwise method. Kernels which are only defined
relative to a grid or to the k-point sampling (e.g. [`WignerSeitzTruncatedCoulomb`](@ref),
[`ProbeCharge`](@ref)) override the grid method instead.

For several momentum transfers at once
```julia
eval_kernel_fourier(kernel, basis, q_points)   # one array per q in q_points
```
calls the single-q method for each `q`. Kernels which are cheaper to compute for all
q-points together (e.g. [`WignerSeitzTruncatedCoulomb`](@ref)) specialise it.

Long-range kernels such as [`BareCoulomb`](@ref) and [`LongRangeCoulomb`](@ref) diverge as
``p \to 0``. On a discrete k-point grid their ``G+q=0`` component needs a **singularity
treatment**, which is a kernel wrapping the divergent `inner_kernel`, e.g.
`ProbeCharge(BareCoulomb())`. Kernels with a finite ``p \to 0`` limit (e.g.
[`ShortRangeCoulomb`](@ref)) return this limit at ``p=0`` themselves and need no wrapper.

Kernels which can be treated by [`ProbeCharge`](@ref) further implement
```julia
compute_probe_charge_integral(kernel, α)   # ∫_{BZ} k̂(q) e^{-α q²} dq  (no 1/|BZ| factor)
```

### Available kernels
- [`BareCoulomb`](@ref): ``1/r``
- [`ShortRangeCoulomb`](@ref): ``\mathrm{erfc}(μr)/r``
- [`LongRangeCoulomb`](@ref): ``\mathrm{erf}(μr)/r``
- [`SphericallyTruncatedCoulomb`](@ref): ``θ(R-r)/r``
- [`WignerSeitzTruncatedCoulomb`](@ref): ``χ(r)/r`` (1 inside Wigner-Seitz cell, 0 otherwise)

### Available singularity treatments (wrapping a divergent kernel)
- [`ProbeCharge`](@ref): Gygi-Baldereschi probe charge method
- [`ReplaceSingularity`](@ref): Set the ``G+q=0`` component to a specific value
- [`VoxelAverage`](@ref): Average the kernel over the Brillouin zone voxel
"""
abstract type InteractionKernel end
Base.Broadcast.broadcastable(k::InteractionKernel) = Ref(k)

function eval_kernel_fourier(kernel::InteractionKernel, p::AbstractVector)
    error("$(nameof(typeof(kernel))) has no pointwise Fourier representation. " *
          "Evaluate it on a grid via eval_kernel_fourier(kernel, basis, q).")
end
function eval_kernel_fourier(kernel::InteractionKernel, basis::PlaneWaveBasis, q::Vec3)
    recip_lattice = basis.model.recip_lattice  # hoist: avoid closure over basis
    map(G -> eval_kernel_fourier(kernel, recip_lattice * (G + q)), G_vectors(basis))
end
function eval_kernel_fourier(kernel::InteractionKernel, basis::PlaneWaveBasis,
                             q_points::AbstractVector{<:Vec3})
    [eval_kernel_fourier(kernel, basis, q) for q in q_points]
end

function compute_probe_charge_integral(kernel::InteractionKernel, α)
    error("$(nameof(typeof(kernel))) does not support the ProbeCharge singularity " *
          "treatment (compute_probe_charge_integral not implemented).")
end

# |G+q|² for all G of the full FFT cube G_vectors(basis)
function _norm2_Gplusq_on_cube(basis::PlaneWaveBasis, q::Vec3)
    recip_lattice = basis.model.recip_lattice  # hoist: avoid closure over basis
    map(G -> norm2(recip_lattice * (G + q)), G_vectors(basis))
end

# Size of the k-point grid, i.e. of the supercell corresponding to the k-point sampling,
# inferred from the grid of momentum transfers q = k - k'. In contrast to the k-points
# themselves the momentum transfers always form a Γ-centred grid, such that this also
# works for shifted and for explicitly given k-point grids.
function _kgrid_size(basis::PlaneWaveBasis; tol=1e-8)
    q_points = build_qpoints(basis)
    kgrid_size = map(1:3) do i
        q_i = sort([q[i] for q in q_points])
        1 + count(>(tol), diff(q_i))
    end
    if prod(kgrid_size) != length(q_points)
        error("The k-point grid is not a regular (Monkhorst-Pack-like) grid.")
    end
    Vec3{Int}(kgrid_size)
end


"""
Bare Coulomb interaction: 1/r

Diverges at ``p=0`` and thus needs to be wrapped in a singularity treatment, e.g.
`ProbeCharge(BareCoulomb())`, see [`InteractionKernel`](@ref).
"""
struct BareCoulomb <: InteractionKernel end
eval_kernel_fourier(::BareCoulomb, p::AbstractVector{T}) where {T} = 4T(π) / norm2(p)
compute_probe_charge_integral(::BareCoulomb, α) = 8π^2 * sqrt(π / α)


"""
Short-range Coulomb interaction via error function: erfc(μr)/r
"""
struct ShortRangeCoulomb{T <: Real} <: InteractionKernel
    μ::T  # Cutoff parameter in inverse length units
end
ShortRangeCoulomb(; μ=0.2/u"Å") = ShortRangeCoulomb(austrip(μ))
ShortRangeCoulomb(μ::Quantity) = ShortRangeCoulomb(austrip(μ))
function eval_kernel_fourier(k::ShortRangeCoulomb, p::AbstractVector{T}) where {T}
    p2 = norm2(p)
    iszero(p2) && return T(π) / T(k.μ)^2  # limit of the expression below for p → 0
    -(4T(π) / p2) * expm1(-p2 / (4 * T(k.μ)^2))
end


"""
Long-range Coulomb interaction via error function: erf(μr)/r

Diverges at ``p=0`` and thus needs to be wrapped in a singularity treatment, e.g.
`ProbeCharge(LongRangeCoulomb())`, see [`InteractionKernel`](@ref).
"""
struct LongRangeCoulomb{T <: Real} <: InteractionKernel
    μ::T  # Cutoff parameter in inverse length units
end
LongRangeCoulomb(; μ=0.2/u"Å") = LongRangeCoulomb(austrip(μ))
LongRangeCoulomb(μ::Quantity) = LongRangeCoulomb(austrip(μ))
function eval_kernel_fourier(k::LongRangeCoulomb, p::AbstractVector{T}) where {T}
    p2 = norm2(p)
    (4T(π) / p2) * exp(-p2 / (4 * T(k.μ)^2))
end
function compute_probe_charge_integral(k::LongRangeCoulomb, α::T) where {T}
    8T(π)^2 * sqrt(T(π) / (α + 1/(4 * T(k.μ)^2)))
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
function eval_kernel_fourier(k::SphericallyTruncatedCoulomb{<:Real},
                             p::AbstractVector{T}) where {T}
    p2 = norm2(p)
    Rcut = T(k.Rcut)
    iszero(p2) && return 2T(π) * Rcut^2  # limit of the expression below for p → 0
    # 4π/p² (1 - cos(Rcut |p|)), using 1 - cos(x) = 2sin²(x/2) to avoid cancellation
    8T(π) / p2 * sin(Rcut * sqrt(p2) / 2)^2
end
function eval_kernel_fourier(::SphericallyTruncatedCoulomb{Nothing},
                             basis::PlaneWaveBasis, q::Vec3)
    Ω_supercell = basis.model.unit_cell_volume * length(basis.kgrid)  # Nk = length(kgrid)
    Rcut = cbrt(3Ω_supercell / (4π))
    eval_kernel_fourier(SphericallyTruncatedCoulomb(Rcut), basis, q)
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

The short-range contribution is then given analytically by [`ShortRangeCoulomb`](@ref)`(ω)`,
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
# TODO: A pointwise eval_kernel_fourier(::WignerSeitzTruncatedCoulomb, p) could be obtained
#       by a slow Fourier transform (explicit sum over the real-space grid).
function eval_kernel_fourier(kernel::WignerSeitzTruncatedCoulomb, basis::PlaneWaveBasis,
                             q::Vec3)
    only(eval_kernel_fourier(kernel, basis, [q]))
end
function eval_kernel_fourier(::WignerSeitzTruncatedCoulomb, basis::PlaneWaveBasis{T},
                             q_points::AbstractVector{<:Vec3}) where {T}
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

    short_range = ShortRangeCoulomb(ω)
    recip_lattice = model.recip_lattice
    map(q_points) do q
        # Fourier coefficients ∫ V_lr(r) e^{-i(G+q)r} dr over the supercell for all G of the
        # unit cell grid: Bloch phase e^{-iqr} (r in unit cell fractional coordinates is
        # kgrid_size .* r_frac) and one FFT on the supercell grid, on which the unit cell
        # vector G is the supercell reciprocal lattice vector kgrid_size .* G.
        phase = map(r_frac -> cis2pi(-dot(q, kgrid_size .* r_frac)), r_vectors(fft_grid_supercell))
        V_lr_fourier = real.(fft(fft_grid_supercell, V_lr_real .* phase))
        V_lr_fourier .*= sqrt(Ω_supercell)

        kernel_fourier = map(to_cpu(G_vectors(basis))) do G
            idx_supercell = CartesianIndex(Tuple(mod.(kgrid_size .* G, fft_size_supercell) .+ 1))
            eval_kernel_fourier(short_range, recip_lattice * (G + q)) + V_lr_fourier[idx_supercell]
        end
        to_device(basis.architecture, kernel_fourier)
    end
end


"""
Probe charge Ewald method for treating the Coulomb singularity of `inner_kernel`.

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

Only usable with kernels implementing `compute_probe_charge_integral`, i.e.
[`BareCoulomb`](@ref) and [`LongRangeCoulomb`](@ref).

## References
- [S. Massidda, M. Posternak, A. Baldereschi. Phys. Rev. B **48**, 5058 (1993)](https://doi.org/10.1103/PhysRevB.48.5058)
"""
struct ProbeCharge{K <: InteractionKernel} <: InteractionKernel
    inner_kernel::K
    α::Union{Float64, Nothing}  # Width of the probe charge
end
ProbeCharge(inner_kernel=BareCoulomb(); α=nothing) = ProbeCharge(inner_kernel, α)
function eval_kernel_fourier(k::ProbeCharge, basis::PlaneWaveBasis{T}, q::Vec3) where {T}
    # Default value well-tested in VASP; ensures that e^(-α*G²) is localized
    # charge with full support on G grid
    α::T = @something k.α π^2/basis.Ecut

    kernel_fourier = eval_kernel_fourier(k.inner_kernel, basis, q)

    if iszero(q)
        # Interaction of Gaussian charges with uniform background (i.e. integral of charges)
        # = 1/Γ ∫_{BZ} kernel(q) e^(-αq²) dq, where the integral is computed by the
        # compute_probe_charge_integral function.
        Γ = basis.model.recip_cell_volume
        Nk = length(basis.kgrid)  # number of (reducible) k-points
        probe_charge_integral = compute_probe_charge_integral(k.inner_kernel, α) * Nk / Γ

        # Potential of the Gaussian charges: sum over all G+Q with Q in the k-point grid,
        # i.e. over all momentum transfers Q = k - k', except G+Q=0.
        # Note: build_qpoints derives the Q-points from basis.kpoints. The uniform
        # weight below rescales the sum to Nk terms, which is exact if basis.kpoints
        # is the full (non-symmetry-reduced) k-grid.
        Q_points = build_qpoints(basis)
        probe_charge_sum = sum(Q_points) do Q
            weight = Nk / length(Q_points)
            summands = (eval_kernel_fourier(k.inner_kernel, basis, Q)
                        .* exp.(-α .* _norm2_Gplusq_on_cube(basis, Q)))
            if iszero(Q)  # skip the singular G+Q=0 term
                GPUArraysCore.@allowscalar summands[1] = zero(T)
            end
            weight * sum(summands)
        end

        GPUArraysCore.@allowscalar begin
            kernel_fourier[1] = probe_charge_integral - probe_charge_sum
        end
    end
    kernel_fourier
end


"""
Simply set the G+q=0 component of `inner_kernel` to `replacement`.
This is useful for testing/comparison purposes.

For [`BareCoulomb`](@ref) and `replacement=0` this leads to slow `O(1/L) = O(1 / ∛(Nk))`
convergence where `L` is the size of the supercell,`Nk` is the number of k-points.
"""
struct ReplaceSingularity{K <: InteractionKernel, R <: Real} <: InteractionKernel
    inner_kernel::K
    replacement::R
end
function eval_kernel_fourier(k::ReplaceSingularity, p::AbstractVector{T}) where {T}
    iszero(p) ? T(k.replacement) : eval_kernel_fourier(k.inner_kernel, p)
end


"""
Calculates the average of the Coulomb kernel K(G+q) of `inner_kernel` over the Brillouin
zone voxel associated with each grid point. It is particularly well suited for highly
anisotropic cells. Note that this singularity treatment only becomes available once the
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
- `n_quadrature_points::Int`: The number of Gauss-Legendre quadrature points used per dimension.
  Defaults to 12. For highly anisotropic cells or rigorous Thermodynamic Limit (TDL) extrapolations,
    it is advisable to check if higher values (e.g., to 18 or 24) eliminate numerical noise.

## Reference
J. Chem. Phys. 160, 051101 (2024) (doi.org/10.1063/5.0182729)
"""
struct VoxelAverage{K <: InteractionKernel} <: InteractionKernel
    inner_kernel::K
    n_quadrature_points::Int
end
function VoxelAverage(inner_kernel=BareCoulomb(); n_quadrature_points=12)
    VoxelAverage(inner_kernel, n_quadrature_points)
end

# For the implementation see DFTKFastGaussQuadratureExt.jl
