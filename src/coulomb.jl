@doc raw"""
Abstract type for electron–electron interaction kernels (Coulomb and Coulomb-like), used e.g.
for the exact-exchange term.

### Interface

The **required** method evaluates the kernel on the plane-wave grid:
```julia
eval_kernel_fourier(kernel, basis; qpt)   # k̂(G+q) for all G of the k-point qpt
```
Kernels that are a genuine (closed-form) function of the Fourier vector `p` may *also* implement
the **optional** pointwise method
```julia
eval_kernel_fourier(kernel, p)            # k̂(p) at one Cartesian Fourier vector p (a Vec3)
```
from which the grid method comes for free (the default loops it). Kernels defined only relative to
the grid ([`WignerSeitzTruncatedCoulomb`](@ref), [`ProbeCharge`](@ref)) instead override the grid
method, and their pointwise method falls back to an error.

`qpt` is a `Kpoint` for now (it carries the momentum transfer `q` and the spherical G-set);
this is slated to become a plain `q::Vec3` on the cube grid.

Long-range kernels diverge as ``p \to 0`` and need their ``G+q=0`` component fixed by a
**singularity treatment**, expressed as a *wrapper kernel* composing around an `inner_kernel`:
[`ProbeCharge`](@ref), [`ReplaceSingularity`](@ref), [`VoxelAverage`](@ref). Kernels with a finite
``p=0`` limit ([`ShortRangeCoulomb`](@ref), [`SphericallyTruncatedCoulomb`](@ref),
[`WignerSeitzTruncatedCoulomb`](@ref)) return the correct value there themselves and need no wrapper.

Kernels that support the Gygi–Baldereschi probe-charge treatment additionally implement
```julia
compute_probe_charge_integral(kernel, α)   # ∫_BZ k̂(q) e^{-α q²} dq  (no 1/Γ factor)
```
which errors by default.

### Available kernels
- [`BareCoulomb`](@ref): ``1/r``
- [`ShortRangeCoulomb`](@ref): ``\mathrm{erfc}(μr)/r``
- [`LongRangeCoulomb`](@ref): ``\mathrm{erf}(μr)/r``
- [`SphericallyTruncatedCoulomb`](@ref): ``θ(R-r)/r``
- [`WignerSeitzTruncatedCoulomb`](@ref): ``1/r`` truncated at the Wigner–Seitz cell boundary

### Singularity treatments (wrapper kernels)
- [`ProbeCharge`](@ref): Gygi–Baldereschi probe charge
- [`ReplaceSingularity`](@ref): set the ``G+q=0`` component to a given value
- [`VoxelAverage`](@ref): average the kernel over the Brillouin-zone voxel
"""
abstract type InteractionKernel end
Base.Broadcast.broadcastable(k::InteractionKernel) = Ref(k)

# TODO: add a real-space evaluation eval_kernel_real(kernel, r) (useful for debugging).

# Required method: kernel on the grid (spherical cutoff |G+q|² < 2Ecut of `qpt`). Default loops
# the pointwise method; overridden by kernels that need the grid or a grid-derived parameter.
function eval_kernel_fourier(kernel::InteractionKernel, basis::PlaneWaveBasis{T};
                             q=zero(Vec3{T})) where {T}
    # Evaluate on the full (cube) FFT grid: a pair density ψᵢ*ψⱼ has content beyond the orbital
    # sphere, so the exchange kernel must live on the density grid, not the k-point sphere.
    # TODO: for q=0 the coefficients could be symmetrized so that the inverse FFT is exactly real.
    recip = basis.model.recip_lattice
    map(G -> eval_kernel_fourier(kernel, recip * (G + q)), G_vectors(basis))
end

# Optional pointwise method k̂(p): kernels that are a closed-form function of `p` implement it;
# kernels defined only on a grid (WignerSeitz, ProbeCharge) fall back to this error.
function eval_kernel_fourier(kernel::InteractionKernel, p::AbstractVector)
    error("$(typeof(kernel)) has no pointwise Fourier evaluation (it is defined only relative " *
          "to a grid); use eval_kernel_fourier(kernel, basis; qpt).")
end

# Probe-charge integral ∫_BZ k̂(q) e^{-α q²} dq; only meaningful for the singular kernels.
function compute_probe_charge_integral(kernel::InteractionKernel, α)
    error("compute_probe_charge_integral is not implemented for $(typeof(kernel)); this kernel " *
          "cannot be regularized with ProbeCharge.")
end


"""
Bare Coulomb interaction ``1/r``, with Fourier kernel ``4π/|p|²``. Diverges at ``p=0``, so it
must be wrapped in a singularity treatment ([`ProbeCharge`](@ref), [`ReplaceSingularity`](@ref),
[`VoxelAverage`](@ref)) to obtain a well-defined ``G+q=0`` component.
"""
struct BareCoulomb <: InteractionKernel end
function eval_kernel_fourier(::BareCoulomb, p::AbstractVector)
    p2 = norm2(p)
    4 * oftype(p2, π) / p2
end
compute_probe_charge_integral(::BareCoulomb, α) = 8π^2 * sqrt(π / α)


"""
Short-range Coulomb interaction ``\\mathrm{erfc}(μr)/r`` (finite at ``p=0``), with cutoff
parameter `μ` in inverse length units.
"""
struct ShortRangeCoulomb{T <: Real} <: InteractionKernel
    μ::T
end
ShortRangeCoulomb(; μ=0.2/u"Å") = ShortRangeCoulomb(austrip(μ))
ShortRangeCoulomb(μ::Quantity) = ShortRangeCoulomb(austrip(μ))
function eval_kernel_fourier(k::ShortRangeCoulomb, p::AbstractVector)
    # Fourier transform of erfc(μr)/r:  (4π/|p|²)(1 - e^{-|p|²/4μ²}) = (π/μ²) φ(|p|²/4μ²),
    # where φ(t) = (1 - e^{-t})/t → 1 as t → 0 (so the p=0 value is π/μ²).
    T = eltype(p)
    μ = T(k.μ)
    t = norm2(p) / (4μ^2)
    φ = divided_difference(x -> -expm1(-x), x -> exp(-x), t, zero(t))  # stable (1-e^{-t})/t
    T(π) / μ^2 * φ
end


"""
Long-range Coulomb interaction ``\\mathrm{erf}(μr)/r``, with Fourier kernel
``(4π/|p|²) e^{-|p|²/4μ²}``. Diverges at ``p=0`` and must be wrapped in a singularity treatment.
`μ` is the cutoff parameter in inverse length units.
"""
struct LongRangeCoulomb{T <: Real} <: InteractionKernel
    μ::T
end
LongRangeCoulomb(; μ=0.2/u"Å") = LongRangeCoulomb(austrip(μ))
LongRangeCoulomb(μ::Quantity) = LongRangeCoulomb(austrip(μ))
function eval_kernel_fourier(k::LongRangeCoulomb, p::AbstractVector)
    # Fourier transform of erf(μr)/r:  (4π/|p|²) e^{-|p|²/4μ²}.
    T = eltype(p)
    p2 = norm2(p)
    4T(π) / p2 * exp(-p2 / (4 * T(k.μ)^2))
end
function compute_probe_charge_integral(k::LongRangeCoulomb, α::T) where {T}
    8T(π)^2 * sqrt(T(π) / (α + 1/(4 * T(k.μ)^2)))
end


"""
Spherically truncated Coulomb interaction ``θ(R_\\text{cut}-r)/r`` (finite at ``p=0``).
If `Rcut` is `nothing` it defaults to ``R_\\text{cut} = \\sqrt[3]{3Ω/4π}`` with `Ω` the unit-cell
volume (resolved once per basis).

## References
- [J. Spencer, A. Alavi. Phys. Rev. B **77**, 193110 (2008)](https://doi.org/10.1103/PhysRevB.77.193110)
"""
@kwdef struct SphericallyTruncatedCoulomb{T} <: InteractionKernel
    Rcut::T = nothing
end
# sin(√v)/√v = sinc(√v), evaluated as a function of v = x² so it stays differentiable at v=0
# (the elementary form's √v is not): Taylor series near 0, direct form away from it.
function sinc_sqrt(v)
    abs(v) < oftype(v, 1//64) && return evalpoly(v, (1//1, -1//6, 1//120, -1//5040, 1//362880))
    s = sqrt(v)
    sin(s) / s
end
function eval_kernel_fourier(k::SphericallyTruncatedCoulomb, p::AbstractVector)
    # Fourier transform of θ(R-r)/r:  (4π/|p|²)(1 - cos(R|p|)) = 2π R² sinc(R|p|/2)²
    # (using 1 - cos x = 2 sin²(x/2)), with p=0 value 2π R².
    T = eltype(p)
    R = T(k.Rcut)
    w = norm2(p)
    2T(π) * R^2 * sinc_sqrt(R^2 * w / 4)^2
end
function eval_kernel_fourier(k::SphericallyTruncatedCoulomb, basis::PlaneWaveBasis{T};
                             q=zero(Vec3{T})) where {T}
    Rcut = @something k.Rcut cbrt(3 * basis.model.unit_cell_volume / (4π))
    kresolved = SphericallyTruncatedCoulomb(Rcut)
    recip = basis.model.recip_lattice
    map(G -> eval_kernel_fourier(kresolved, recip * (G + q)), G_vectors(basis))
end


"""
Coulomb interaction ``1/r`` truncated at the Wigner–Seitz cell boundary.

We split ``1/r = \\mathrm{erfc}(ωr)/r + \\mathrm{erf}(ωr)/r``. The short-range part is exactly
[`ShortRangeCoulomb`](@ref)`(ω)` and is added analytically (replacing the Brillouin zone by all
space, valid since ``\\mathrm{erfc}(ω R_\\text{in})`` is tiny for the inradius ``R_\\text{in}``).
The long-range part ``\\mathrm{erf}(ωr)/r``, restricted to the Wigner–Seitz cell, is obtained by
FFT of its real-space samples. The range separation `ω` is fixed by balancing the truncation
error ``\\mathrm{erfc}(ω R_\\text{in})`` of the short-range part against the FFT-grid
representability error ``e^{-G_\\text{Nyquist}/ω}`` of the long-range part.

This kernel depends on the grid (through `ω`) and therefore only defines the batch method.

## Reference
- [R. Sundararaman, T. A. Arias. Phys. Rev. B **87**, 165122 (2013)](https://doi.org/10.1103/PhysRevB.87.165122)
"""
struct WignerSeitzTruncatedCoulomb <: InteractionKernel end
# No pointwise method: WignerSeitz has no closed-form k̂(p), so it uses the generic error fallback.
# TODO: a basis-free 'slow Fourier transform' pointwise method (issue #1322).
@views function eval_kernel_fourier(::WignerSeitzTruncatedCoulomb,
                                    basis::PlaneWaveBasis{T}; q=zero(Vec3{T})) where {T}
    model = basis.model

    # === Inradius R_in of the Wigner–Seitz cell ===
    # R_in = min over nonzero lattice vectors R of |R|/2 (distance to the perpendicular bisector
    # plane). Cauchy–Schwarz bounds the integer coefficients by |n_i| ≤ a_min |b_i| / 2π.
    L_min = minimum(norm, eachcol(model.lattice))
    nx, ny, nz = estimate_integer_lattice_bounds(model.lattice, L_min)
    R_in = T(Inf)
    for ix in -nx:nx, iy in -ny:ny, iz in -nz:nz
        ix == 0 && iy == 0 && iz == 0 && continue
        R_in = min(R_in, norm(model.lattice * [ix, iy, iz]) / 2)
    end

    # Range separation ω, balancing the two errors described in the docstring.
    G_Nyquist = minimum(basis.fft_size[d] / 2 * norm(model.recip_lattice[:, d]) for d = 1:3)
    ε = exp(-T(1)/2 * G_Nyquist * R_in)
    ω = sqrt(-log(ε)) / R_in
    ε_actual = erfc(ω * R_in)
    if ε_actual > 1e-8
        @warn "Coarse grid for Wigner–Seitz truncation. Effective error: $ε_actual"
    end

    # === Long-range term erf(ωr)/r restricted to the Wigner–Seitz cell, via FFT ===
    # TODO: sampling erf(ωr)/r on the real-space grid and FFTing it introduces aliasing (the
    #       function is not band-limited); a finer real-space grid would reduce it. See Appendix
    #       A.1 of the reference.
    erfder(x) = 2 / sqrt(T(π)) * exp(-x^2)  # d/dx erf(x)
    rvecs = r_vectors(basis)
    V_lr_real = zeros(Complex{T}, basis.fft_size...)
    for idx in CartesianIndices(V_lr_real)
        r_frac = rvecs[idx]
        # Shortest Cartesian distance to a lattice point (minimum image). The fractional
        # minimum-image convention is not enough for non-orthorhombic cells, so we also scan
        # neighboring cells to find the true minimum.
        r_centered = r_frac .- round.(r_frac)
        d_min = norm(model.lattice * r_centered)
        for dx in -nx:nx, dy in -ny:ny, dz in -nz:nz
            dx == 0 && dy == 0 && dz == 0 && continue
            d_min = min(d_min, norm(model.lattice * (r_centered - T[dx, dy, dz])))
        end
        # erf(ω d)/d, with the AD-clean d→0 limit 2ω/√π supplied by divided_difference
        V_lr_real[idx]  = ω * divided_difference(erf, erfder, ω * d_min, zero(T))
        V_lr_real[idx] *= cis2pi(-dot(q, r_frac))  # Bloch phase e^{-2πi q·r}
    end
    kernel_fourier_lr = real.(fft(basis, V_lr_real)) .* sqrt(model.unit_cell_volume)

    # === Add the analytic short-range term erfc(ωr)/r = ShortRangeCoulomb(ω), on the cube grid ===
    short_range = ShortRangeCoulomb(ω)
    recip = model.recip_lattice
    map(to_cpu(G_vectors(basis)), kernel_fourier_lr) do G, lr
        eval_kernel_fourier(short_range, recip * (G + q)) + lr
    end
end


"""
Replace the ``G+q=0`` component of `inner_kernel` by `replacement`, leaving every other component
untouched. Use for genuinely singular kernels ([`BareCoulomb`](@ref), [`LongRangeCoulomb`](@ref));
kernels with a finite ``p=0`` limit already return the correct value there.

For [`BareCoulomb`](@ref) with `replacement = 0` this leads to slow ``O(1/∛N_k)`` convergence with
the number of k-points ``N_k``.
"""
struct ReplaceSingularity{K <: InteractionKernel, R} <: InteractionKernel
    inner_kernel::K
    replacement::R
end
function eval_kernel_fourier(k::ReplaceSingularity, p::AbstractVector)
    iszero(p) ? eltype(p)(k.replacement) : eval_kernel_fourier(k.inner_kernel, p)
end


@doc raw"""
Probe-charge (Gygi–Baldereschi) treatment of the ``G+q=0`` singularity of `inner_kernel`.

Regularize the ``G+q=0`` component by adding and subtracting the potential generated by an array
of unit Gaussian charges of width `sqrt(2α)` placed at the supercell grid points (except the
origin). `α` should give a localized charge well-representable in the plane-wave basis; we default
to `α = π²/Ecut` (VASP default). Convergence is ``O(1/L³) = O(1/N_k)`` with `N_k` the number of
k-points. Only usable with kernels implementing `compute_probe_charge_integral`
([`BareCoulomb`](@ref), [`LongRangeCoulomb`](@ref)).

## References
- [S. Massidda, M. Posternak, A. Baldereschi. Phys. Rev. B **48**, 5058 (1993)](https://doi.org/10.1103/PhysRevB.48.5058)
"""
struct ProbeCharge{K <: InteractionKernel} <: InteractionKernel
    inner_kernel::K
    α::Union{Float64, Nothing}
end
ProbeCharge(inner_kernel::InteractionKernel=BareCoulomb(); α=nothing) =
    ProbeCharge(inner_kernel, α)
@views function eval_kernel_fourier(k::ProbeCharge, basis::PlaneWaveBasis{T};
                                    q=zero(Vec3{T})) where {T}
    # Default well-tested in VASP; ensures e^{-α G²} is a localized charge with full support
    # on the G grid.
    α::T = @something k.α π^2 / basis.Ecut

    recip = basis.model.recip_lattice
    Gpq2 = map(G -> norm2(recip * (G + q)), G_vectors(basis))
    # Raw kernel; its G+q=0 entry may be Inf/NaN and is overwritten below.
    kernel_fourier = eval_kernel_fourier(k.inner_kernel, basis; q)

    # Potential of the Gaussian charges, skipping the G+q=0 term (linear index 1 is G=0).
    probe_charge_sum = sum((kernel_fourier .* exp.(-α .* Gpq2))[2:end])
    # Interaction of the Gaussian charges with the uniform background,
    # (1/Γ) ∫_BZ k̂(q) e^{-α q²} dq, with Γ the Brillouin-zone volume.
    Γ = basis.model.recip_cell_volume
    probe_charge_integral = compute_probe_charge_integral(k.inner_kernel, α) / Γ

    if iszero(q)
        GPUArraysCore.@allowscalar begin
            kernel_fourier[1] = probe_charge_integral - probe_charge_sum
        end
    end
    kernel_fourier
end


@doc raw"""
Average the Coulomb kernel ``\hat k(G+q)`` over the Brillouin-zone voxel associated with each grid
point (well suited to highly anisotropic cells). Conceptually the `HFMEANPOT` flag in VASP, using
improved integration. Only available once `FastGaussQuadrature` is loaded (see
`ext/DFTKFastGaussQuadratureExt.jl`).

- At the singularity ``G+q=0`` the ``1/(G+q)²`` part of `inner_kernel` is integrated by an exact
  volume-to-surface reduction, the remainder by high-order Gauss–Legendre quadrature.
- Elsewhere the whole kernel is integrated by Gauss–Legendre quadrature.

## Arguments
- `n_quadrature_points::Int`: Gauss–Legendre points per dimension (default 12). Increase (18–24)
  for highly anisotropic cells or rigorous thermodynamic-limit extrapolations.

## Reference
- [T. Schäfer et al. J. Chem. Phys. **160**, 051101 (2024)](https://doi.org/10.1063/5.0182729)
"""
struct VoxelAverage{K <: InteractionKernel} <: InteractionKernel
    inner_kernel::K
    n_quadrature_points::Int
end
VoxelAverage(inner_kernel::InteractionKernel=BareCoulomb(); n_quadrature_points=12) =
    VoxelAverage(inner_kernel, n_quadrature_points)
# eval_kernel_fourier(::VoxelAverage, basis, qpt) is implemented in DFTKFastGaussQuadratureExt.
