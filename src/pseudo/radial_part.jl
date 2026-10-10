# The radial part of an atom-centred function of a norm-conserving pseudopotential: a
# projector, a pseudo-atomic wavefunction, the local potential, or a core/valence density.
# It bundles the raw radial data (for real-space evaluation) with a tabulation of its
# modified Hankel transform
#
#     H̃[f](p) = 4π/p^l ∫_0^∞ r² f(r) jₗ(p r) dr,
#
# so that `eval_fourier` is an O(1) spline evaluation rather than a quadrature over the
# radial mesh at every |G|. The transform is needed at every plane wave, and for stresses
# and response in `Dual` arithmetic, so tabulating once at load time is a large speed-up.
#
# H̃ is smooth and *even* in p (that is the point of the 1/p^l factor). We sample it on a
# uniform p grid resolving its oscillations — whose period ≈ 2π/rmax, so the node count
# grows with the radial extent rmax — and interpolate with a cubic spline clamped at p = 0
# to H̃'(0) = 0 (exact, by evenness; the natural end condition would instead misrepresent the
# first cell by O(1) for the l = 0 quantities). The grid being uniform, evaluation finds its
# interval branch-free, keeping it GPU- and ForwardDiff-friendly.

# Top of the p grid, which the tabulation cannot answer above. Two values: the local
# potential and the densities are evaluated on the whole (dense) FFT grid, whose corners
# reach |G| far beyond the orbital cutoff, while projectors and pseudo-wavefunctions are only
# needed on the wavefunction cutoff sphere. This is the same split ABINIT/QE make between
# their vloc and form-factor q-grids; like them, we tabulate on a linear grid and interpolate
# with a cubic spline.
const RADIAL_TABLE_PMAX_LOCAL = 150.0   # covers the density-grid corners up to Ecut ≈ 250 Ha
const RADIAL_TABLE_PMAX_PROJ  = 50.0    # covers the orbital sphere up to Ecut ≈ 1250 Ha
# p-grid resolution, in nodes per oscillation period (≈ 2π/rmax) of H̃. Unlike ABINIT's fixed
# mqgrid=3001 or QE's fixed dq=0.01, the grid is matched to each quantity's bandwidth; this
# lands the widest quantities near ABINIT's node count, with the spline error (≈ few·1e-6 on
# derivatives) still under the pseudopotential data's own ~1e-6 quality.
const RADIAL_TABLE_POINTS_PER_PERIOD = 15

struct RadialPart{T}
    l::Int
    rgrid::Vector{T}    # native radial mesh (possibly cut before its end) ...
    r2_f::Vector{T}     # ... and r² f(r) on it, for real-space evaluation
    Δp::T               # uniform p-grid spacing; the nodes are pⱼ = (j-1) Δp
    values::Vector{T}   # H̃ at the nodes
    moments::Vector{T}  # cubic-spline second derivatives at the nodes (`cubic_spline_moments`)
end

function RadialPart(l::Integer, rgrid::AbstractVector, r2_f::AbstractVector;
                    pmax=RADIAL_TABLE_PMAX_LOCAL)
    @assert length(rgrid) == length(r2_f)
    T = float(promote_type(eltype(rgrid), eltype(r2_f)))

    # Trim the negligible tail: r² f(r) is taken as zero past its cutoff anyway, and the mesh
    # often runs far beyond it (pseudo-wavefunctions especially). Dropping it shrinks both the
    # p-grid (fewer oscillations to resolve) and each transform's quadrature.
    ncut = something(findlast(f -> abs(f) > 1e-8 * maximum(abs, r2_f), r2_f), length(r2_f))
    rgrid = @view rgrid[1:ncut]
    r2_f  = @view r2_f[1:ncut]
    rmax  = T(last(rgrid))

    # Enough nodes to resolve H̃'s ≈ pmax·rmax/2π oscillations up to pmax, with a floor so
    # very narrow quantities still get a usable spline.
    n_periods = pmax * rmax / 2T(π)
    npoints   = max(16, ceil(Int, RADIAL_TABLE_POINTS_PER_PERIOD * n_periods) + 1)
    Δp        = T(pmax) / (npoints - 1)

    values  = hankel_table_values(l, rgrid, r2_f, npoints, Δp)
    moments = cubic_spline_moments(values, Δp)
    RadialPart{T}(l, collect(T, rgrid), collect(T, r2_f), Δp, values, moments)
end

# Weights W with `quadrature(f) == Σₖ Wₖ f(rₖ)`, recovered by probing the (linear) quadrature
# functional with unit vectors. O(nr²) but done once per mesh, and dwarfed by the transform.
function _quadrature_weights(quadrature, r::AbstractVector{T}) where {T}
    W = Vector{T}(undef, length(r))
    e = zeros(T, length(r))
    for k in eachindex(r)
        e[k] = one(T)
        W[k] = quadrature((i, _) -> e[i], r)
        e[k] = zero(T)
    end
    W
end

# The tabulated transform H̃ₗ at the uniform p-grid nodes pⱼ = (j-1)Δp, by quadrature over the
# radial mesh. A fresh `sincos` per (node, mesh point) is the load-time bottleneck, so instead
# sin(pⱼr) and cos(pⱼr) are advanced along the p grid by angle addition — one `sincos` per mesh
# point rather than one per pair — and handed to `sphericalbesselj_fast`, up to that
# recurrence's ~1e-14 drift.
function hankel_table_values(l::Integer, rgrid, r2_f, npoints::Integer, Δp)
    T = float(promote_type(eltype(rgrid), eltype(r2_f)))
    quadrature = default_psp_quadrature(rgrid)
    g = _quadrature_weights(quadrature, rgrid) .* r2_f  # ∫· dr = Σₖ gₖ (·)(rₖ)
    values = zeros(T, npoints)
    @inbounds for k in eachindex(rgrid)
        r = T(rgrid[k]); gk = T(g[k])
        (iszero(r) || iszero(gk)) && continue
        sθ, cθ = sincos(Δp * r)
        s = zero(T); c = one(T)                       # node j = 1: angle 0
        for j = 2:npoints
            s, c = s * cθ + c * sθ, c * cθ - s * sθ   # angle (j-1) Δp r
            values[j] += gk * sphericalbesselj_fast(l, (j - 1) * Δp * r, (s, c))
        end
    end
    for j = 2:npoints
        values[j] *= 4T(π) / ((j - 1) * Δp)^l
    end
    # p → 0 limit: jₗ(x)/x^l → 1/(2l+1)!!, so H̃ₗ(0) = 4π/(2l+1)!! ∫ r² f(r) rˡ dr.
    dfact = T(prod(1:2:(2l+1)))
    values[1] = 4T(π) / dfact * sum(g[k] * T(rgrid[k])^l for k in eachindex(g))
    values
end

# Real-space value f(r) = r²f(r) / r², by linear interpolation of the raw data (zero past
# the end of the mesh). Not performance critical.
function eval_real(rp::RadialPart, r::T) where {T <: Real}
    (; rgrid, r2_f) = rp
    r ≥ rgrid[end] && return zero(T)
    r ≤ rgrid[1]   && return T(r2_f[1] / rgrid[1]^2)
    j = searchsortedlast(rgrid, r)
    t = (r - rgrid[j]) / (rgrid[j+1] - rgrid[j])
    ((1 - t) * r2_f[j] + t * r2_f[j+1]) / r^2
end

eval_fourier(rp::RadialPart, p::Real) = _eval_cubic(rp.values, rp.moments, rp.Δp, p)

# Vectorized (and GPU) path: destructure into the two `isbits` node arrays — a `RadialPart`
# itself cannot be captured by a GPU kernel — move them to the device and map. No reduction
# over `ps` (which would force a device→host sync between the form-factor kernels): the spline
# holds flat past its range instead (see `_eval_cubic`).
function eval_fourier(rp::RadialPart, ps::AbstractVector)
    arch    = architecture(ps)
    values  = to_device(arch, rp.values)
    moments = to_device(arch, rp.moments)
    Δp      = rp.Δp
    map(p -> _eval_cubic(values, moments, Δp, p), ps)
end

# Evaluate the clamped cubic spline of `cubic_spline_moments` at `p`, on the uniform grid
# pⱼ = (j-1) Δp. Beyond the tabulated range `u` is clamped, so the spline is held flat at its
# endpoint rather than cubic-extrapolated to nonsense — the transform is negligible there for
# any sane Ecut (pmax covers the density grid). Branch-free, so it runs inside a GPU kernel.
function _eval_cubic(values::AbstractVector, moments::AbstractVector, Δp, p)
    n = length(values)
    u = clamp(p / Δp, zero(p), oftype(p, n - 1))
    i = clamp(floor(Int, u), 0, n - 2) + 1  # interval [pᵢ, pᵢ₊₁]
    b = u - (i - 1)
    a = 1 - b
    @inbounds oftype(p, a * values[i] + b * values[i+1]
                        + ((a^3 - a) * moments[i] + (b^3 - b) * moments[i+1]) * Δp^2 / 6)
end

# Second derivatives of the interpolating cubic spline on a uniform grid of spacing `h`,
# clamped at the left end to y'(0) = 0 (exact for the even transform H̃) and natural at the
# right end, where the tabulated quantity has decayed to ≈ 0. Solving the tridiagonal system
# once is the only linear algebra the tabulation needs.
function cubic_spline_moments(y::AbstractVector{T}, h::T) where {T}
    n = length(y)
    n ≤ 2 && return zeros(T, n)
    # Unknowns M[1..n-1] (with M[n] = 0). Row 1 imposes the clamped condition 2M₁ + M₂ =
    # 6/h² (y₂ - y₁), i.e. y'(0) = 0; the interior rows are Mⱼ₋₁ + 4Mⱼ + Mⱼ₊₁ = 6/h² δ²yⱼ.
    dl = fill(T(1), n - 2)
    dd = fill(T(4), n - 1); dd[1] = T(2)
    du = fill(T(1), n - 2)
    rhs = zeros(T, n - 1)
    rhs[1] = 6 / h^2 * (y[2] - y[1])
    for j = 2:n-1
        rhs[j] = 6 / h^2 * (y[j-1] - 2y[j] + y[j+1])
    end
    M = zeros(T, n)
    M[1:n-1] .= Tridiagonal(dl, dd, du) \ rhs
    M
end
