using BSplineKit: BSplineOrder
using SphericalBesselTransforms: SBTPlan, sbt
using StaticArrays: MVector
import BSplineKit

# Tabulation of the modified Hankel transform (see `hankel`)
#
#     H̃[f](p) = 4π/p^l ∫_0^∞ r² f(r) j_l(p r) dr
#
# of the radial quantities of a pseudopotential. Evaluating this transform by quadrature
# over the radial mesh costs O(n_radial) per p; since it is needed at every |G| (and, in the
# response and stress code paths, in Dual arithmetic) we instead transform once at psp load
# time and interpolate afterwards, making `eval_psp_*_fourier` O(1).
#
# The transform itself is Talman's FFT-based spherical Bessel transform
# (`SphericalBesselTransforms`), which computes H̃ on a whole logarithmic p grid in one shot,
# but requires the input on a logarithmic r grid -- radial meshes of UPF files are linear,
# logarithmic or of the form e^x - 1, so the data is resampled first (`resample_radial`).
#
# The tabulated H̃ is smooth and *even* in p (that is the point of the 1/p^l factor), so it
# splines well in log p, and near p = 0 is carried by the first three terms of its Taylor
# series -- which is also what makes p = 0 representable at all (the log grid cannot reach it).
#
# There are two splines here, doing different jobs and flooring different errors -- which is
# why raising one of the orders alone appears to achieve nothing:
#
#   * in r, the spline carries the sampled psp data onto the transform's log grid. It limits
#     the accuracy of the tabulated *values* (order 4: 1e-10; order 6: 6e-14).
#   * in log p, the spline is the representation we keep, evaluate at every |G| and hand to
#     ForwardDiff. It limits the *derivatives*: an order-k spline is only C^{k-2}, so a cubic
#     one has a merely piecewise-linear second derivative and gets H̃″ wrong by 4e-8, against
#     1e-10 at order 6. High order is cheap here because H̃(p) is analytic (Paley-Wiener: the
#     radial data has compact support).
#
# They are tuned separately (`HANKEL_TABLE_ORDER_R` / `_P`); that both land on 6 is a
# coincidence of two independent limits.

const HANKEL_TABLE_NPOINTS = 4096  # Points of the log-r (and thus log-p) grid
const HANKEL_TABLE_RMIN    = 1e-5  # Bottom of the log-r grid
const HANKEL_TABLE_PMAX    = 1e3   # Top of the log-p grid (Ecut = pmax²/2 ≈ 5·10⁵ Ha)
const HANKEL_TABLE_PCUT    = 1e-2  # Below this, use the small-p series instead of the spline

# The two spline orders. They come out equal, but for unrelated reasons -- do not merge them.
#
# In r the order is limited by the *data*: the psp's own mesh is all we have, and the quantities
# have a cutoff kink. 6 measures 6e-14 on the tabulated values against 1e-10 at order 4, while
# order 8 is *worse* (5e-13) -- it has saturated, so there is nothing above 6 to buy.
const HANKEL_TABLE_ORDER_R = 6
# In log p the order is the smoothness class we need: an order-k spline is only C^{k-2}, so a
# cubic gets H̃″ wrong by 4e-8 where 6 gives 1e-10. Raising it further would keep paying (H̃(p)
# is analytic) if higher derivatives were ever wanted, at ~12 ns/evaluation per two orders.
# **Must be even**: `eval_hankel_table` indexes a cardinal basis, i.e. it assumes B-spline `i`
# is centred on node `i`, which only holds for even order.
const HANKEL_TABLE_ORDER_P = 6

"""
Tabulated modified Hankel transform H̃ of one radial quantity, see `build_hankel_table`.
`coefficients` are the B-spline coefficients of order `K` on the logarithmic grid
`p_i = exp(logpmin + (i-1) Δlogp)`, one per node. Below `HANKEL_TABLE_PCUT` the series
`moment0 + moment2 p² + moment4 p⁴` is used instead.
"""
struct HankelTable{K,T,AT<:AbstractVector{T}}
    coefficients::AT
    logpmin::T
    Δlogp::T
    pmax::T      # Largest |p| the table covers, see `max_momentum_fourier`
    moment0::T   # Coefficients of the Taylor series of H̃ at p = 0. It is even in p, so these
    moment2::T   # are H̃(0), H̃⁽²⁾(0)/2! and H̃⁽⁴⁾(0)/4!. Stopping at p² would leave a 1.8e-8
    moment4::T   # step at pcut; with p⁴ the two branches meet to ~2e-12.
end
hankel_order(::HankelTable{K}) where {K} = K

to_device(::CPU, table::HankelTable) = table
function to_device(architecture::GPU, table::HankelTable{K}) where {K}
    coefficients = to_device(architecture, table.coefficients)
    HankelTable{K,eltype(coefficients),typeof(coefficients)}(
        coefficients, table.logpmin, table.Δlogp, table.pmax,
        table.moment0, table.moment2, table.moment4)
end

"""
The `K` B-splines of order `K` that are nonzero on a cell, at local coordinate `t ∈ [0, 1)`,
by the Cox-de Boor recursion. On a uniform knot vector every denominator of the recursion
collapses to `j`, which is what makes this a short branch-free loop: no knot search and no
lookup into a knot vector. Safe for `Dual` numbers and inside a GPU kernel.
"""
@inline function uniform_bsplines(::Val{K}, t::T) where {K,T}
    N = zero(MVector{K,T})
    N[1] = one(T)
    for j = 1:(K-1)
        saved = zero(T)
        for m = 1:j
            temp  = N[m] / j
            N[m]  = saved + (m - t) * temp   # right = m - t
            saved = (t + j - m) * temp       # left  = t + j - m
        end
        N[j+1] = saved
    end
    N
end

"""
Evaluate a `HankelTable` at `p`. Written in terms of plain scalars and one array rather than
the `HankelTable` itself, so that it can be called from a GPU kernel (a `HankelTable` is not
`isbits`); the vectorized `eval_psp_*_fourier` destructure the table and call this.
"""
function eval_hankel_table(coefficients::AbstractVector, ::Val{K}, logpmin, Δlogp,
                           moment0, moment2, moment4, p) where {K}
    # The transform is even in p, so a few terms of its Taylor series carry it to pcut. This
    # is also the p = 0 branch (the table lives on a logarithmic grid, log(0) = -Inf).
    p < HANKEL_TABLE_PCUT && return moment0 + moment2 * p^2 + moment4 * p^4

    # Position of p on the log grid, in (fractional) node units, and the cell holding it.
    # Clamping keeps the K coefficients below in bounds; it also means p > pmax silently
    # extrapolates rather than erroring, which `max_momentum_fourier` rules out up front.
    u = (log(p) - logpmin) / Δlogp
    icell = clamp(floor(Int, u), K ÷ 2, length(coefficients) - K ÷ 2)

    # B-spline i is centred on node i, so the K nonzero ones on this cell are
    # i = icell+1-K/2 … icell+K/2. The interpolant is cardinal only away from the ends of the
    # grid, which is what the clamp above keeps us clear of -- and we never evaluate there in
    # any case: below pcut we take the series, and pmax sits far past any |G|.
    N = uniform_bsplines(Val(K), u - icell)
    acc = zero(promote_type(eltype(coefficients), typeof(u)))
    for m = 1:K
        @inbounds acc += N[m] * coefficients[icell + 1 - K ÷ 2 + m]
    end
    acc
end

function (table::HankelTable{K})(p) where {K}
    eval_hankel_table(table.coefficients, Val(K), table.logpmin, table.Δlogp,
                      table.moment0, table.moment2, table.moment4, p)
end

"""
Evaluate a `HankelTable` on a whole vector of momenta, which is how the form factors are
built and the only path that runs on the GPU. The table is destructured into its `isbits`
pieces first: the kernel can capture those and the coefficient array, but not the struct.
"""
function eval_hankel_table(table::HankelTable, ps::AbstractVector)
    (; coefficients, logpmin, Δlogp, moment0, moment2, moment4) =
        to_device(architecture(ps), table)
    order = Val(hankel_order(table))
    map(p -> eval_hankel_table(coefficients, order, logpmin, Δlogp,
                               moment0, moment2, moment4, p), ps)
end

"""
Plan the spherical Bessel transforms of all radial quantities living on a mesh ending at
`rmax` and having angular momenta up to `lmax`. The plan holds the (expensive) FFT plans and
kernel matrices, and is shared by all quantities of a pseudopotential, see `PspUpf`.
"""
function hankel_table_plan(rmax::T, lmax::Integer) where {T}
    rgrid = exp.(range(log(T(HANKEL_TABLE_RMIN)), log(rmax), length=HANKEL_TABLE_NPOINTS))
    SBTPlan{T}(collect(rgrid), Int(lmax), T(HANKEL_TABLE_PMAX))
end

"""
Tabulate the modified Hankel transform of the radial quantity `r2_f` (that is r² f(r))
given on the radial mesh `rgrid`, for angular momentum `l`. The transform is done by `plan`,
whose r grid must end at `last(rgrid)`.
"""
function build_hankel_table(plan::SBTPlan{T}, rgrid, r2_f, l::Integer) where {T}
    @assert length(rgrid) == length(r2_f)
    # Quantities ending before the plan's grid (projectors, which are strictly zero past
    # their cutoff radius) are zero-padded by `resample_radial`; ending after it would
    # silently truncate.
    @assert last(rgrid) ≤ last(plan.r) * (1 + 8eps(T))

    # `sbt` evaluates the bare sum ∑ᵢ f(rᵢ) rᵢ³ jₗ(p rᵢ) Δρ -- a *rectangle* rule in log r, so
    # only O(Δρ) on its own. We can fix that for free: the transform kernel depends on i+j only,
    # so multiplying fᵢ by any per-node quadrature weight leaves it a convolution (and hence an
    # FFT). See `gregory_weights` for which weights, and why.
    f = resample_radial(rgrid, r2_f, plan.r) ./ plan.r .^ 2 .* gregory_weights(T, plan.nr)
    values = 4T(π) .* sbt(l, f, plan) ./ plan.k .^ l

    # Interpolate in log p. The grid is uniform, so away from its ends these B-spline
    # coefficients are the cardinal ones `eval_hankel_table` expects.
    K = HANKEL_TABLE_ORDER_P
    itp = BSplineKit.interpolate(collect(log.(plan.k)), values, BSplineOrder(K))
    coefficients = collect(BSplineKit.coefficients(BSplineKit.spline(itp)))

    # The series must take over before the spline runs out of nodes (pmin = pmax rmin/rmax,
    # so a pseudopotential cut off at a very small rcut could push pmin past pcut).
    @assert plan.kmin < HANKEL_TABLE_PCUT

    # jₗ(x)/x^l = 1/(2l+1)!! (1 - x²/(2(2l+3)) + x⁴/(8(2l+3)(2l+5)) - …) integrated against
    # r2_f gives the small-p series of H̃, i.e. moments of the radial quantity. Integrate them
    # on the *same* log grid, with the same weights, as the transform above: they are then the
    # p → 0 limit of the very same discrete sum, so the two branches meet at pcut to roundoff.
    # (A Simpson rule over the psp's own mesh is a cheaper but ~1e-6-accurate answer, which
    # shows up as a visible step at the crossover.) ∫g dr = Δρ ∑ᵢ wᵢ g(rᵢ) rᵢ on a log grid,
    # and `f` already carries the weights wᵢ and a factor 1/rᵢ².
    dfact = prod(1:2:(2l+1); init=1)
    moment(n, weight) = 4T(π) / (dfact * weight) * plan.Δρ * sum(f .* plan.r .^ (l+n+3))
    moment0 =  moment(0, 1)
    moment2 = -moment(2, 2 * (2l + 3))
    moment4 =  moment(4, 8 * (2l + 3) * (2l + 5))

    HankelTable{K,T,Vector{T}}(coefficients, log(plan.kmin), plan.Δρ,
                               T(HANKEL_TABLE_PMAX), moment0, moment2, moment4)
end

"""
Weights of the 4th-order Gregory quadrature rule ∫f ≈ h ∑ᵢ wᵢ f(xᵢ) on a uniform grid: the
trapezoidal rule, with the three outermost nodes at each end reweighted.

Why this rule. By Euler–Maclaurin, the trapezoidal rule's error on a smooth integrand is made
up *entirely of boundary terms* — h²/12·[g′] − h⁴/720·[g‴] + … evaluated at the two ends — and
the interior cancels to all orders. So the accuracy of a uniform-grid sum is decided at its two
endpoints and nowhere else. Here the integrand r³ f(r) jₗ(pr) dies like r^{l+3} at rmin, with
all its derivatives, so that end is already free; but at rmax the densities and (especially) the
pseudo-atomic wavefunctions are still at ~1e-2 of their peak, so the boundary terms there are
what limits everything. Gregory's rule is precisely "cancel those boundary terms", by
approximating the Euler–Maclaurin derivatives with finite differences of the end nodes — it
fixes the error where the error actually is, and extends to higher order by simply correcting
more nodes.

Measured on the log grid (error/H̃(0) at p ≈ 5, N = 4096): the bare sum `sbt` performs is a
*rectangle* rule and only O(h) — 4.8e-7. Trapezoid alone is O(h²) — 5.5e-8. Gregory is O(h⁴) —
5.6e-12. (Composite Simpson would score the same, 1.5e-12, and would fold in just as freely:
*any* per-node weight does. Gregory is preferred only because it needs no parity constraint on
`n` — composite Simpson wants an odd number of points, and our grid is 4096 — and because it
generalises to higher order, where Newton–Cotes goes unstable.)
"""
function gregory_weights(::Type{T}, n::Integer) where {T}
    weights = ones(T, n)
    n < 6 && return weights  # Too few points for the corrections to make sense
    endpoint = T[3/8, 7/6, 23/24]
    weights[1:3] .= endpoint
    weights[n-2:n] .= reverse(endpoint)
    weights
end

"""
Interpolating spline through the radial quantity `y` sampled on the mesh `r` (not uniform:
UPF meshes are linear, logarithmic or of the form e^x - 1), as a callable that is zero past
the end of the data.

**Never pass `Natural()`.** The quantities splined are r² f(r), whose curvature at the origin
is 2f(0) ≠ 0 whenever l = 0 (every density, the local potential, the l = 0 projectors), so
pinning y′′ = 0 there misrepresents the first cell by O(1). Confined to within one mesh spacing
of r = 0, that error is nearly a delta function -- and the transform of a delta is *flat*, so it
leaves p = 0 (hence every norm and charge, and thus most tests) untouched while putting a ~6e-7
floor under the entire tail of H̃. It surfaces only as a spuriously negative core density in
real space. BSplineKit's default end condition is the right one.
"""
function radial_spline(r::AbstractVector, y::AbstractVector)
    @assert length(r) == length(y)
    # BSplineKit needs at least `order` points. Every real radial mesh has thousands, but a
    # projector cut off very early could in principle be shorter than the order.
    order = min(HANKEL_TABLE_ORDER_R, length(r))
    itp = BSplineKit.interpolate(collect(float.(r)), collect(float.(y)), BSplineOrder(order))
    spline = BSplineKit.spline(itp)
    rlast = float(last(r))
    function (x)
        # Past the end of the data the quantity is zero (it is either cut off there, or the
        # mesh ends). The tolerance matters: exp(log(rmax)) can land one ulp above rmax, and
        # dropping that last sample costs the transform three orders of magnitude of accuracy.
        x > rlast * (1 + 8eps(typeof(rlast))) && return zero(rlast)
        spline(min(x, rlast))
    end
end

"""
Resample the radial quantity `y` given on the mesh `r` onto the mesh `rs` (`radial_spline`).
"""
function resample_radial(r::AbstractVector, y::AbstractVector, rs::AbstractVector)
    spline = radial_spline(r, y)
    spline.(rs)
end
