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
# The tabulated H̃ is smooth and *even* in p (that is the point of the 1/p^l factor), so it is
# well represented by a spline in log p, and near p = 0 by the first three terms of its Taylor
# series -- which is also what makes p = 0 exactly representable (the log grid cannot reach it).
#
# Two splines appear here. They are of the same (high) order but play very different roles:
#
#   * in r, the spline carries the sampled psp data onto the transform's log grid. Its error
#     sets the accuracy of the tabulated *values*.
#   * in log p, the spline is the representation we keep and evaluate. Its order sets how well
#     *derivatives* come out -- it is differentiated by ForwardDiff for stresses and response.
#     An order-k spline is only C^{k-2}, so a cubic one (k = 4) has a merely piecewise-linear
#     second derivative: at k = 4 the error on H̃″ is 4e-8, at k = 6 it is 1e-10. And since
#     H̃(p) is analytic (Paley-Wiener: the data has compact support), high order in p converges
#     very fast and costs almost nothing.

const HANKEL_TABLE_NPOINTS = 4096  # Points of the log-r (and thus log-p) grid
const HANKEL_TABLE_RMIN    = 1e-5  # Bottom of the log-r grid
const HANKEL_TABLE_PMAX    = 1e3   # Top of the log-p grid (Ecut = pmax²/2 ≈ 5·10⁵ Ha)
const HANKEL_TABLE_PCUT    = 1e-2  # Below this, use the small-p series instead of the spline
const HANKEL_TABLE_ORDER   = 6     # B-spline order, in r and in log p alike. Must be even.

"""
Tabulated modified Hankel transform H̃ of one radial quantity, see `build_hankel_table`.
`coefficients` are the B-spline coefficients of order `K` on the logarithmic grid
`p_i = exp(logpmin + (i-1) Δlogp)`, `i = 1:n_nodes`. Below `pcut` the series
`moment0 + moment2 p² + moment4 p⁴` is used instead.
"""
struct HankelTable{K,T,AT<:AbstractVector{T}}
    coefficients::AT
    logpmin::T
    Δlogp::T
    n_nodes::Int
    pmax::T
    pcut::T
    moment0::T   # Coefficients of the Taylor series of H̃ at p = 0. It is even in p, so these
    moment2::T   # are H̃(0), H̃⁽²⁾(0)/2! and H̃⁽⁴⁾(0)/4!. Stopping at p² would leave a 1.8e-8
    moment4::T   # step at pcut; with p⁴ the two branches meet to ~2e-12.
end
hankel_order(::HankelTable{K}) where {K} = K

to_device(::CPU, table::HankelTable) = table
function to_device(architecture::GPU, table::HankelTable{K}) where {K}
    coefficients = to_device(architecture, table.coefficients)
    HankelTable{K,eltype(coefficients),typeof(coefficients)}(
        coefficients, table.logpmin, table.Δlogp, table.n_nodes, table.pmax,
        table.pcut, table.moment0, table.moment2, table.moment4)
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
function eval_hankel_table(coefficients::AbstractVector, ::Val{K}, logpmin, Δlogp, n_nodes,
                           pcut, moment0, moment2, moment4, p) where {K}
    # The transform is even in p, so a few terms of its Taylor series carry it to pcut. This
    # is also the p = 0 branch (the table lives on a logarithmic grid, log(0) = -Inf).
    p < pcut && return moment0 + moment2 * p^2 + moment4 * p^4

    # Position of p on the log grid, in (fractional) node units, and the cell holding it.
    # Clamping keeps the K coefficients below in bounds; it also means p > pmax silently
    # extrapolates rather than erroring, which `max_momentum_fourier` rules out up front.
    u = (log(p) - logpmin) / Δlogp
    icell = clamp(floor(Int, u), K ÷ 2, n_nodes - K ÷ 2)

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
    eval_hankel_table(table.coefficients, Val(K), table.logpmin, table.Δlogp, table.n_nodes,
                      table.pcut, table.moment0, table.moment2, table.moment4, p)
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

    # `sbt` evaluates ∑ᵢ f(rᵢ) rᵢ³ jₗ(p rᵢ) Δρ, i.e. a rectangle rule in log r. That is only
    # spectrally accurate if the integrand vanishes at both ends of the grid: it does at rmin
    # (killed by r³) but not at rmax (the atomic densities and, especially, the pseudo-atomic
    # wavefunctions are still at ~1e-2 of their peak there). The transform kernel depends on
    # i+j only, so any per-node quadrature weight can be folded into f for free: fold in
    # 4th-order Gregory endpoint corrections, which restores O(Δρ⁴) at the mesh end.
    f = resample_radial(rgrid, r2_f, plan.r) ./ plan.r .^ 2 .* gregory_weights(T, plan.nr)
    values = 4T(π) .* sbt(l, f, plan) ./ plan.k .^ l

    # Interpolate in log p. The grid is uniform, so away from its ends these B-spline
    # coefficients are the cardinal ones `eval_hankel_table` expects.
    K = HANKEL_TABLE_ORDER
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

    HankelTable{K,T,Vector{T}}(coefficients, log(plan.kmin), plan.Δρ, plan.nr,
                               T(HANKEL_TABLE_PMAX), T(HANKEL_TABLE_PCUT),
                               moment0, moment2, moment4)
end

"""
Weights of the 4th-order Gregory quadrature rule ∫f ≈ h ∑ᵢ wᵢ f(xᵢ) on a uniform grid: the
trapezoidal rule with endpoint corrections. Interior weights are 1, which is what allows
them to be folded into the (weight-free) sum `sbt` performs.
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

The end condition at r = 0 is not a detail. The quantities splined are r² f(r), whose curvature
there is 2f(0) ≠ 0 whenever l = 0 -- every density, the local potential, the l = 0 projectors.
A *natural* spline, which pins y′′ = 0, therefore misrepresents the first cell by O(1). That
error lives within one mesh spacing of the origin, i.e. it is nearly a delta function, and the
Hankel transform of a delta is *flat*: it leaves p = 0 (hence every norm and charge) untouched
while putting a ~6e-7 floor under the whole tail of H̃, which surfaces as a spuriously negative
core density in real space. BSplineKit's default end condition does the right thing; do **not**
pass `Natural()`. (For l ≥ 1, r² f ~ r^{l+2} does have zero curvature at the origin, and there
the end condition is immaterial.)
"""
function radial_spline(r::AbstractVector, y::AbstractVector)
    @assert length(r) == length(y)
    # The order is capped by how much data there is (a projector can be very short).
    order = clamp(2 * (length(r) ÷ 2), 2, HANKEL_TABLE_ORDER)
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
