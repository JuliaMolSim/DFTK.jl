using Interpolations: interpolate, BSpline, Cubic, Line, OnGrid
using SphericalBesselTransforms: SBTPlan, sbt

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
# is well represented by a cubic spline in log p, and near p = 0 by its two-term Taylor
# series (`moment0`, `moment2`) -- which is also what makes p = 0 exactly representable.

const HANKEL_TABLE_NPOINTS = 4096  # Points of the log-r (and thus log-p) grid
const HANKEL_TABLE_RMIN    = 1e-5  # Bottom of the log-r grid
const HANKEL_TABLE_PMAX    = 1e3   # Top of the log-p grid (Ecut = pmax²/2 ≈ 5·10⁵ Ha)
const HANKEL_TABLE_PCUT    = 1e-2  # Below this, use the small-p series instead of the spline

"""
Tabulated modified Hankel transform H̃ of one radial quantity, see `build_hankel_table`.
`coefficients` are cubic B-spline coefficients on the logarithmic grid
`p_i = exp(logpmin + (i-1) Δlogp)`, `i = 1:n_nodes`; they are ghost-padded, so node `i`
lives at index `i+1`. Below `pcut` the series `moment0 + moment2 p²` is used instead.
"""
struct HankelTable{T,AT<:AbstractVector{T}}
    coefficients::AT
    logpmin::T
    Δlogp::T
    n_nodes::Int
    pmax::T
    pcut::T
    moment0::T   # Coefficients of the Taylor series of H̃ at p = 0. It is even in p, so
    moment2::T   # these are H̃(0), H̃⁽²⁾(0)/2! and H̃⁽⁴⁾(0)/4!. Two terms would leave a
    moment4::T   # ~1e-6 jump at pcut; three bring the branches together to ~1e-10.
end

to_device(::CPU, table::HankelTable) = table
function to_device(architecture::GPU, table::HankelTable)
    coefficients = to_device(architecture, table.coefficients)
    HankelTable{eltype(coefficients),typeof(coefficients)}(
        coefficients, table.logpmin, table.Δlogp, table.n_nodes, table.pmax,
        table.pcut, table.moment0, table.moment2, table.moment4)
end

"""
Evaluate a `HankelTable` at `p`. Written in terms of plain scalars and one array rather than
the `HankelTable` itself, so that it can be called from a GPU kernel (a `HankelTable` is not
`isbits`); the vectorized `eval_psp_*_fourier` destructure the table and call this.
"""
function eval_hankel_table(coefficients::AbstractVector, logpmin, Δlogp, n_nodes,
                           pcut, moment0, moment2, moment4, p)
    # The transform is even in p, so a few terms of its Taylor series carry it to pcut. This
    # is also the p = 0 branch (the table lives on a logarithmic grid, log(0) = -Inf).
    p < pcut && return moment0 + moment2 * p^2 + moment4 * p^4

    # Position of p on the log grid, in (fractional) node units.
    x = (log(p) - logpmin) / Δlogp + 1
    # Cell containing p. Clamping means p > pmax silently extrapolates rather than erroring;
    # with pmax = 1000 that is unreachable in practice (see HANKEL_TABLE_PMAX).
    icell = clamp(floor(Int, x), 1, n_nodes - 1)
    t = x - icell

    # Uniform cubic B-spline basis on the four coefficients around the cell. `coefficients`
    # is ghost-padded (node i ↦ index i+1), so the four are icell-1 .. icell+2 ↦ icell .. icell+3.
    t2 = t * t
    t3 = t2 * t
    b0 = (1 - 3t + 3t2 -  t3) / 6
    b1 = (4      - 6t2 + 3t3) / 6
    b2 = (1 + 3t + 3t2 - 3t3) / 6
    b3 =                  t3  / 6
    @inbounds (  b0 * coefficients[icell]     + b1 * coefficients[icell+1]
               + b2 * coefficients[icell+2]   + b3 * coefficients[icell+3])
end

(table::HankelTable)(p) = eval_hankel_table(table.coefficients, table.logpmin, table.Δlogp,
                                            table.n_nodes, table.pcut, table.moment0,
                                            table.moment2, table.moment4, p)

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
given on the radial mesh `rgrid`, for angular momentum `l`. `quadrature` is used for the
small-p moments only; the transform itself is done by `plan`, whose r grid must end at
`last(rgrid)`.
"""
function build_hankel_table(plan::SBTPlan{T}, rgrid, r2_f, l::Integer, quadrature) where {T}
    @assert length(rgrid) == length(r2_f)
    # Quantities ending before the plan's grid (projectors, which are strictly zero past
    # their cutoff radius) are zero-padded by `resample_radial`; ending after it would
    # silently truncate.
    @assert last(rgrid) ≤ last(plan.r) * (1 + 8eps(T))

    # `sbt` evaluates ∑ᵢ f(rᵢ) rᵢ³ jₗ(p rᵢ) Δρ, i.e. a rectangle rule in log r. That is only
    # spectrally accurate if the integrand vanishes at both ends of the grid: it does at rmin
    # (killed by r³) but not at rmax (atomic densities and pseudo-wavefunctions are still
    # nonzero there). The transform kernel depends on i+j only, so any per-node quadrature
    # weight can be folded into f for free: fold in 4th-order Gregory endpoint corrections,
    # which restores O(Δρ⁴) at the mesh end.
    f = resample_radial(rgrid, r2_f, plan.r) ./ plan.r .^ 2 .* gregory_weights(T, plan.nr)
    values = 4T(π) .* sbt(l, f, plan) ./ plan.k .^ l

    # Prefilter to cubic B-spline coefficients. `parent` drops the ghost-point offset, so the
    # coefficient of node i sits at index i+1; we evaluate the B-spline basis ourselves in
    # `eval_hankel_table` because an Interpolations object cannot be used on the GPU.
    itp = interpolate(values, BSpline(Cubic(Line(OnGrid()))))
    coefficients = collect(parent(itp.coefs))

    # The series must take over before the spline runs out of nodes (pmin = pmax rmin/rmax,
    # so a pseudopotential cut off at a very small rcut could push pmin past pcut).
    @assert plan.kmin < HANKEL_TABLE_PCUT

    # jₗ(x)/x^l = 1/(2l+1)!! (1 - x²/(2(2l+3)) + x⁴/(8(2l+3)(2l+5)) - …) integrated against
    # r2_f gives the small-p series of H̃, i.e. moments of the radial quantity.
    dfact = prod(1:2:(2l+1); init=1)
    moment(n, weight) = 4T(π) / (dfact * weight) * quadrature((i, r) -> r2_f[i] * r^(l+n),
                                                              rgrid)
    moment0 =  moment(0, 1)
    moment2 = -moment(2, 2 * (2l + 3))
    moment4 =  moment(4, 8 * (2l + 3) * (2l + 5))

    HankelTable{T,Vector{T}}(coefficients, log(plan.kmin), plan.Δρ, plan.nr,
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
Natural cubic spline through the points `(r[i], y[i])`, with `r` not necessarily uniform.
Used to move radial quantities from the pseudopotential's own mesh (linear, logarithmic or
e^x - 1) to the logarithmic mesh the spherical Bessel transform needs.
"""
struct RadialSpline{T}
    r::Vector{T}
    y::Vector{T}
    y′′::Vector{T}
end
function RadialSpline(r::AbstractVector, y::AbstractVector)
    @assert length(r) == length(y)
    T = promote_type(eltype(r), eltype(y))
    n = length(r)
    h = diff(r)
    # Second derivatives from continuity of the first derivative, with y′′ = 0 at both ends.
    y′′ = zeros(T, n)
    if n > 2
        d  = [2 * (h[i-1] + h[i])                                for i = 2:n-1]
        dl = [h[i]                                               for i = 2:n-2]
        rhs = [6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1]) for i = 2:n-1]
        y′′[2:n-1] = Tridiagonal(dl, d, copy(dl)) \ rhs
    end
    RadialSpline{T}(collect(r), collect(y), y′′)
end

function (spline::RadialSpline{T})(x) where {T}
    (; r, y, y′′) = spline
    n = length(r)
    # Past the end of the data the quantity is zero (it is either cut off there, or the mesh
    # ends). The tolerance matters: exp(log(rmax)) can land one ulp above rmax, and dropping
    # that last sample costs the transform three orders of magnitude of accuracy.
    x > r[n] * (1 + 8eps(T)) && return zero(T)
    icell = clamp(searchsortedlast(r, x), 1, n - 1)
    h = r[icell+1] - r[icell]
    a = (r[icell+1] - x) / h
    b = 1 - a
    (  a * y[icell] + b * y[icell+1]
     + ((a^3 - a) * y′′[icell] + (b^3 - b) * y′′[icell+1]) * h^2 / 6)
end

"""
Resample the radial quantity `y` given on the mesh `r` onto the mesh `rs`, with a natural
cubic spline (`RadialSpline`).
"""
function resample_radial(r::AbstractVector, y::AbstractVector, rs::AbstractVector)
    spline = RadialSpline(r, y)
    spline.(rs)
end
