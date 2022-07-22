using Optim

@doc raw"""
Provides a ``C^2`` interpolation between the x->x^2 function and given g3
on a given interval of [0,1).
The default interval have been set experimentaly to match the most the
x^2 curve and to avoid scf convergence issues.
"""
function C2_pol_interpolation(g3; interval=[0.85, 0.90])
    # ForwardDiff second derivative of g3
    dg3(x) = ForwardDiff.derivative(g3, x)
    d2g3(x) = ForwardDiff.derivative(dg3, x)

    # Points of interpolation
    a, b = interval
    x1, x2 = a^2, g3(b)
    y1, y2 = 2*a, dg3(b)
    z1, z2 = 2, d2g3(b)

    # Solve interpolation linear system
    A = [1 a a^2  a^3   a^4    a^5;
         1 b b^2  b^3   b^4    b^5;
         0 1 2*a  3*a^2 4*a^3  5*a^4;
         0 1 2*b  3*b^2 4*b^3  5*b^4;
         0 0 2    6*a   12*a^2 20*a^3;
         0 0 2    6*b   12*b^2 20*b^3]
    B = [x1, x2, y1, y2, z1, z2]                      
    C = A\B

    # Assemble polynomial
    x -> C'*((x * ones(6)) .^(0:5))
end

"""
Define the blow-up function on the interval [0,1)
"""
function blow_up_function(x, g1, g2, g3; interval=[0.85, 0.90])
    x1, x2 = interval
    # Define 3 parts of the function
    (0 ≤ x < x1)   && (return g1(x))
    (x1 ≤ x < x2)  && (return g2(x))
    (x2 ≤ x < 1)   && (return g3(x))
    error("The blow-up function is defined on [0,1). Did you devide by √Ecut ?")
    nothing
end
function blow_up_function(g3; interval=[0.85, 0.90])
    g2 = C2_pol_interpolation(g3; interval)
    g1 = x->x'x
    x->blow_up_function(x, g1, g2, g3; interval)
end
function blow_up_function(blow_up_rate::T; interval=[0.85, 0.90]) where {T<:Real}
    g3 = optimal_g3(blow_up_rate, interval)
    blow_up_function(g3; interval)
end

@doc raw"""
Automatically provide a candidate for the blow-up part of the blow-up function
so that the blow-up function matches the most the x->x^2 curve.
Arguments are:
 - ``ε`` so that the blow-up function has ``|⋅|^{-ε}`` blow-up rate at 1.
 - the interval on which to interpolate between the blow-up part and ``x^2``.
"""
Ca(a, ε) = (3/2)*(a^2)*(1-a)^(ε)
ha(a, ε) = y-> Ca(a, ε)/( (1-y)^(ε) )
function optimal_g3(ε, interval)
    x_axis = LinRange(0,0.99, 1000)
    function f(a)
        # Penalize a ∉ [0,1)
        !(0≤only(a)<1) && (return 1e6)
        g3 = ha(only(a), ε)
        Gm = blow_up_function(g3; interval)
        out = Gm.(x_axis) - x_axis .^2
        # Penalize Gm that are bellow the x->x^2 curve
        norm(out,1) .+ norm(out[out .< 0],1)*1e3
    end
    # Optimize with auto-differentiation.
    a_opti = only(optimize(f, [0.3], LBFGS()).minimizer)
    ha(a_opti, ε)
end
