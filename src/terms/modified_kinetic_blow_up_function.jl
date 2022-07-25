# Implementation of the blow-up function used to compute modified kinetic energies.
# For a given interval [x_1,x_2] ⊂ [0,1) and a blow-up order p>0, the blow-up function Gp is
# composed of three parts:
#
#  - g1(x) = x^2 defined on [0,x_1),
#  - g3(x, p) ≥ x^2 ∀p, defined on [x_2,1), that blows-up at x=1 with blow-up rate |⋅|^{-p},
#  - g2, a C^2 interpolation between g1 and g3_p on [x_1,x_2].
#
# In practice we define:
#
# g3(x,p) = g3(x, a, p) = C(a,p)/(1-x)^p, where C(a,p) = (3/2)*(a^2)*(1-a)^(p)
#
# and where the parameter a is optimized so that ||Gp - x^2||_{L^1((0,1))} is minimized.

using Optim

@doc raw"""
Provides a ``C^2`` polynomial interpolation on a given interval of ``[0,1)``
between the ``x\mapsto x^2`` function and a given blow-up part ``g_3``.
The default interval have been numericaly set to match the most the
``x \mapsto x^2`` curve over ``[0,1)`` while avoiding SCF convergence issues.
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
    println(C)
    # Assemble polynomial
    x -> C'*((x * ones(6)) .^(0:5))
end

"""
Assemble all parts of the blow-up function.
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
Given blow-up order p and interpolation interval, optimizes the parameter
a of g3(⋅,a,p) to minimize ``||G_p - x^2||_{L^1((0,1))}`` with constraints:
- ``a \in [0,1)``
- ``\forall x,\quad G_p(x) \geq x^2``
"""
Ca(a, p) = (3/2)*(a^2)*(1-a)^(p)
g3(a, p) = y-> Ca(a, p)/( (1-y)^(p) )
function optimal_g3(p, interval)
    x_axis = LinRange(0,0.99, 1000)
    function f(a)
        # Penalize a ∉ [0,1)
        !(0≤only(a)<1) && (return 1e6)
        g3_p = g3(only(a), p)
        Gp = blow_up_function(g3_p; interval)
        out = Gp.(x_axis) - x_axis .^2
        # Penalize Gp that are bellow the x->x^2 curve
        norm(out,1) .+ norm(out[out .< 0],1)*1e3
    end
    # Optimize with auto-differentiation.
    @show a_opti = only(optimize(f, [0.3], LBFGS()).minimizer)
    g3(a_opti, p)
end
