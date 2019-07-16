# Adapted from the maple scripts in libxc
# Look at scripts/maple2c_new.pl in particular
#         maple/util.jl


using SymPy

r0, r1 = symbols("ρ1 ρ2")
s0, s1, s2 = symbols("σ0 σ1 σ2")

# unpol, 3D
DIMENSIONS = 3
RS_FACTOR    = (3/(4*PI))^(1//3)
X_FACTOR_C   = 3//8*(3/PI)^(1//3)*4^(2//3)
r_ws(n) = RS_FACTOR/n^(1//DIMENSIONS)
dens(r0, r1) = r0
zeta(r0, r1) = 0
xs0(r0, r1, sigma0, sigma2) = sqrt(sigma0/4)/((r0/2)^(1 + 1//DIMENSIONS))
xs1(r0, r1, sigma0, sigma2) = sqrt(sigma0/4)/((r0/2)^(1 + 1//DIMENSIONS))
xt(r0, r1, sigma0, sigma1, sigma2) = sqrt(sigma0)/r0^(1 + 1//DIMENSIONS)

f_zeta(z) = ((1 + z)^(4//3) + (1 - z)^(4//3) - 2)/(2^(4//3) - 2)

#include("lda.jl")
include("vwn.jl")

# Energy per unit particle
f_expr = f(r_ws(dens(r0, r1)), zeta(r0, r1), xt(r0, r1, s0, s1, s2), xs0(r0, r1, s0, s2), xs1(r0, r1, s0, s2))

# Energy per unit volume
E_expr = f_expr * dens(r0, r1)

# Derivative with respect to r0
dE_expr = diff(E_expr, r0)

symbol = ["ε", "Vρ"]
f_expr = simplify(f_expr)
E_expr = simplify(E_expr)

sexpr, simplified = sympy.cse([f_expr, dE_expr])

for (var, expr) in sexpr
    println(sympy.julia_code(expr, assign_to=var))
end
println()
for (i, simp) in enumerate(simplified[1])
    println(symbol[i], " = ", sympy.julia_code(simp))
end
