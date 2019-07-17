# This is basically maple/lda_c_vwn.mpl and maple/vwn.mpl from libxc
#
# LDA correlation according to Vosko Wilk,and Nusair, (DOI 10.1139/p80-159)
#
# This version generates the "standard" parametrisation, where a Pade approximation of a
# numerical diffusion Monte-Carlo simulation is used.

as_rational(x) = nsimplify(Sym(x))

A_vwn  = as_rational.([ 0.0310907, 0.01554535, -1/(6*PI^2)])
b_vwn  = as_rational.([ 3.72744,   7.06042,    1.13107  ])
c_vwn  = as_rational.([12.9352,   18.0578,    13.0045   ])
x0_vwn = as_rational.([-0.10498,  -0.32500,   -0.0047584])

A_rpa  = as_rational.([ 0.0310907,  0.01554535,  -1/(6*PI^2)])
b_rpa  = as_rational.([13.0720,    20.1231,      1.06835  ])
c_rpa  = as_rational.([42.7198,   101.578,      11.4813   ])
x0_rpa = as_rational.([-0.409286,  -0.743294,   -0.228344 ])

Q(b, c)      = sqrt(4*c - b^2)
f1(b, c)     = 2*b/Q(b, c)
f2(b, c, x0) = b*x0/(x0^2 + b*x0 + c)
f3(b, c, x0) = 2*(2x0 + b)/Q(b, c)

fpp = 4/(9*(2^(1//3) - 1))

fx(b, c, rs) = rs + b*sqrt(rs) + c

f_aux(A, b, c, x0, rs) = A*(
  + log(rs/fx(b, c, rs))
  + (f1(b, c) - f2(b, c, x0)*f3(b, c, x0))*atan(Q(b, c)/(2*sqrt(rs) + b))
  - f2(b, c, x0)*log((sqrt(rs) - x0)^2/fx(b, c, rs))
)

DMC(rs, z) = (
    + f_aux(A_vwn[2], b_vwn[2], c_vwn[2], x0_vwn[2], rs)
    - f_aux(A_vwn[1], b_vwn[1], c_vwn[1], x0_vwn[1], rs)
)

DRPA(rs, z) = (
    + f_aux(A_rpa[2], b_rpa[2], c_rpa[2], x0_rpa[2], rs)
    - f_aux(A_rpa[1], b_rpa[1], c_rpa[1], x0_rpa[1], rs)
)

f_vwn(rs, z) = (
  + f_aux(A_vwn[1], b_vwn[1], c_vwn[1], x0_vwn[1], rs)
  + f_aux(A_vwn[3], b_vwn[3], c_vwn[3], x0_vwn[3], rs)*f_zeta(z)*(1 - z^4)/fpp
  +  DMC(rs, z)*f_zeta(z)*z^4
)

f(rs, z, junk...) = f_vwn(rs, z)
