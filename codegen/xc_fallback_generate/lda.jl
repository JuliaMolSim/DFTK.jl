# This is basically maple/lda_x.mpl from libxc

α = symbols("α")

lda_x_ax = -α*RS_FACTOR*X_FACTOR_C/2^(4//3)

f_lda_x(rs, z) = lda_x_ax*((1 + z)^(4//3) + (1 - z)^(4//3))/rs
f(rs, z, junk...) = f_lda_x(rs, z)
