# # Arbitrary floating-point types
#
# Since DFTK is completely generic in the floating-point type
# in its routines, there is no reason to perform the computation
# using double-precision arithmetic (i.e.`Float64`).
# Other floating-point types such as `Float32` (single precision)
# are readily supported as well.
# On top of that we already reported[^HLC2020] calculations
# in DFTK using elevated precision
# from [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl)
# or interval arithmetic
# using [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl).
# In this example, however, we will concentrate on single-precision
# computations with `Float32`.
#
# The setup of such a reduced-precision calculation is basically identical
# to the regular case, since Julia automatically compiles all routines
# of DFTK at the precision, which is used for the lattice vectors.
# Apart from setting up the model with an explicit cast of the lattice
# vectors to `Float32`, there is thus no change in user code required:
#
# [^HLC2020]:
#     M. F. Herbst, A. Levitt, E. Cancès.
#     *A posteriori error estimation for the non-self-consistent Kohn-Sham equations*
#     [ArXiv 2004.13549](https://arxiv.org/abs/2004.13549)

using DFTK

## Setup silicon lattice
a = 10.263141334305942  # lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp(:Si, functional="lda"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

## Cast to Float32, setup model and basis
model = model_DFT(Array{Float32}(lattice), atoms, [:lda_x, :lda_c_vwn])
Ecut = 7
basis = PlaneWaveBasis(model, Ecut, kgrid=[4, 4, 4])

## Run the SCF
scfres = self_consistent_field(basis, tol=1e-4);

# To check the calculation has really run in Float32,
# we check the energies and density are expressed in this floating-point type:

scfres.energies
#-
eltype(scfres.energies.total)
#-
eltype(scfres.ρ)

#
# !!! note "Generic linear algebra routines"
#     For more unusual floating-point types (like IntervalArithmetic or DoubleFloats),
#     which are not directly supported in the standard `LinearAlgebra` library of Julia
#     one additional step is required: One needs to explicitly enable the generic versions
#     of standard linear-algebra operations like `cholesky` or `qr`, which are needed
#     inside DFTK by loading the `GenericLinearAlgebra` package in the user script
#     (i.e. just add ad `using GenericLinearAlgebra` next to your `using DFTK` call).
#
