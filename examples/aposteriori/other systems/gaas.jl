using Pkg
Pkg.activate("../../")

using DFTK
using Printf
using LinearAlgebra
using FFTW

BLAS.set_num_threads(4)
FFTW.set_num_threads(4)

include("./gaas_aposteriori_SCF.jl")
include("./gaas_aposteriori_Ecut.jl")
