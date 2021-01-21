using Pkg
Pkg.activate("../../")

using DFTK
using Printf
using LinearAlgebra
using FFTW

BLAS.set_num_threads(16)
FFTW.set_num_threads(16)

include("./SiO2_aposteriori_SCF.jl")
include("./SiO2_aposteriori_Ecut.jl")
