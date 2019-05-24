using Test
using DFTK
using LinearAlgebra

include("test_FFTs.jl")
include("test_lobpcg.jl")
include("test_PlaneWaveBasis.jl")
include("test_PspHgh.jl")

include("test_PotLocal.jl")
include("test_PotNonlocal.jl")
