using Test
using DFTK
using LinearAlgebra

include("utils.jl")

include("test_FFTs.jl")
include("test_PlaneWaveBasis.jl")
include("test_PspHgh.jl")

include("test_build_local_potential.jl")
include("test_PotNonlocal.jl")

include("test_lobpcg.jl")

include("test_noXC_silicon.jl")
