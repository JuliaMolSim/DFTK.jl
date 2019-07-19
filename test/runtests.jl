using Test
using DFTK

include("determine_grid_size.jl")
include("fourier_transforms.jl")
include("PlaneWaveBasis.jl")
include("PspHgh.jl")
include("Species.jl")

include("build_local_potential.jl")
include("build_nonlocal_projectors.jl")

include("lobpcg.jl")
include("xc_fallback.jl")

include("scf_compare.jl")
include("silicon_noXC.jl")
include("silicon_lda.jl")

include("energy_ewald.jl")
include("energy_nuclear.jl")
