using Test
using DFTK

# Wrap in an outer testset to get a full report if one test fails
@testset "All tests" begin
    # Synthetic tests at the beginning, so it fails faster if
    # something has gone badly wrong
    include("silicon_redHF.jl")
    include("silicon_lda.jl")
    include("silicon_pbe.jl")
    include("scf_compare.jl")

    include("determine_grid_size.jl")
    include("fourier_transforms.jl")
    include("PlaneWaveBasis.jl")
    include("interpolation.jl")
    include("PspHgh.jl")
    include("Species.jl")
    include("bzmesh.jl")

    include("term_external.jl")
    include("term_nonlocal.jl")
    # TODO Test for term_hartree
    # TODO Test for term_xc

    include("HamiltonianBlock.jl")
    include("lobpcg.jl")
    include("xc_fallback.jl")
    include("interval_arithmetic.jl")

    include("energy_ewald.jl")
    include("energy_nuclear.jl")
    include("occupation.jl")
    include("energies_guess_density.jl")
    include("compute_density.jl")

    include("variational.jl")
    include("compute_bands.jl")
end
