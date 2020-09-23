using Test
using DFTK
using Random

#
# This test suite test arguments. For example:
#     Pkg.test("DFTK"; test_args = ["fast"])
# only runs the "fast" tests (i.e. not the expensive ones)
#     Pkg.test("DFTK"; test_args = ["example"])
# runs only the tests tagged as "example" and
#     Pkg.test("DFTK"; test_args = ["example", "all"])
# runs all tests plus the "example" tests.
#

# By default run expensive tests, but not if in CI environment
# If user supplies the "fast" tag
const FAST_TESTS = ifelse("CI" in keys(ENV), parse(Bool, get(ENV, "CI", "false")),
                          "fast" in ARGS)

# Tags supplied by the user ... filter out "fast" (already dealt with)
TAGS = filter(e -> !(e in ["fast"]), ARGS)
isempty(TAGS) && (TAGS = ["all"])

if FAST_TESTS
    println("   Running fast tests (TAGS = $(join(TAGS, ", "))).")
else
    println("   Running tests (TAGS = $(join(TAGS, ", "))).")
end

# Initialize seed
Random.seed!(0)

# Wrap in an outer testset to get a full report if one test fails
@testset "DFTK.jl" begin
    # Synthetic tests at the beginning, so it fails faster if
    # something has gone badly wrong
    if "all" in TAGS || "functionality" in TAGS
        include("hydrogen_all_electron.jl")
        include("silicon_redHF.jl")
        include("silicon_lda.jl")
        include("silicon_pbe.jl")
        include("scf_compare.jl")
    end

    if "all" in TAGS
        include("determine_fft_size.jl")
        include("fourier_transforms.jl")
        include("PlaneWaveBasis.jl")
        include("interpolation.jl")
        include("load_psp.jl")
        include("PspHgh.jl")
        include("elements.jl")
        include("bzmesh.jl")
        include("bzmesh_symmetry.jl")
        include("external_pymatgen.jl")
        include("external_ase.jl")
    end

    if "all" in TAGS
        include("hamiltonian_consistency.jl")
    end

    if "all" in TAGS
        include("lobpcg.jl")
        include("diag_compare.jl")
        include("xc_fallback.jl")
        include("interval_arithmetic.jl")
    end

    if "all" in TAGS
        include("ewald.jl")
        include("energy_nuclear.jl")
        include("occupation.jl")
        include("energies_guess_density.jl")
        include("compute_density.jl")
        include("forces.jl")
    end

    if "all" in TAGS
        include("variational.jl")
        include("compute_bands.jl")
        include("chi0.jl")
        include("kernel.jl")
    end

    if "all" in TAGS
        include("aqua.jl")
    end

    ("example" in TAGS) && include("runexamples.jl")
end
