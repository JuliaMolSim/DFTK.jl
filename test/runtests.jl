using Test
using DFTK
using Random

#
# This test suite supports test arguments. For example:
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

# Setup threading in DFTK
setup_threading()

# Initialize seed
Random.seed!(0)

# Wrap in an outer testset to get a full report if one test fails
@testset "DFTK.jl" begin
    # Super quick tests
    if "all" in TAGS || "quick" in TAGS
        include("helium_all_electron.jl")
        include("silicon_lda.jl")
        include("iron_pbe.jl")
    end

    # Synthetic tests at the beginning, so it fails faster if
    # something has gone badly wrong
    if "all" in TAGS || "functionality" in TAGS
        include("helium_all_electron.jl")
        include("silicon_redHF.jl")
        include("silicon_pbe.jl")
        include("silicon_scan.jl")
        include("scf_compare.jl")
        include("iron_lda.jl")
        include("oxygen_pbe.jl")
        include("external_potential.jl")
    end

    if "all" in TAGS
        include("split_evenly.jl")
        include("compute_fft_size.jl")
        include("fourier_transforms.jl")
        include("PlaneWaveBasis.jl")
        include("interpolation.jl")
        include("load_psp.jl")
        include("PspHgh.jl")
        include("elements.jl")
        include("bzmesh.jl")
        include("bzmesh_symmetry.jl")
        include("spglib.jl")
        include("external_pymatgen.jl")
        include("external_ase.jl")
        include("external_wannier90.jl")
    end

    if "all" in TAGS
        include("hamiltonian_consistency.jl")
    end

    if "all" in TAGS
        include("lobpcg.jl")
        include("diag_compare.jl")
        include("xc_fallback.jl")

        # This fails with multiple MPI procs, seems like a race condition
        # with MPI + DoubleFloats. TODO debug
        mpi_nprocs() == 1 && include("interval_arithmetic.jl")
    end

    if "all" in TAGS
        include("ewald.jl")
        include("anyons.jl")
        include("energy_nuclear.jl")
        include("occupation.jl")
        include("energies_guess_density.jl")
        include("compute_density.jl")
        include("forces.jl")
        include("stresses.jl")
    end

    if "all" in TAGS
        include("adaptive_damping.jl")
        include("variational.jl")
        include("compute_bands.jl")
        include("random_spindensity.jl")
        include("chi0.jl")
        include("kernel.jl")
        include("serialisation.jl")
        include("compute_jacobian_eigen.jl")
        include("printing.jl")
    end

    if "all" in TAGS && mpi_master()
        include("aqua.jl")
    end

    if "all" in TAGS && mpi_nprocs() == 1  # Distributed implementation not yet available
        include("omegaplusk.jl")
    end

    ("example" in TAGS) && include("runexamples.jl")
end
