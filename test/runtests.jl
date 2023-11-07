# Work around JuliaLang/Pkg.jl#2500
# Must be before loading anything
if VERSION < v"1.8-"
    test_project = first(Base.load_path())
    preferences_file = "../LocalPreferences.toml"
    test_preferences_file = joinpath(dirname(test_project), "LocalPreferences.toml")
    if isfile(preferences_file) && !isfile(test_preferences_file)
        cp(preferences_file, test_preferences_file)
        @info "copied LocalPreferences.toml to $test_preferences_file"
    end
end

using TestItemRunner
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

const DFTK_TEST_ARGS = let
    if "DFTK_TEST_ARGS" in keys(ENV)
        append!(split(ENV["DFTK_TEST_ARGS"], ","), ARGS)
    else
        ARGS
    end
end

const RUN_EXAMPLES = "example" in DFTK_TEST_ARGS
# Tags supplied by the user.
const TAGS = isempty(DFTK_TEST_ARGS) ? [:all] : Symbol.(DFTK_TEST_ARGS)

# Work around some issues with test preferences
let
    test_project = first(Base.load_path())
    preferences_file = joinpath(dirname(@__DIR__), "LocalPreferences.toml")
    test_preferences_file = joinpath(dirname(test_project), "LocalPreferences.toml")
    if isfile(preferences_file) && !isfile(test_preferences_file)
        cp(preferences_file, test_preferences_file)
    end
end


# Setup threading in DFTK
setup_threading(; n_blas=2)

# Initialize seed
Random.seed!(0)

exclude_tags = []
:fast ∈ TAGS && push!(exclude_tags, :slow)
:gpu ∉ TAGS  && push!(exclude_tags, :gpu)
mpi_nprocs() > 1 && push!(exclude_tags, :dont_test_mpi)
Sys.iswindows() && push!(exclude_tags, :unix)

fast_string = :fast ∈ TAGS ? "fast" : ""
println("   Running $(fast_string) tests")
println("       User TAGS     = $(join(TAGS, ", ")).")
println("       Excluded TAGS = $(join(exclude_tags, ", ")).")
push!(exclude_tags, :example)  # we run the examples at the end if requested


# Synthetic tests at the beginning, so it fails faster if something has gone badly wrong
if :all ∈ TAGS || :core ∈ TAGS
    @run_package_tests filter=ti->!(any(e -> e ∈ ti.tags, exclude_tags)) && (:core ∈ ti.tags)
    push!(exclude_tags, :core)
end
@run_package_tests filter=ti->!(any(e -> e ∈ ti.tags, exclude_tags))

if :all in TAGS && mpi_master()
    include("aqua.jl")
end

if RUN_EXAMPLES
    @run_package_tests filter=ti->(:example ∈ ti.tags)
end
