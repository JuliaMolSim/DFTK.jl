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

using Aqua
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

# Setup threading in DFTK
setup_threading(; n_blas=2)

# Initialize seed
Random.seed!(0)

const DFTK_TEST_ARGS = let
    if "DFTK_TEST_ARGS" in keys(ENV)
        append!(split(ENV["DFTK_TEST_ARGS"], ","), ARGS)
    else
        ARGS
    end
end

# Tags supplied by the user.
const TAGS = isempty(DFTK_TEST_ARGS) ? [:all] : Symbol.(DFTK_TEST_ARGS)

# Tags excluded from the first run of "all" tests
const EXCLUDED_FROM_ALL = Symbol[:example, :gpu]
:fast ∈ TAGS     && push!(EXCLUDED_FROM_ALL, :slow)
mpi_nprocs() > 1 && push!(EXCLUDED_FROM_ALL, :dont_test_mpi)
Sys.iswindows()  && push!(EXCLUDED_FROM_ALL, :dont_test_windows)

# Tags explicitly included
const EXTRA_TAGS = filter(!in((:all, :fast)), TAGS)

println("Running tests")
if :all ∈ TAGS || :fast ∈ TAGS
    println("    all except: $(join(EXCLUDED_FROM_ALL, ", "));")
end
if !isempty(EXTRA_TAGS)
    println("    plus:       $(join(EXTRA_TAGS, ", "));")
end

if :all ∈ TAGS || :fast ∈ TAGS
    is_excluded(ti) = any(in(ti.tags), EXCLUDED_FROM_ALL)
    @run_package_tests filter=!is_excluded

    # TODO For now disable type piracy check, as we use that at places to patch
    #      up missing functionality. Should disable this on a more fine-grained scale.
    Aqua.test_all(DFTK, ambiguities=false, piracy=false, stale_deps=(ignore=[:Primes, ], ))
end

is_explicitly_selected(ti) = any(in(ti.tags), EXTRA_TAGS)
@run_package_tests filter=is_explicitly_selected
