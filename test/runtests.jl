#
# This test suite supports test arguments. For example:
#     Pkg.test("DFTK"; test_args = ["fast"])
# only runs the "fast" tests (i.e. not the expensive ones)
#     Pkg.test("DFTK"; test_args = ["gpu"])
# runs only the tests tagged as "gpu" and
#     Pkg.test("DFTK"; test_args = ["example", "all"])
# runs all tests plus the "example" tests.
#

using MPI
include("runtests_parser.jl")
(; base_tag) = parse_test_args()
runfile = joinpath(@__DIR__, "runtests_runner.jl")

# Trigger some precompilation and build steps
using ASEconvert
using CUDA

if base_tag == :mpi
    nprocs  = parse(Int, get(ENV, "DFTK_TEST_NPROCS", "$(clamp(Sys.CPU_THREADS, 2, 4))"))
    run(`$(mpiexec()) -n $nprocs $(Base.julia_cmd())
        --check-bounds=yes --depwarn=yes --project --color=yes --startup-file=no
        $runfile $ARGS`)
else
    include(runfile)
end
