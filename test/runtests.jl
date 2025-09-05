#
# This test suite supports test arguments. For example:
#     Pkg.test("DFTK"; test_args = ["minimal"])
# only runs the "fast" tests (i.e. not the expensive ones)
#     Pkg.test("DFTK"; test_args = ["gpu"])
# runs only the tests tagged as "gpu" and
#     Pkg.test("DFTK"; test_args = ["mpi"])
# runs only MPI parallel tests (all except those with :dont_test_mpi tag)
#     Pkg.test("DFTK"; test_args = ["mpi", "minimal"])
# runs only MPI parallel tests from the :minimal subset
#     Pkg.test("DFTK"; test_args = ["noslow"])
# runs all tests except those tagges as :slow

using MPI
include("runtests_parser.jl")
(; base_tag) = parse_test_args()
runfile = joinpath(@__DIR__, "runtests_runner.jl")

# Trigger some precompilation and build steps
using ASEconvert
using CUDA
using AMDGPU
using DFTK
using Interpolations

if base_tag == :mpi
    nprocs = parse(Int, get(ENV, "DFTK_TEST_NPROCS", "$(clamp(Sys.CPU_THREADS, 2, 4))"))
    run(`$(mpiexec()) -n $nprocs $(Base.julia_cmd())
        --project --startup-file=no --compiled-modules=no
        --check-bounds=yes --depwarn=yes --color=yes
        $runfile $ARGS`)
else
    include(runfile)
end
