# TODO: Temporary, explanations too scarce. To be changed with proper phonon computations.
using Test

# TODO This is not the best solution, we should use testitemrunners for all the tests,
#      but this is just to see the rebase has not caused too many issues
@testitem "Phonons" tags=[:dont_test_mpi] begin

include("lowlevel.jl")
include("pairwise.jl")
include("ewald.jl")

end
