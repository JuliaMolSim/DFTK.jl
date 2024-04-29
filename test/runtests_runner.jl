# Small payload script to actually run the tests
# This is needed to play nicely with MPI parallelised tests
#
using TestItemRunner

include("runtests_parser.jl")
(; base_tag, excluded, included) = parse_test_args()

println("Running $base_tag tests")
if !isempty(excluded)
    println("    Excluded: $(join(excluded, ", "));")
end
if !isempty(included)
    println("    Included: $(join(included, ", "));")
end

function dftk_testfilter(ti)
    if any(in(ti.tags), included)
        return true
    elseif any(in(ti.tags), excluded)
        return false
    elseif (base_tag == :all || base_tag == :mpi) && !(:all in excluded)
        # TODO Remove the :dont_test_mpi tag and run
        # only selective mpi tests by supplying a special :mpi tag
        return true
    elseif base_tag in ti.tags
        return true
    else
        return false
    end
end
@run_package_tests filter=dftk_testfilter verbose=true
