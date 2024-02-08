# Small payload script to actually run the tests
# This is needed to play nicely with MPI parallelised tests
#
using TestItemRunner
using Suppressor

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

function run_tests()
    output = @capture_out try
        @run_package_tests filter=dftk_testfilter verbose=true
    catch err
        Base.showerror(stderr, err, Base.catch_backtrace())
    end

    lines = split(output, "\n")
    # Print failed tests.
    println()
    for id in findall(occursin.("Test Failed", lines))
        id_context = id
        while !isempty(lines[id_context])
            println(lines[id_context])
            id_context += 1
        end
        println()
    end
    # Print the summary.
    idx = findfirst(occursin.(r"^Test Summary:", lines))
    if !isnothing(idx)
        println(join(lines[idx:end], "\n"))
    end
end

run_tests()
