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

using Test
function TestItemRunner.run_testitem(filepath, use_default_usings, setups, package_name,
                                     original_code, line, column, test_setup_module_set)
    mod = Core.eval(Main, :(module $(gensym()) end))

    if use_default_usings
        Core.eval(mod, :(using Test))

        if package_name!=""
            Core.eval(mod, :(using $(Symbol(package_name))))
        end
    end

    for m in setups
        Core.eval(mod, Expr(:using, Expr(:., :., :., nameof(test_setup_module_set.setupmodule), m)))
    end

    code = string('\n'^line, ' '^column, original_code)

    TestItemRunner.withpath(filepath) do
        # Replace the test by the current testset.
        description = Test.pop_testset().description
        @testset "$(description)" begin
            Base.invokelatest(include_string, mod, code, filepath)
        end
        Test.push_testset(Test.FallbackTestSet())  # so the parent pops nothing
    end
end

@run_package_tests filter=dftk_testfilter verbose=true
