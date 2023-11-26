function parse_test_args()
    test_args = ARGS
    if "DFTK_TEST_ARGS" in keys(ENV) && isempty(ARGS)
        test_args = split(ENV["DFTK_TEST_ARGS"], ",")
    end
    args = isempty(test_args) ? [:all] : Symbol.(test_args)

    if :gpu in args
        base_tag = :gpu
        excluded = Symbol[]
    elseif :mpi in args
        base_tag = :mpi
        excluded = Symbol[:dont_test_mpi]
    else
        base_tag = :all
        excluded = Symbol[:gpu, :mpi]
    end
    if :fast in args
        push!(excluded, :slow)
    end
    if Sys.iswindows()
        push!(excluded, :dont_test_windows)
    end

    included = filter(!in((:all, :fast, :gpu, :mpi)), args)
    (; base_tag, excluded, included)
end
