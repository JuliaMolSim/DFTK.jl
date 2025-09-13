function parse_test_args()
    # Parsing code
    args = Symbol.(ARGS)
    if "DFTK_TEST_ARGS" in keys(ENV) && isempty(ARGS)
        args = Symbol.(split(ENV["DFTK_TEST_ARGS"], "-"))
    end

    # Figure out the base tag
    incl_excl = Dict(
        :gpu => (; included=Symbol[], excluded=[]),
        :mpi => (; included=Symbol[], excluded=[:gpu, :dont_test_mpi]),
        :all => (; included=Symbol[], excluded=[:gpu]),
    )
    base_tag = filter(in(keys(incl_excl)), args)
    if isempty(base_tag)
        base_tag = :all
    elseif length(base_tag) > 2
        error("Cannot have more than one of $(join(string.(keys(incl_excl)), ", "))")
    else
        base_tag = base_tag[1]
    end
    (; excluded, included) = incl_excl[base_tag]

    # Perform extra modifications
    if Sys.iswindows()
        push!(excluded, :dont_test_windows)
    end
    for arg in filter(!in(keys(incl_excl)), args)
        if startswith(string(arg), "no")
            push!(excluded, Symbol(string(arg)[3:end]))  # Strip "no" and add
        else
            push!(included, arg)
        end
    end

    # If there is any specifc tag to be included, this implies that not all tests are run
    if length(included) > 0
        push!(excluded, :all)
    end

    (; base_tag, excluded, included)
end
