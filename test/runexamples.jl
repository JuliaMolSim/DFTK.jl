using Test

function list_examples()
    res = String[]
    basedir = joinpath(@__DIR__, "..", "examples")
    for file in readdir(basedir)
        fullpath = joinpath(basedir, file)
        if isfile(fullpath) && endswith(file, ".jl")
            push!(res, fullpath)
        end
    end
    res
end

function example_tags(fullpath)
    open(fullpath) do fp
        for l in readlines(fp)
            regex = "^#src tags:"
            isnothing(match(Regex(regex), l)) || return split(l[length(regex):end])
        end
        return Vector{String}()
    end
end

@testset "Run examples" begin
    for file in list_examples()
        "long" in example_tags(file) && continue
        @testset "$(file)" begin
            include(file)
        end
    end
end
