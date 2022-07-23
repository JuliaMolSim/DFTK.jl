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

@testset "Run examples" begin
    for file in list_examples()
        @testset "$(file)" begin
            include(file)
        end
    end
end
