using Test

@testset "Run examples" begin
    basedir = joinpath(@__DIR__, "..", "examples")
    for file in readdir(basedir)
        fullpath = joinpath(basedir, file)
        if isfile(fullpath) && endswith(file, ".jl")
            tags = open(fullpath) do fp
                for l in readlines(fp)
                    isnothing(match(r"^## tags:", l)) || return split(l[9:end])
                end
                return Vector{String}()
            end
            "long" in tags && continue
            @testset "$(file)" begin
                include(fullpath)
            end
        end
    end
end
