@testitem "Run examples" tags=[:slow, :example, :dont_test_mpi] begin
    function list_examples()
        res = String[]
        for (root, dirs, files) in walkdir(joinpath(@__DIR__, "..", "examples"))
            for file in files
                if endswith(file, ".jl")
                    push!(res, joinpath(root, file))
                end
            end
        end
        res
    end

    for path in list_examples()
        dir, file = splitdir(path)
        @testset "$(file)" begin
            cd(dir) do
                include(path)
            end
        end
    end
end
