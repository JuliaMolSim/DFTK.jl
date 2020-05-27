using Documenter
using DFTK
using Literate

for (dir, directories, files) in walkdir(joinpath(@__DIR__, "src"))
    for file in files
        if endswith(file, ".jl")
            Literate.markdown(joinpath(dir, file), dir;
                              documenter=true, credit=false)
        end
    end
end

makedocs(
    sitename = "DFTK",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [DFTK]
)

deploydocs(
    repo = "github.com/JuliaMolSim/DFTK.jl.git",
)
