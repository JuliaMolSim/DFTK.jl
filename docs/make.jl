using Documenter
using DFTK
using Literate

Literate.markdown("./src/index.jl", "./src/"; documenter=true)

makedocs(
    sitename = "DFTK",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [DFTK]
)

deploydocs(
    repo = "github.com/JuliaMolSim/DFTK.jl.git",
)
