using Documenter
using DFTK

makedocs(
    sitename = "DFTK",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [DFTK]
)

deploydocs(
    repo = "github.com/JuliaMolSim/DFTK.jl.git",
)
