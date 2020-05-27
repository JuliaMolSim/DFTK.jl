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

# Latex macros:
# https://github.com/JuliaAtoms/AtomicLevels.jl/blob/b989896f6e7f62e9213f14b5ce38dffc50bc14b1/docs/make.jl

makedocs(
    modules = [DFTK],
    format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://juliamolsim.github.io/DFTK.jl/stable/",
        # assets = ["assets/favicon.ico"],
    ),
    sitename = "DFTK.jl",
    authors = "Michael F. Herbst, Antoine Levitt and contributors.",
    linkcheck = false,  # TODO
    linkcheck_ignore = [
        # Ignore links that point to GitHub's edit pages, as they redirect to the
        # login screen and cause a warning:
        r"https://github.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/edit(.*)",
    ],
    pages = [
        "Home" => "index.md",
        # "Manual" => Any[
        #     "Guide" => "man/guide.md",
        #     "man/examples.md",
        #     "man/syntax.md",
        #     "man/doctests.md",
        #     "man/latex.md",
        #     hide("man/hosting.md", [
        #         "man/hosting/walkthrough.md"
        #     ]),
        #     "man/other-formats.md",
        # ],
        # "showcase.md",
        # "Library" => Any[
        #     "Public" => "lib/public.md",
        #     "Internals" => map(
        #         s -> "lib/internals/$(s)",
        #         sort(readdir(joinpath(@__DIR__, "src/lib/internals")))
        #     ),
        # ],
        # "contributing.md",
    ],
    strict = false,  # TODO
)

deploydocs(
    repo = "github.com/JuliaMolSim/DFTK.jl.git",
)
