using Documenter
using DFTK
using Literate
import Pkg
import LibGit2

# Where to get files from and where to build them
SRCPATH = joinpath(@__DIR__, "src")
# Where the examples .md exported from .jl by Literate go
EXAMPLEPATH = joinpath(@__DIR__, "src", "literate_build")
BUILDPATH = joinpath(@__DIR__, "build")
ROOTPATH = joinpath(@__DIR__, "..")

# Python and Julia dependencies needed by the notebooks
PYDEPS = ["ase", "spglib", "pymatgen"]
JLDEPS = [
    Pkg.PackageSpec(url="https://github.com/JuliaMolSim/DFTK.jl.git",
                    rev=LibGit2.head(ROOTPATH)),  # The current DFTK
]

# Collect examples from the example index (src/index.md)
# The chosen examples are taken from the examples/ folder to be processed by Literate
EXAMPLES = [String(m[1])
            for m in match.(r"\"(examples/[^\"]+.md)\"",
                            readlines(joinpath(SRCPATH, "index.md")))
            if !isnothing(m)]

# Collect files to treat with Literate (i.e. the examples and the .jl files in the docs)
# The examples go to docs/literate_build/examples, the .jl files stay where they are
literate_files = [(src=joinpath(ROOTPATH, splitext(file)[1] * ".jl"),
                   dest=joinpath(EXAMPLEPATH, "examples"), example=true)
                  for file in EXAMPLES]
for (dir, directories, files) in walkdir(SRCPATH)
    for file in files
        if endswith(file, ".jl")
            push!(literate_files, (src=joinpath(dir, file), dest=dir, example=false))
        end
    end
end

# Function to insert badges to examples
function add_badges(str)
    badges = [
        "[![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/examples/@__NAME__.ipynb)",
        "[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/examples/@__NAME__.ipynb)"
    ]

    # Find the Header and insert the badges right below
    splitted = split(str, "\n")
    idx = findfirst(startswith.(splitted, "# # "))
    insert!(splitted, idx + 1, "#md # " * badges[1])
    insert!(splitted, idx + 2, "#md # " * badges[2])
    join(splitted, "\n")
end

# Run Literate on them all
for file in literate_files
    preprocess = file.example ? add_badges : identity
    Literate.markdown(file.src, file.dest; documenter=true, credit=false,
                      preprocess=preprocess)
    # comment that line out for a faster build
    # Literate.notebook(file.src, file.dest; credit=false)
end

# Generate the docs in BUILDPATH
makedocs(
    modules = [DFTK],
    format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://juliamolsim.github.io/DFTK.jl/stable/",
        assets = ["assets/favicon.ico"],
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
        "Getting started" => Any[
            "guide/installation.md",
            "Tutorial" => "guide/tutorial.md",
        ],
        "Examples" => [joinpath("literate_build", ex) for ex in examples],
        "Advanced topics" => Any[
            "advanced/conventions.md",
            "advanced/data_structures.md",
            "advanced/useful_formulas.md",
            "advanced/symmetries.md",
        ],
        "api.md",
        "publications.md",
        "contributing.md",
    ],
    strict = false,
)

# Dump files for managing dependencies in binder
if get(ENV, "CI", nothing) == "true"
    cd(BUILDPATH) do
        open("environment.yml", "w") do io
            print(io,
                    """
                    name: dftk
                    channels:
                      - defaults
                      - conda-forge
                    dependencies:
                    """)
            for dep in PYDEPS
                println(io, "  - " * dep)
            end
        end

        # Install Julia dependencies into build
        Pkg.activate(".")
        for dep in JLDEPS
            Pkg.add(dep)
        end
    end
    Pkg.activate(@__DIR__)  # Back to Literate / Documenter environment
end

# Deploy docs to gh-pages branch
deploydocs(
    repo = "github.com/JuliaMolSim/DFTK.jl.git",
)
