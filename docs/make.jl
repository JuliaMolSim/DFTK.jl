import LibGit2
import Pkg

# To manually generate the docs:
#
# 1. Install all python dependencies from the PYDEPS array below.
# 2. Run "julia make.jl" to generate the docs

# Set to true to disable some checks and cleanup
DEBUG = false

# Where to get files from and where to build them
SRCPATH   = joinpath(@__DIR__, "src")
BUILDPATH = joinpath(@__DIR__, "build")
ROOTPATH  = joinpath(@__DIR__, "..")
CONTINUOUS_INTEGRATION = get(ENV, "CI", nothing) == "true"
DFTKREV   = LibGit2.head(ROOTPATH)
DFTKREPO  = "github.com/JuliaMolSim/DFTK.jl.git"

# Python and Julia dependencies needed for running the notebooks
PYDEPS = ["ase", "pymatgen"]

# Setup julia dependencies for docs generation if not yet done
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
    Pkg.develop(Pkg.PackageSpec(path=ROOTPATH))
    Pkg.instantiate()
end

# Setup environment for making plots
ENV["GKS_ENCODING"] = "utf8"
ENV["GKSwstype"] = "100"
ENV["PLOTS_TEST"] = "true"

# Import packages for docs generation
using DFTK
using Documenter
using Literate

# Collect examples from the example index (src/index.md)
# The chosen examples are taken from the examples/ folder to be processed by Literate
EXAMPLES = [String(m[1])
            for m in match.(r"\"(examples/[^\"]+.md)\"",
                            readlines(joinpath(SRCPATH, "index.md")))
            if !isnothing(m)]

# Collect files to treat with Literate (i.e. the examples and the .jl files in the docs)
# The examples go to docs/literate_build/examples, the .jl files stay where they are
literate_files = [(src=joinpath(ROOTPATH, splitext(file)[1] * ".jl"),
                   dest=joinpath(SRCPATH, "examples"), example=true)
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
        "[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/examples/@__NAME__.ipynb)",
    ]

    # Find the Header and insert the badges right below
    splitted = split(str, "\n")
    idx = findfirst(startswith.(splitted, "# # "))
    idx === nothing && error("Example files must start with # #")
    insert!(splitted, idx + 1, "#md # " * badges[1])
    insert!(splitted, idx + 2, "#md # " * badges[2])
    insert!(splitted, idx + 3, "#md #")
    join(splitted, "\n")
end

# Run Literate on them all
for file in literate_files
    preprocess = file.example ? add_badges : identity
    Literate.markdown(file.src, file.dest; documenter=true, credit=false,
                      preprocess=preprocess)
    Literate.notebook(file.src, file.dest; credit=false,
                      execute=CONTINUOUS_INTEGRATION || DEBUG)
end

# Generate the docs in BUILDPATH
makedocs(
    modules = [DFTK],
    format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = CONTINUOUS_INTEGRATION,
        canonical = "https://docs.dftk.org/stable/",
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
        "school2022.md",
        "Getting started" => Any[
            "guide/installation.md",
            "Tutorial" => "guide/tutorial.md",
            "guide/input_output.md",
            "guide/parallelization.md",
            "Introduction to periodic problems" => "guide/periodic_problems.md",
            "Density-functional theory" => "guide/density_functional_theory.md",
        ],
        "Examples" => EXAMPLES,
        "Advanced topics" => Any[
            "advanced/conventions.md",
            "advanced/data_structures.md",
            "advanced/useful_formulas.md",
            "advanced/symmetries.md",
        ],
        "api.md",
        "publications.md",
    ],
    checkdocs=:exports,
    strict = !DEBUG,
)

# Dump files for managing dependencies in binder
if CONTINUOUS_INTEGRATION
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
        Pkg.add(Pkg.PackageSpec(url="https://" * DFTKREPO, rev=DFTKREV))
        cp(joinpath(@__DIR__, "Project.toml"), joinpath(BUILDPATH, "Project.toml"), force=true)
    end
    Pkg.activate(@__DIR__)  # Back to Literate / Documenter environment
end

# Deploy docs to gh-pages branch
deploydocs(; repo=DFTKREPO)

# Remove generated example files
if !DEBUG
    for file in literate_files
        base = splitext(basename(file.src))[1]
        for ext in [".ipynb", ".md"]
            rm(joinpath(file.dest, base * ext), force=true)
        end
    end
end

if !CONTINUOUS_INTEGRATION
    println("\nDocs generated, try $(joinpath(BUILDPATH, "index.html"))")
end
