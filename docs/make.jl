using Documenter
using DFTK
using Literate

# Collect examples to include in the documentation
examples = [String(m[1])
            for m in match.(r"\"(examples/[^\"]+.md)\"",
                            readlines(joinpath(@__DIR__, "src/index.md")))
            if !isnothing(m)]

# Collect files to treat with literate (i.e. the examples and the .jl files in the docs)
literate_files = [(src=joinpath(@__DIR__, "..", splitext(file)[1] * ".jl"),
                   dest=joinpath(@__DIR__, "src/examples"), is_example=true)
                  for file in examples]
for (dir, directories, files) in walkdir(joinpath(@__DIR__, "src"))
    for file in files
        if endswith(file, ".jl")
            push!(literate_files, (src=joinpath(dir, file), dest=dir, is_example=false))
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

# Run literate on them all
for file in literate_files
    preprocess = file.is_example ? add_badges : identity
    Literate.markdown(file.src, file.dest; documenter=true, credit=false,
                      preprocess=preprocess)
    Literate.notebook(file.src, file.dest; credit=false)
end

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
        "Examples" => examples,
        "Advanced topics" => Any[
            "advanced/conventions.md",
            "advanced/useful_formulas.md",
            "advanced/symmetries.md",
        ],
        "api.md",
        "publications.md",
        "contributing.md",
    ],
    strict = true,
)

deploydocs(
    repo = "github.com/JuliaMolSim/DFTK.jl.git",
)
