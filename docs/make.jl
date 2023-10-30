# To manually generate the docs:
#     1. Run "julia make.jl"
#
# To add a new example to the docs:
#     1. Add the *.jl file to /examples, along with assets you require (e.g. input files,
#        images, plots etc.)
#     2. Add the example to the PAGES variable below, the asset files to EXAMPLE_ASSETS.
#
# To add a new page to the docs (not an example):
#     1. Add the file to /docs/src, along with all assets. It can be a *.jl to be processed
#        with Literate.jl and a *.md to be included as is.
#     2. Add the file to the PAGES variable below. You don't need to track the assets.

# Structure of the docs. List files as *.jl or *.md here. All files
# ending in *.jl will be processed to *.md with Literate.
PAGES = [
    "Home" => "index.md",
    "features.md",
    "Getting started" => [
        # Installing DFTK, tutorial, theoretical background
        "guide/installation.md",
        "Tutorial" => "guide/tutorial.jl",
        "guide/periodic_problems.jl",
        "guide/introductory_resources.md",
        "school2022.md",
    ],
    "Basic DFT calculations" => [
        # Ground-state DFT calculations, standard problems and modelling techniques
        # Some basic show cases; may feature integration of DFTK with other packages.
        "examples/metallic_systems.jl",
        "examples/collinear_magnetism.jl",
        "examples/convergence_study.jl",
        "examples/pseudopotentials.jl",
        "examples/supercells.jl",
        "examples/gaas_surface.jl",
        "examples/graphene.jl",
        "examples/geometry_optimization.jl",
        "examples/energy_cutoff_smearing.jl",
    ],
    "Response and properties" => [
        "examples/polarizability.jl",
        "examples/forwarddiff.jl",
        "examples/dielectric.jl",
    ],
    "Ecosystem integration" => [
        # This concerns the discussion of interfaces, IO and integration
        # options we have
        "examples/atomsbase.jl",
        "examples/input_output.jl",
        "examples/wannier.jl",
    ],
    "Tipps and tricks" => [
        # Resolving convergence issues, what solver to use, improving performance or
        # reliability of calculations.
        "tricks/parallelization.md",
        "tricks/scf_checkpoints.jl",
    ],
    "Solvers" => [
        "examples/custom_solvers.jl",
        "examples/scf_callbacks.jl",
        "examples/compare_solvers.jl",
    ],
    "Nonstandard models" => [
        "examples/gross_pitaevskii.jl",
        "examples/gross_pitaevskii_2D.jl",
        "examples/custom_potential.jl",
        "examples/cohen_bergstresser.jl",
        "examples/anyons.jl",
    ],
    "Error control" => [
        "examples/arbitrary_floattype.jl",
        "examples/error_estimates_forces.jl",
    ],
    "Developer resources" => [
        "developer/setup.md",
        "developer/conventions.md",
        "developer/data_structures.md",
        "developer/useful_formulas.md",
        "developer/symmetries.md",
        "developer/gpu_computations.md"
    ],
    "api.md",
    "publications.md",
]

# Files from the /examples folder that need to be copied over to the docs
# (typically images, input or data files etc.)
EXAMPLE_ASSETS = ["examples/Fe_afm.pwi", "examples/Si.extxyz"]

#
# Configuration and setup
#
DEBUG = false  # Set to true to disable some checks and cleanup

import LibGit2
import Pkg
# Where to get files from and where to build them
SRCPATH   = joinpath(@__DIR__, "src")
BUILDPATH = joinpath(@__DIR__, "build")
ROOTPATH  = joinpath(@__DIR__, "..")
CONTINUOUS_INTEGRATION = get(ENV, "CI", nothing) == "true"
DFTKREV    = LibGit2.head(ROOTPATH)
DFTKBRANCH = try LibGit2.branch(LibGit2.GitRepo(ROOTPATH)) catch end
DFTKGH     = "github.com/JuliaMolSim/DFTK.jl"
DFTKREPO   = DFTKGH * ".git"

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
import Artifacts

#
# Generate the docs
#

# Get list of files from PAGES
extract_paths(file::AbstractString) = [file]
extract_paths(pages::AbstractArray) = collect(Iterators.flatten(extract_paths.(pages)))
extract_paths(pair::Pair) = extract_paths(pair.second)

# Transform files to *.md
transform_to_md(file::AbstractString) = first(splitext(file)) * ".md"
transform_to_md(pages::AbstractArray) = transform_to_md.(pages)
transform_to_md(pair::Pair) = (pair.first => transform_to_md(pair.second))

# Setup Artifacts.toml system
macro artifact_str(s)
    @eval Artifacts.@artifact_str $s
end
cp(joinpath(ROOTPATH, "Artifacts.toml"), joinpath(@__DIR__, "Artifacts.toml"), force=true)

# Copy assets over
mkpath(joinpath(SRCPATH, "examples"))
for asset in EXAMPLE_ASSETS
    cp(joinpath(ROOTPATH, asset), joinpath(SRCPATH, asset), force=true)
end

# Collect files to treat with Literate (i.e. the examples and the .jl files in the docs)
# The examples go to docs/literate_build/examples, the .jl files stay where they are
literate_files = map(filter!(endswith(".jl"), extract_paths(PAGES))) do file
    if startswith(file, "examples/")
        (src=joinpath(ROOTPATH, file), dest=joinpath(SRCPATH, "examples"), example=true)
    else
        (src=joinpath(SRCPATH, file), dest=joinpath(SRCPATH, dirname(file)), example=false)
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
    Literate.markdown(file.src, file.dest;
                      flavor=Literate.DocumenterFlavor(),
                      credit=false, preprocess)
    Literate.notebook(file.src, file.dest; credit=false,
                      execute=CONTINUOUS_INTEGRATION || DEBUG)
end

# Generate the docs in BUILDPATH
makedocs(;
    modules=[DFTK],
    format=Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = CONTINUOUS_INTEGRATION,
        canonical = "https://docs.dftk.org/stable/",
        edit_link = "master",
        assets = ["assets/favicon.ico"],
        mathengine = Documenter.MathJax(Dict(:TeX => Dict(
            :Macros => Dict(
                :ket    => [raw"\left|#1\right\rangle", 1],
                :bra    => [raw"\left\langle#1\right|", 1],
                :braket => [raw"\left\langle#1\middle|#2\right\rangle", 2],
            ),
        ))),
        size_threshold=nothing,  # do not fail build if large HTML outputs
    ),
    sitename = "DFTK.jl",
    authors = "Michael F. Herbst, Antoine Levitt and contributors.",
    pages=transform_to_md(PAGES),
    checkdocs=:exports,
    warnonly=DEBUG,
)

# Dump files for managing dependencies in binder
if CONTINUOUS_INTEGRATION && DFTKBRANCH == "master"
    cp(joinpath(@__DIR__, "Project.toml"),   joinpath(BUILDPATH, "Project.toml");   force=true)
    cp(joinpath(ROOTPATH, "Artifacts.toml"), joinpath(BUILDPATH, "Artifacts.toml"); force=true)
end

# Deploy docs to gh-pages branch
deploydocs(; repo=DFTKREPO, devbranch="master")

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
