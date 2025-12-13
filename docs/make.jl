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
        "guide/installation.md",
        "General Tutorial" => "guide/tutorial.jl",
        "guide/tutorialmath.jl",
        "guide/introductory_resources.md",
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
        "examples/hubbard.jl",
    ],
    "Response and properties" => [
        "examples/elastic_constants.jl",
        "examples/polarizability.jl",
        "examples/forwarddiff.jl",
        "examples/phonons.jl",
    ],
    "Ecosystem integration" => [
        # This concerns the discussion of interfaces, IO and integration
        # options we have
        "ecosystem/atomsbase.jl",
        "ecosystem/atomscalculators.jl",
        "ecosystem/input_output.jl",
        "ecosystem/atomistic_simulation_environment.md",
        "ecosystem/wannier.jl",
    ],
    "Tips and tricks" => [
        # Resolving convergence issues, what solver to use, improving performance or
        # reliability of calculations.
        "tricks/achieving_convergence.md",
        "tricks/parallelization.md",
        "tricks/gpu.jl",
        "tricks/scf_checkpoints.jl",
        "tricks/compute_clusters.md",
    ],
    "Solvers" => [
        "examples/custom_solvers.jl",
        "examples/scf_callbacks.jl",
        "examples/compare_solvers.jl",
        "examples/analysing_scf_convergence.jl",
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
    "Theoretical Background" => [
        "guide/periodic_problems.jl",
        "guide/discretisation.jl",
        "guide/atomic_chains.jl",
        "guide/density_functional_theory.md",
        "guide/self_consistent_field.jl",
    ],
    "Developer resources" => [
        "developer/setup.md",
        "developer/testsystem.md",
        "developer/conventions.md",
        "developer/style_guide.md",
        "developer/data_structures.md",
        "developer/useful_formulas.md",
        "developer/symmetries.md",
        "developer/gpu_computations.md",
    ],
    "api.md",
    "publications.md",
]

# .jl files to execute with Literate.jl. This can be helpful when working on
# the documentation to not run all the code examples, e.g. when working on a
# particular file or even when working just on the text or the structure of
# the documentation. When set to :ALL, everything is executed.
# JL_FILES_TO_EXECUTE = :ALL
JL_FILES_TO_EXECUTE = [
    "examples/arbitrary_floattype.jl",
    "examples/error_estimates_forces.jl",
]

# Files from the /examples folder that need to be copied over to the docs
# (typically images, input or data files etc.)
EXAMPLE_ASSETS = []  # Specify e.g. as "examples/Fe_afm.pwi"

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
    Pkg.develop(Pkg.PackageSpec(; path=ROOTPATH))
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

# Copy assets over
mkpath(joinpath(SRCPATH, "examples"))
for asset in EXAMPLE_ASSETS
    cp(joinpath(ROOTPATH, asset), joinpath(SRCPATH, asset), force=true)
end

# Collect files to treat with Literate (i.e. the examples and the .jl files in the docs)
# The examples go to docs/literate_build/examples, the .jl files stay where they are
literate_files = map(filter!(endswith(".jl"), extract_paths(PAGES))) do file
    if startswith(file, "examples/")
        (; src=joinpath(ROOTPATH, file), dest=joinpath(SRCPATH, "examples"), example=true, original_src=file)
    else
        (; src=joinpath(SRCPATH, file), dest=joinpath(SRCPATH, dirname(file)), example=false, original_src=file)
    end
end

# Function to insert badges to markdown files
function add_badges(badges)
    function preprocess(str)
        # Find the Header and insert the badges right below
        splitted = split(str, "\n")
        idx = findfirst(startswith.(splitted, "# # "))
        isnothing(idx) && error("Literate files must start with # #")
        for (i, bad) in enumerate(badges)
            insert!(splitted, idx + i, "#md # " * bad)
        end
        insert!(splitted, idx + length(badges) + 1, "#md #")
        join(splitted, "\n")
    end
end

# Run Literate on them all
if get(ENV, "DFTK_EXECUTE_CODE", false) == "true"
    @debug "Processing literate files"
    for file in literate_files
        @debug "Processing" file.src file.dest
        subfolder = relpath(file.dest, SRCPATH)
        if CONTINUOUS_INTEGRATION
            badges = [
                "[![](https://mybinder.org/badge_logo.svg)]" *
                "(@__BINDER_ROOT_URL__/$subfolder/@__NAME__.ipynb)",
                "[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)]" *
                "(@__NBVIEWER_ROOT_URL__/$subfolder/@__NAME__.ipynb)",
            ]
        else
            badges = ["Binder links to `/$subfolder/@__NAME__.ipynb`"]
        end
        execute = if JL_FILES_TO_EXECUTE == :ALL || file.original_src in JL_FILES_TO_EXECUTE
            @debug "Executing code"
            true
        else
            @debug "Not executing code"
            false
        end
        Literate.markdown(
            file.src, file.dest;
            flavor=Literate.DocumenterFlavor(),
            credit=false,
            preprocess=add_badges(badges),
            execute=execute,
            codefence="````julia" => "````",
        )
        # Literate.notebook(file.src, file.dest; credit=false,
        #                   execute=CONTINUOUS_INTEGRATION || DEBUG)
    end
else
    @info "Skipping Literate processing of files"
end

# Generate the docs in BUILDPATH
remote_args = CONTINUOUS_INTEGRATION ? (; ) : (; remotes=nothing)
mathengine  = Documenter.MathJax3(Dict(
    :tex => Dict(
        :inlineMath => [["\$","\$"], ["\\(","\\)"]],
        :tags       => "ams",
        :packages   => ["base", "ams", "autoload", "configmacros"],
        :macros     => Dict(
            :abs    => [raw"\left|#1\right|",       1],
            :norm   => [raw"\left\|#1\right\|",     1],
            :ket    => [raw"\left|#1\right\rangle", 1],
            :bra    => [raw"\left\langle#1\right|", 1],
            :braket => [raw"\left\langle#1\middle|#2\right\rangle", 2],
        ),
    ),
))

makedocs(;
    modules=[DFTK],
    format=Documenter.HTML(;
        # Use clean URLs, unless built as a "local" build
        prettyurls=CONTINUOUS_INTEGRATION,
        canonical="https://docs.dftk.org/stable/",
        edit_link="master",
        assets=["assets/favicon.ico", "assets/custom.css"],
        size_threshold=nothing,  # do not fail build if large HTML outputs
        mathengine,
    ),
    sitename = "DFTK.jl",
    authors = "Michael F. Herbst, Antoine Levitt and contributors.",
    pages=transform_to_md(PAGES),
    checkdocs=:exports,
    warnonly=DEBUG,
    remote_args...,
)

# Dump files for managing dependencies in binder
if CONTINUOUS_INTEGRATION && DFTKBRANCH == "master"
    cp(joinpath(@__DIR__, "Project.toml"), joinpath(BUILDPATH, "Project.toml"); force=true)
end

# Deploy docs to gh-pages branch
# Note: Overwrites the commit history via a force push (saves storage space)
deploydocs(; repo=DFTKREPO, devbranch="master", forcepush=true)

if !CONTINUOUS_INTEGRATION
    println("\nDocs generated, try $(joinpath(BUILDPATH, "index.html"))")
end
